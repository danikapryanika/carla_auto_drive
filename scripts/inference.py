"""
CILDriveNet inference on CARLA Town01.

The model receives:
  - resized+normalised camera frame
  - LIDAR bird's-eye-view
  - normalised vehicle speed
  - navigation command inferred from the road geometry ahead

Control outputs are smoothed and a two-layer safety brake is applied:
  1. sensor.other.obstacle — CARLA collision sensor (vehicles & pedestrians)
  2. obstacle_ahead(pts)   — LIDAR point-cloud fallback for static objects

Usage:
    python scripts/inference.py
    python scripts/inference.py --ckpt checkpoints/best_model.pth --town Town01
"""

import os
import sys
import time
import queue
import random
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import CILDriveNet
from utils import (
    lidar_to_bev, get_command, obstacle_ahead,
    IMG_MEAN, IMG_STD, IMG_SIZE, MAX_SPEED_KMH,
    CMD_NAMES, CMD_FOLLOW, CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT,
    BEV_SIZE, HEIGHT_SLICES,
)

# ── Config ────────────────────────────────────────────────────────────────────
FPS        = 20
DELTA_SEC  = 1.0 / FPS
TARGET_SPD = 30.0   # km/h cruise speed
STEER_SM   = 0.35   # сглаживание руля (меньше → быстрее реакция)
THROT_SM   = 0.65   # throttle smoothing
RECORD     = True
OUT_VIDEO  = "demo_cil.mp4"

# Pre-junction deceleration — тормозим за 28 м, въезжаем на ~14 км/ч.
# get_command() теперь активирует LEFT/RIGHT уже с 22 м, так что
# модель успевает начать поворот при низкой скорости.
JUNCTION_DECEL_M = 28.0
JUNCTION_SPD     = 14.0


# ── Junction lookahead ────────────────────────────────────────────────────────

def _junction_ahead(vehicle, world_map, lookahead_m: float = 28.0) -> bool:
    """
    True если впереди в радиусе lookahead_m метров есть перекрёсток.
    Проверяем несколько точек вдоль дороги, потому что wp.next(d) при большом d
    может «проскочить» через короткие junction-зоны.
    """
    import carla
    loc = vehicle.get_location()
    wp  = world_map.get_waypoint(loc, project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
    if wp is None:
        return False
    if wp.is_junction:
        return True
    step = lookahead_m / 4.0          # 4 промежуточные точки
    cur  = wp
    for _ in range(4):
        nxt = cur.next(step)
        if not nxt:
            break
        cur = nxt[0]
        if cur.is_junction:
            return True
    return False


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


# ── HUD overlay ───────────────────────────────────────────────────────────────

def draw_hud(frame, steer, throttle, speed, cmd, braking):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"CMD: {CMD_NAMES[cmd]}",      (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"S: {steer:+.3f}  T: {throttle:.3f}",
                (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"V: {speed:.1f} km/h",        (10, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    if braking:
        cv2.putText(frame, "!! OBSTACLE !!",           (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    cv2.putText(frame, "CILDriveNet v2 (Town01)",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = CILDriveNet().to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded  (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
    if 'cmd_losses' in ckpt:
        cl = ckpt['cmd_losses']
        print(f"   Branch losses — L:{cl[1]:.4f}  R:{cl[2]:.4f}  Str:{cl[3]:.4f}")

    import carla

    print("Connecting to CARLA ...")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    world  = client.get_world()

    current = world.get_map().name.split('/')[-1]
    print(f"   Current map: {current}")
    if args.town and current != args.town:
        print(f"   Switching {current} -> {args.town} ...")
        world = client.load_world(args.town)
        import time; time.sleep(3.0)
        world = client.get_world()
        print(f"   Map loaded: {world.get_map().name}")

    world_map = world.get_map()
    settings  = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = DELTA_SEC
    world.apply_settings(settings)

    bp_lib       = world.get_blueprint_library()
    spawn_points = world_map.get_spawn_points()

    # Ego vehicle
    v_bp    = bp_lib.find('vehicle.tesla.model3')
    vehicle = None
    for sp in random.sample(spawn_points, min(15, len(spawn_points))):
        vehicle = world.try_spawn_actor(v_bp, sp)
        if vehicle:
            break
    if not vehicle:
        print("Could not spawn vehicle.")
        return
    vehicle.set_autopilot(False)

    # Camera — параметры должны совпадать с collect_data.py
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    cam_bp.set_attribute('enable_postprocess_effects', 'False')
    cam = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=2.5, z=1.0), carla.Rotation(yaw=0)),
        attach_to=vehicle
    )
    cam_q = queue.Queue(maxsize=2)
    def _cam_cb(img):
        if cam_q.full():
            try: cam_q.get_nowait()
            except queue.Empty: pass
        cam_q.put_nowait(img)
    cam.listen(_cam_cb)

    # LIDAR — идентичные параметры collect_data.py (иначе BEV будет другим)
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels',           '32')
    lidar_bp.set_attribute('range',              '40')
    lidar_bp.set_attribute('points_per_second',  '56000')
    lidar_bp.set_attribute('rotation_frequency', str(FPS))
    lidar_bp.set_attribute('upper_fov',          '10.0')
    lidar_bp.set_attribute('lower_fov',          '-30.0')
    lidar = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=0.0, z=2.4)),
        attach_to=vehicle
    )
    lidar_q = queue.Queue(maxsize=2)
    def _lidar_cb(data):
        if lidar_q.full():
            try: lidar_q.get_nowait()
            except queue.Empty: pass
        lidar_q.put_nowait(data)
    lidar.listen(_lidar_cb)

    # ── Obstacle sensor (sensor.other.obstacle) ───────────────────────────────
    # Приоритетный датчик препятствий: работает по физическому лучу, не по LIDAR.
    # Дополняет LIDAR-based obstacle_ahead() для статических объектов.
    obs_bp = bp_lib.find('sensor.other.obstacle')
    obs_bp.set_attribute('distance',      '15')    # дальность обнаружения, м
    obs_bp.set_attribute('hit_radius',    '0.5')   # радиус луча
    obs_bp.set_attribute('only_dynamics', 'False') # ловим и статику, и динамику
    obs_sensor = world.spawn_actor(
        obs_bp,
        carla.Transform(),   # без смещения — на позиции машины
        attach_to=vehicle)

    # Простой thread-safe флаг; сбрасывается каждые N кадров
    obstacle_flag = {'detected': False}

    def _obs_cb(event):
        atype = event.other_actor.type_id
        if 'vehicle' in atype or 'walker' in atype:
            obstacle_flag['detected'] = True

    obs_sensor.listen(_obs_cb)

    # Video writer
    vw = None
    if RECORD:
        vw = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'),
                             FPS, (640, 480))

    # Warm-up
    for _ in range(30):
        world.tick()

    prev_steer  = 0.0
    prev_throt  = 0.0
    frame_cnt   = 0
    OBSTACLE_GRACE = 60  # первые 3 секунды (60 кадров) пропускаем obstacle check

    print("Inference running.  Press 'q' to quit.")

    try:
        while True:
            world.tick()

            try:
                cam_data   = cam_q.get(timeout=0.5)
                lidar_data = lidar_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── Sensor data ─────────────────────────────────────────────────
            raw    = np.frombuffer(cam_data.raw_data, dtype=np.uint8)
            img_bgr = cv2.cvtColor(
                raw.reshape(480, 640, 4)[:, :, :3], cv2.COLOR_RGB2BGR
            )
            pts    = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
            bev    = lidar_to_bev(pts)

            # ── Vehicle state ───────────────────────────────────────────────
            vel      = vehicle.get_velocity()
            speed    = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            spd_norm = float(np.clip(speed / MAX_SPEED_KMH, 0.0, 1.0))
            cmd      = get_command(vehicle, world_map)

            # ── Model inference ─────────────────────────────────────────────
            with torch.no_grad():
                img_t   = preprocess_image(img_bgr).to(device)
                lidar_t = torch.from_numpy(bev).unsqueeze(0).to(device)
                spd_t   = torch.tensor([[spd_norm]], dtype=torch.float32).to(device)

                all_branches, _ = model(img_t, lidar_t, spd_t)  # (1,4,2); speed_pred ignored
                steer_raw, throt_raw = all_branches[0, cmd].cpu().numpy()

            steer_raw = float(steer_raw)
            throt_raw = float(throt_raw)

            # ── Smooth control ──────────────────────────────────────────────
            steer = STEER_SM * prev_steer + (1 - STEER_SM) * steer_raw
            throt = THROT_SM * prev_throt + (1 - THROT_SM) * throt_raw

            # Speed governor (крейсерская скорость)
            if speed > TARGET_SPD:
                excess = (speed - TARGET_SPD) / TARGET_SPD
                throt  = max(0.0, throt * (1.0 - excess))

            steer = float(np.clip(steer, -1.0,  1.0))
            throt = float(np.clip(throt,  0.0,  1.0))
            brake = 0.0

            # ── Торможение перед перекрёстком ───────────────────────────────
            junction_near = _junction_ahead(vehicle, world_map, JUNCTION_DECEL_M)
            if junction_near:
                throt = min(throt, 0.30)
                if speed > JUNCTION_SPD:
                    brake = 0.25

            # ── Safety brake: obstacle sensor + LIDAR fallback ───────────────
            obs_sensor_hit = obstacle_flag['detected']
            obstacle_flag['detected'] = False
            lidar_hit = obstacle_ahead(pts)

            # grace period: первые OBSTACLE_GRACE кадров не тормозим по датчикам —
            # на спавне LIDAR часто видит столбы/знаки и даёт ложный brake=0.8
            emerg = (obs_sensor_hit or lidar_hit) and (frame_cnt >= OBSTACLE_GRACE)
            if emerg:
                throt = 0.0
                brake = 0.8

            # при скорости < 2 км/ч модель сама не трогается (throttle bias низкий)
            if speed < 2.0 and not emerg:
                throt = max(throt, 0.15)

            if frame_cnt < 5:
                print(f"[DBG #{frame_cnt}] cmd={CMD_NAMES[cmd]} "
                      f"steer_raw={steer_raw:+.3f} throt_raw={throt_raw:.3f} "
                      f"throt={throt:.3f} speed={speed:.1f} "
                      f"obs_sensor={obs_sensor_hit} lidar={lidar_hit} emerg={emerg}")

            prev_steer = steer
            prev_throt = throt

            # Apply
            vehicle.apply_control(carla.VehicleControl(
                throttle=throt, steer=steer, brake=brake,
                manual_gear_shift=False,
            ))

            # ── Display ─────────────────────────────────────────────────────
            draw_hud(img_bgr, steer, throt, speed, cmd, emerg)
            if vw:
                vw.write(img_bgr)
            cv2.imshow("CIL Navigation", img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_cnt += 1
            if frame_cnt % 100 == 0:
                print(f"[{frame_cnt:5d}] CMD={CMD_NAMES[cmd]:8s} "
                      f"S={steer:+.3f}  T={throt:.3f}  V={speed:.1f} km/h")

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop();        cam.destroy()
        lidar.stop();      lidar.destroy()
        obs_sensor.stop(); obs_sensor.destroy()
        vehicle.destroy()
        if vw:
            vw.release()
            print(f"Video saved to {OUT_VIDEO}")
        cv2.destroyAllWindows()
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/best_model.pth")
    parser.add_argument("--town", default="Town10HD")
    main(parser.parse_args())
