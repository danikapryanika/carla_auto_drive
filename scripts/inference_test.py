"""
CILDriveNet — тестовый инференс с трафиком.

Отличия от inference.py:
  - Спавнится N NPC-машин под управлением TM
  - Датчик столкновения: при аварии машина телепортируется на новую точку и продолжает
  - Счётчик столкновений выводится в HUD
  - Автоматический respawn — тест продолжается даже после аварии

Управление:
  Q — выход
  R — ручной respawn (телепорт в новую точку)

Запуск:
    python scripts/inference_test.py
    python scripts/inference_test.py --ckpt checkpoints/best_model.pth --town Town03
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
    CMD_NAMES,
)

# ── Конфиг ────────────────────────────────────────────────────────────────────

FPS            = 20
DELTA_SEC      = 1.0 / FPS
TM_PORT        = 8000

TARGET_SPD     = 30.0   # км/ч — крейсерская скорость
STEER_SM       = 0.35   # сглаживание руля
THROT_SM       = 0.60   # сглаживание газа
NPC_COUNT      = 30     # машин трафика

JUNCTION_DECEL_M = 28.0
JUNCTION_SPD     = 12.0  # км/ч — скорость въезда на перекрёсток

RECORD   = True
OUT_VIDEO = "demo_cil_test.mp4"

RESPAWN_AFTER_COLLISION = True   # автоматически возобновлять после аварии
RESPAWN_PAUSE_TICKS     = 60     # тиков паузы перед respawn (3 сек)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, IMG_SIZE)
    img = (img - IMG_MEAN) / IMG_STD
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


# ── Junction lookahead ────────────────────────────────────────────────────────

def junction_ahead(vehicle, world_map, lookahead_m=28.0) -> bool:
    import carla
    loc = vehicle.get_location()
    wp  = world_map.get_waypoint(loc, project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
    if wp is None:
        return False
    if wp.is_junction:
        return True
    step = lookahead_m / 4.0
    cur  = wp
    for _ in range(4):
        nxt = cur.next(step)
        if not nxt:
            break
        cur = nxt[0]
        if cur.is_junction:
            return True
    return False


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(frame, steer, throttle, speed, cmd, braking, collisions, respawns):
    h, _ = frame.shape[:2]

    # Полупрозрачный фон под текстом
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"CMD:  {CMD_NAMES[cmd]}",
                (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, f"STR:  {steer:+.3f}   THR: {throttle:.3f}",
                (10, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
    cv2.putText(frame, f"SPD:  {speed:.1f} km/h",
                (10, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 200, 0), 2)
    cv2.putText(frame, f"COLL: {collisions}   RESP: {respawns}",
                (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    if braking:
        cv2.putText(frame, "!! OBSTACLE !!",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.putText(frame, "CILDriveNet  [Q=quit  R=respawn]",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


# ── Respawn эго-машины ────────────────────────────────────────────────────────

def respawn_vehicle(vehicle, world, spawn_points, camera, lidar_sensor,
                    img_queue, lidar_queue):
    """Телепортировать машину на свободную точку, сбросить очереди."""
    import carla

    # Ищем spawn-точку, рядом с которой нет других машин
    actors = world.get_actors().filter('vehicle.*')
    actor_locs = [a.get_location() for a in actors if a.id != vehicle.id]

    for sp in random.sample(spawn_points, min(30, len(spawn_points))):
        # Проверяем что в радиусе 8 м нет других машин
        clear = all(
            sp.location.distance(loc) > 8.0
            for loc in actor_locs
        )
        if clear:
            vehicle.set_transform(sp)
            break

    # Сбрасываем очереди сенсоров
    for q in (img_queue, lidar_queue):
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break

    # Даём физике устояться
    for _ in range(30):
        world.tick()


# ── Спавн NPC ─────────────────────────────────────────────────────────────────

def spawn_npcs(world, tm, count, spawn_points, bp_lib):
    npc_bps = [bp for bp in bp_lib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]
    npcs, used = [], set()
    for _ in range(count * 6):
        if len(npcs) >= count:
            break
        sp  = random.choice(spawn_points)
        key = (round(sp.location.x, 1), round(sp.location.y, 1))
        if key in used:
            continue
        used.add(key)
        bp = random.choice(npc_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True, TM_PORT)
            npcs.append(npc)
    return npcs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    import carla

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Загрузка модели ───────────────────────────────────────────────────────
    model = CILDriveNet().to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Модель загружена  (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")

    # ── CARLA ─────────────────────────────────────────────────────────────────
    print("Подключение к CARLA …")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    world  = client.get_world()

    current = world.get_map().name.split('/')[-1]
    if current != args.town:
        print(f"   {current} → {args.town} …")
        world = client.load_world(args.town)
        time.sleep(3.0)
        world = client.get_world()
    print(f"   Карта: {world.get_map().name}")

    world_map = world.get_map()
    settings  = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = DELTA_SEC
    world.apply_settings(settings)

    # ── TrafficManager ────────────────────────────────────────────────────────
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(False)
    tm.set_global_distance_to_leading_vehicle(2.5)

    bp_lib       = world.get_blueprint_library()
    spawn_points = world_map.get_spawn_points()

    # ── Спавн NPC ─────────────────────────────────────────────────────────────
    print(f"Спавн {NPC_COUNT} NPC …")
    npcs = spawn_npcs(world, tm, NPC_COUNT, spawn_points, bp_lib)
    print(f"NPC: {len(npcs)}")

    # ── Эго-машина ────────────────────────────────────────────────────────────
    v_bp    = bp_lib.find('vehicle.tesla.model3')
    vehicle = None
    for sp in random.sample(spawn_points, min(20, len(spawn_points))):
        vehicle = world.try_spawn_actor(v_bp, sp)
        if vehicle:
            break
    if not vehicle:
        print("Не удалось заспавнить эго-машину.")
        return

    # Эго-машина управляется моделью, НЕ автопилотом
    vehicle.set_autopilot(False)
    print(f"Эго-машина id={vehicle.id}")

    # ── Камера ────────────────────────────────────────────────────────────────
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '90')
    cam_bp.set_attribute('enable_postprocess_effects', 'False')
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=2.5, z=1.0), carla.Rotation(yaw=0)),
        attach_to=vehicle)
    img_queue = queue.Queue(maxsize=2)
    def _cam_cb(img):
        if img_queue.full():
            try: img_queue.get_nowait()
            except queue.Empty: pass
        img_queue.put_nowait(img)
    camera.listen(_cam_cb)

    # ── LIDAR ─────────────────────────────────────────────────────────────────
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels',           '32')
    lidar_bp.set_attribute('range',              '40')
    lidar_bp.set_attribute('points_per_second',  '56000')
    lidar_bp.set_attribute('rotation_frequency', str(FPS))
    lidar_bp.set_attribute('upper_fov',          '10.0')
    lidar_bp.set_attribute('lower_fov',          '-30.0')
    lidar_sensor = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=0.0, z=2.4)),
        attach_to=vehicle)
    lidar_queue = queue.Queue(maxsize=2)
    def _lidar_cb(data):
        if lidar_queue.full():
            try: lidar_queue.get_nowait()
            except queue.Empty: pass
        lidar_queue.put_nowait(data)
    lidar_sensor.listen(_lidar_cb)

    # ── Датчик препятствий (obstacle) ─────────────────────────────────────────
    obs_bp = bp_lib.find('sensor.other.obstacle')
    obs_bp.set_attribute('distance',      '12')
    obs_bp.set_attribute('hit_radius',    '0.5')
    obs_bp.set_attribute('only_dynamics', 'False')
    obs_sensor = world.spawn_actor(
        obs_bp, carla.Transform(), attach_to=vehicle)
    obstacle_flag = {'on': False}
    def _obs_cb(event):
        if 'vehicle' in event.other_actor.type_id or \
           'walker' in event.other_actor.type_id:
            obstacle_flag['on'] = True
    obs_sensor.listen(_obs_cb)

    # ── Датчик столкновения ───────────────────────────────────────────────────
    coll_bp = bp_lib.find('sensor.other.collision')
    coll_sensor = world.spawn_actor(
        coll_bp, carla.Transform(), attach_to=vehicle)
    collision_flag = {'hit': False}
    def _coll_cb(event):
        collision_flag['hit'] = True
    coll_sensor.listen(_coll_cb)

    # ── Видео ─────────────────────────────────────────────────────────────────
    vw = None
    if RECORD:
        vw = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'),
                             FPS, (640, 480))

    # ── Прогрев ───────────────────────────────────────────────────────────────
    print("Прогрев (60 тиков) …", end="", flush=True)
    for _ in range(60):
        world.tick()
    print(" готово.\n")
    print("Тест запущен.  Q — выход,  R — ручной respawn\n")

    prev_steer = 0.0
    prev_throt = 0.0
    collisions = 0
    respawns   = 0
    frame_cnt  = 0
    GRACE      = 40   # тиков после спавна — не реагируем на препятствия
    gru_h      = None  # скрытое состояние GRU; сбрасывается при respawn

    try:
        while True:
            world.tick()

            # ── Получить кадры ────────────────────────────────────────────────
            try:
                cam_data   = img_queue.get(timeout=0.5)
                lidar_data = lidar_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            raw      = np.frombuffer(cam_data.raw_data, dtype=np.uint8)
            img_bgr  = cv2.cvtColor(
                raw.reshape(480, 640, 4)[:, :, :3], cv2.COLOR_RGB2BGR)
            pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
            bev = lidar_to_bev(pts)

            # ── Состояние машины ──────────────────────────────────────────────
            vel      = vehicle.get_velocity()
            speed    = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            spd_norm = float(np.clip(speed / MAX_SPEED_KMH, 0.0, 1.0))
            cmd      = get_command(vehicle, world_map)

            # ── Столкновение → respawn ────────────────────────────────────────
            if collision_flag['hit'] and RESPAWN_AFTER_COLLISION:
                collision_flag['hit'] = False
                collisions += 1
                tqdm_print = f"Столкновение #{collisions} → respawn через 3 сек"
                print(tqdm_print)

                # Покажем кадр аварии
                cv2.putText(img_bgr, f"COLLISION #{collisions}",
                            (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 0, 255), 4)
                if vw: vw.write(img_bgr)
                cv2.imshow("CILDriveNet TEST", img_bgr)
                cv2.waitKey(1)

                # Пауза перед respawn
                for _ in range(RESPAWN_PAUSE_TICKS):
                    world.tick()

                respawn_vehicle(vehicle, world, spawn_points,
                                camera, lidar_sensor, img_queue, lidar_queue)
                respawns  += 1
                frame_cnt  = 0
                prev_steer = 0.0
                prev_throt = 0.0
                gru_h      = None  # сброс памяти после аварии
                continue

            # ── Инференс модели ───────────────────────────────────────────────
            with torch.no_grad():
                img_t   = preprocess_image(img_bgr).to(device)
                bev_t   = torch.from_numpy(bev).unsqueeze(0).to(device)
                spd_t   = torch.tensor([[spd_norm]], dtype=torch.float32).to(device)
                # GRU: передаём скрытое состояние между тиками
                actions, _, gru_h = model(img_t, bev_t, spd_t, h=gru_h)
                steer_raw, throt_raw = actions[0, cmd].cpu().numpy()

            steer_raw = float(steer_raw)
            throt_raw = float(throt_raw)

            # ── Сглаживание ───────────────────────────────────────────────────
            steer = STEER_SM * prev_steer + (1 - STEER_SM) * steer_raw
            throt = THROT_SM * prev_throt + (1 - THROT_SM) * throt_raw

            # ── Ограничение скорости ──────────────────────────────────────────
            if speed > TARGET_SPD:
                excess = (speed - TARGET_SPD) / TARGET_SPD
                throt  = max(0.0, throt * (1.0 - excess))

            steer = float(np.clip(steer, -1.0, 1.0))
            throt = float(np.clip(throt,  0.0, 1.0))
            brake = 0.0

            # ── Торможение перед перекрёстком ─────────────────────────────────
            if junction_ahead(vehicle, world_map, JUNCTION_DECEL_M):
                throt = min(throt, 0.30)
                if speed > JUNCTION_SPD:
                    brake = 0.20

            # ── Двухуровневая защита от столкновений ──────────────────────────
            in_grace   = frame_cnt < GRACE
            obs_hit    = obstacle_flag['on'] and not in_grace
            lidar_hit  = obstacle_ahead(pts)  and not in_grace
            obstacle_flag['on'] = False

            emerg = obs_hit or lidar_hit
            if emerg:
                throt = 0.0
                brake = 0.8

            # Минимальный газ при малой скорости (не даём встать)
            if speed < 2.0 and not emerg:
                throt = max(throt, 0.15)

            prev_steer = steer
            prev_throt = throt

            # ── Применить управление ──────────────────────────────────────────
            vehicle.apply_control(carla.VehicleControl(
                throttle=throt, steer=steer, brake=brake,
                manual_gear_shift=False))

            # ── HUD + дисплей ─────────────────────────────────────────────────
            draw_hud(img_bgr, steer, throt, speed, cmd, emerg, collisions, respawns)
            if vw:
                vw.write(img_bgr)
            cv2.imshow("CILDriveNet TEST", img_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nВыход.")
                break
            elif key == ord('r'):
                print("Ручной respawn")
                respawn_vehicle(vehicle, world, spawn_points,
                                camera, lidar_sensor, img_queue, lidar_queue)
                respawns  += 1
                frame_cnt  = 0
                prev_steer = prev_throt = 0.0
                gru_h      = None

            frame_cnt += 1
            if frame_cnt % 200 == 0:
                print(f"  [{frame_cnt:5d}] CMD={CMD_NAMES[cmd]:8s} "
                      f"S={steer:+.3f}  T={throt:.3f}  "
                      f"V={speed:.1f} km/h  "
                      f"COLL={collisions}  RESP={respawns}")

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop();      camera.destroy()
        lidar_sensor.stop();lidar_sensor.destroy()
        obs_sensor.stop();  obs_sensor.destroy()
        coll_sensor.stop(); coll_sensor.destroy()
        vehicle.destroy()
        for npc in npcs:
            try: npc.destroy()
            except Exception: pass
        tm.set_synchronous_mode(False)
        if vw:
            vw.release()
            print(f"Видео: {OUT_VIDEO}")
        cv2.destroyAllWindows()
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)
        print("Готово.")
        print(f"   Итого: столкновений={collisions}  respawn-ов={respawns}  кадров={frame_cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/best_model.pth')
    parser.add_argument('--town', default='Town10HD')
    main(parser.parse_args())
