"""
Сбор данных для CIL-модели — три камеры + LiDAR.

Три камеры (центр, левая -0.5м, правая +0.5м) на каждый тик дают три строки в CSV
с разными значениями steer. Это решает главную проблему обучения — covariate shift:
модель получает примеры того, как выглядит дорога при смещении от центра полосы
и какой руль нужен чтобы вернуться.

  Левая камера  (y=-0.5м): steer = original + CORRECTION  (+0.10)
                             Дорога смещена вправо → нужно подруливать вправо
  Центральная   (y= 0.0м): steer = original
  Правая камера (y=+0.5м): steer = original - CORRECTION  (-0.10)
                             Дорога смещена влево → нужно подруливать влево

LiDAR один (центральный) — одна .npy-запись на тик, на неё ссылаются все три строки CSV.

На каждый тик сохраняется:
  • 3 PNG-кадра (images/)
  • 1 NPY-файл лидара (lidar/)
  • 3 строки в labels.csv

При TOTAL_FRAMES=24000 нужно ~8000 тиков реальной езды (~7 минут).
"""

import os, sys, csv, time, queue, random
import numpy as np
import cv2
from tqdm import tqdm

# ── Конфиг ───────────────────────────────────────────────────────────────────

TOWN             = "Town10HD"
TOTAL_FRAMES     = 24_000       # итоговых строк в CSV (÷3 = тиков езды)
IMAGE_SIZE       = (640, 480)
SAVE_SIZE        = (224, 224)
FPS              = 20
DELTA_SEC        = 1.0 / FPS
NPC_COUNT        = 25
WEATHER_INTERVAL = 600          # тиков между сменой погоды
TM_PORT          = 8000

TARGET_SPEED     = 30.0         # км/ч
WHEELBASE        = 2.875        # м, Tesla Model 3
MAX_STEER_RAD    = np.deg2rad(70.0)

WARMUP_TICKS     = 200
MIN_SPEED_KMH    = 1.0
STUCK_KMH        = 1.5
STUCK_TICKS_MAX  = 300          # 15 сек → перезапуск автопилота (без телепорта)

STEER_CORRECTION = 0.10         # коррекция руля для боковых камер

# (название, смещение y в м, коррекция steer)
# y<0 = влево, y>0 = вправо (система координат CARLA: y=right)
CAM_CONFIGS = [
    ('C',  0.0,  0.00),   # центральная камера — без коррекции
    ('L', -0.5, +STEER_CORRECTION),  # левая — нужно подруливать вправо
    ('R', +0.5, -STEER_CORRECTION),  # правая — нужно подруливать влево
]

FRESH_START = True

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "train")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LIDAR_DIR  = os.path.join(OUTPUT_DIR, "lidar")
CSV_PATH   = os.path.join(OUTPUT_DIR, "labels.csv")

if FRESH_START:
    import shutil
    for _d in (IMAGES_DIR, LIDAR_DIR):
        if os.path.exists(_d): shutil.rmtree(_d)
    if os.path.exists(CSV_PATH): os.remove(CSV_PATH)
    print("   Старый датасет удалён (FRESH_START=True).")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LIDAR_DIR,  exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from utils import lidar_to_bev, get_command, CMD_NAMES, HEIGHT_SLICES, BEV_SIZE


# ── Delta-based управление ────────────────────────────────────────────────────

def compute_controls(loc_before, yaw_before, loc_after, yaw_after):
    dx, dy    = loc_after.x - loc_before.x, loc_after.y - loc_before.y
    speed_ms  = np.sqrt(dx**2 + dy**2) / DELTA_SEC
    speed_kmh = float(speed_ms * 3.6)

    yaw_diff = ((yaw_after - yaw_before) + 180) % 360 - 180
    if speed_kmh > 1.0:
        yaw_rate = (yaw_diff * np.pi / 180.0) / DELTA_SEC
        delta    = np.arctan(yaw_rate * WHEELBASE / speed_ms)
        steer    = float(np.clip(delta / MAX_STEER_RAD, -1.0, 1.0))
    else:
        steer = 0.0

    if speed_kmh < TARGET_SPEED * 0.85:
        throttle, brake = 0.70, 0.0
    elif speed_kmh > TARGET_SPEED * 1.10:
        throttle, brake = 0.0,  0.25
    else:
        throttle, brake = 0.45, 0.0

    return speed_kmh, steer, throttle, brake


# ── Погода ────────────────────────────────────────────────────────────────────

def apply_weather(world):
    import carla
    world.set_weather(carla.WeatherParameters(
        cloudiness         = np.random.uniform(5,  70),
        precipitation      = np.random.uniform(0,  20),
        wind_intensity     = np.random.uniform(5,  40),
        sun_azimuth_angle  = np.random.uniform(0,  360),
        sun_altitude_angle = np.random.uniform(35, 75),
        fog_density        = np.random.uniform(0,  5),
        wetness            = np.random.uniform(0,  25),
    ))


# ── NPC ──────────────────────────────────────────────────────────────────────

def spawn_npcs(world, tm, count, spawn_points, bp_lib):
    npc_bps = bp_lib.filter('vehicle.*')
    npcs, used = [], set()
    for _ in range(count * 5):
        if len(npcs) >= count: break
        sp  = random.choice(spawn_points)
        key = (round(sp.location.x, 1), round(sp.location.y, 1))
        if key in used: continue
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

def main():
    import carla

    np.random.seed(42); random.seed(42)

    print(f"Загрузка {TOWN} …")
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)
    world  = client.get_world()

    current = world.get_map().name.split('/')[-1]
    if current != TOWN:
        print(f"   {current} → {TOWN} …")
        client.load_world(TOWN); time.sleep(4.0)
        world = client.get_world()
    world_map = world.get_map()

    # ── Синхронный режим ─────────────────────────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = DELTA_SEC
    world.apply_settings(settings)

    # ── TrafficManager ───────────────────────────────────────────────────────
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_hybrid_physics_mode(False)
    try: tm.set_respawn_dormant_vehicles(True)
    except AttributeError: pass

    # ── Эго-машина ───────────────────────────────────────────────────────────
    bp_lib       = world.get_blueprint_library()
    vehicle_bp   = bp_lib.find('vehicle.tesla.model3')
    spawn_points = world_map.get_spawn_points()

    vehicle = None
    for sp in random.sample(spawn_points, min(20, len(spawn_points))):
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle: break
    if not vehicle:
        print("Не удалось заспавнить машину."); return

    vehicle.set_autopilot(True, TM_PORT)
    tm.vehicle_percentage_speed_difference(vehicle, -50)
    print(f"Эго-машина id={vehicle.id}")

    # ── NPC ──────────────────────────────────────────────────────────────────
    print(f"Спавн {NPC_COUNT} NPC …")
    npcs = spawn_npcs(world, tm, NPC_COUNT, spawn_points, bp_lib)
    print(f"NPC: {len(npcs)}")

    # ── Три камеры ────────────────────────────────────────────────────────────
    cam_actors = {}
    cam_queues = {}
    for name, y_offset, _ in CAM_CONFIGS:
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMAGE_SIZE[0]))
        cam_bp.set_attribute('image_size_y', str(IMAGE_SIZE[1]))
        cam_bp.set_attribute('fov', '90')
        cam_bp.set_attribute('enable_postprocess_effects', 'False')
        cam = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=2.5, y=y_offset, z=1.0),
                            carla.Rotation(yaw=0)),
            attach_to=vehicle)
        q = queue.Queue(maxsize=5)
        cam.listen(lambda img, q=q: q.put(img) if not q.full() else None)
        cam_actors[name] = cam
        cam_queues[name]  = q
    print(f"Камеры: {list(cam_actors.keys())} (центр / левая / правая)")

    # ── LiDAR (один, центральный) ─────────────────────────────────────────────
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
    lidar_queue = queue.Queue(maxsize=5)
    lidar_sensor.listen(lambda d: lidar_queue.put(d) if not lidar_queue.full() else None)

    # ── Начальная погода ─────────────────────────────────────────────────────
    apply_weather(world)
    weather_tick = 0

    # ── Прогрев ──────────────────────────────────────────────────────────────
    print(f"Прогрев ({WARMUP_TICKS} тиков) …", end="", flush=True)
    for _ in range(WARMUP_TICKS): world.tick()
    for q in list(cam_queues.values()) + [lidar_queue]:
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break
    print(" готово.")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_file = open(CSV_PATH, 'w', newline='', encoding='utf-8')
    writer   = csv.writer(csv_file)
    writer.writerow(['frame_id', 'image_path', 'lidar_path',
                     'steer', 'throttle', 'brake', 'speed_kmh', 'command', 'cam'])

    EMPTY_BEV = np.zeros((HEIGHT_SLICES + 1, BEV_SIZE, BEV_SIZE), dtype=np.float32)

    ticks_needed = TOTAL_FRAMES // len(CAM_CONFIGS)
    print(f"\nСбор {TOTAL_FRAMES} строк = {ticks_needed} тиков езды | {TOWN}")
    print(f"   Камеры: C / L(+{STEER_CORRECTION:.2f}) / R(-{STEER_CORRECTION:.2f})\n")

    cmd_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    frame_idx  = 0   # глобальный счётчик кадров (строк CSV)
    tick_idx   = 0   # счётчик тиков с данными лидара (для имени .npy)
    saved      = 0   # сколько строк сохранено
    stuck_ticks = 0
    diag_shown  = False

    try:
        for i in tqdm(range(ticks_needed * 4), desc="📸 Сбор"):
            if saved >= TOTAL_FRAMES: break
            if not vehicle.is_alive:
                print("\nМашина уничтожена."); break

            # ── До тика ──────────────────────────────────────────────────────
            t_before   = vehicle.get_transform()
            loc_before = t_before.location
            yaw_before = t_before.rotation.yaw
            cmd        = get_command(vehicle, world_map)

            world.tick()
            weather_tick += 1

            # ── После тика ───────────────────────────────────────────────────
            snapshot   = world.get_snapshot()
            actor_snap = snapshot.find(vehicle.id)
            t_after    = actor_snap.get_transform() if actor_snap else vehicle.get_transform()

            speed_kmh, steer, throttle, brake = compute_controls(
                loc_before, yaw_before, t_after.location, t_after.rotation.yaw)

            # ── Детектор застревания → перезапуск автопилота (без телепорта) ──
            if speed_kmh < STUCK_KMH:
                stuck_ticks += 1
                if stuck_ticks >= STUCK_TICKS_MAX:
                    tqdm.write("Застряли 15 сек → перезапуск автопилота")
                    vehicle.set_autopilot(False, TM_PORT)
                    for _ in range(10): world.tick()
                    vehicle.set_autopilot(True, TM_PORT)
                    tm.vehicle_percentage_speed_difference(vehicle, -50)
                    stuck_ticks = 0
                for q in list(cam_queues.values()) + [lidar_queue]:
                    while not q.empty():
                        try: q.get_nowait()
                        except queue.Empty: break
                continue
            else:
                stuck_ticks = 0

            if speed_kmh < MIN_SPEED_KMH:
                continue

            # ── Смена погоды ──────────────────────────────────────────────────
            if weather_tick % WEATHER_INTERVAL == 0:
                apply_weather(world)

            # ── Получить кадры всех трёх камер ───────────────────────────────
            frames = {}
            ok = True
            for name in [c[0] for c in CAM_CONFIGS]:
                try:
                    frames[name] = cam_queues[name].get(timeout=2.0)
                except queue.Empty:
                    tqdm.write(f"  WARNING: нет кадра камеры {name} на тике {i}")
                    ok = False; break
            if not ok: continue

            # ── LiDAR ─────────────────────────────────────────────────────────
            bev = EMPTY_BEV.copy()
            try:
                ld  = lidar_queue.get(timeout=1.0)
                pts = np.frombuffer(ld.raw_data, dtype=np.float32).reshape(-1, 4)
                bev = lidar_to_bev(pts)
                while not lidar_queue.empty():
                    try:
                        ld  = lidar_queue.get_nowait()
                        bev = lidar_to_bev(
                            np.frombuffer(ld.raw_data, dtype=np.float32).reshape(-1, 4))
                    except queue.Empty: break
            except queue.Empty:
                pass  # LiDAR не критичен

            # ── Сохранить LiDAR (один на тик) ────────────────────────────────
            lname = f"lidar_{tick_idx:05d}.npy"
            np.save(os.path.join(LIDAR_DIR, lname), bev)

            # ── Сохранить 3 кадра + 3 строки CSV ─────────────────────────────
            for cam_name, _, steer_corr in CAM_CONFIGS:
                corrected_steer = float(np.clip(steer + steer_corr, -1.0, 1.0))

                raw     = np.frombuffer(frames[cam_name].raw_data, dtype=np.uint8)
                img     = raw.reshape((IMAGE_SIZE[1], IMAGE_SIZE[0], 4))[:, :, :3]
                img_out = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), SAVE_SIZE)

                fname = f"frame_{frame_idx:05d}.png"
                cv2.imwrite(os.path.join(IMAGES_DIR, fname), img_out,
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])

                writer.writerow([
                    frame_idx, fname, lname,
                    round(corrected_steer,       4),
                    round(float(throttle),        4),
                    round(float(brake),           4),
                    round(float(speed_kmh),       2),
                    cmd, cam_name,
                ])
                cmd_counts[cmd] += 1
                frame_idx += 1
                saved     += 1

            # ── Диагностика первых 2 тиков ────────────────────────────────────
            if tick_idx < 2 and not diag_shown:
                tqdm.write(
                    f"  [тик {tick_idx}] spd={speed_kmh:.1f} km/h  "
                    f"steer(C)={steer:+.3f}  "
                    f"steer(L)={steer+STEER_CORRECTION:+.3f}  "
                    f"steer(R)={steer-STEER_CORRECTION:+.3f}  "
                    f"cmd={CMD_NAMES[cmd]}  "
                    f"lidar={'✓' if bev.max()>0 else '✗'}"
                )
                if tick_idx == 1: diag_shown = True

            tick_idx += 1
            csv_file.flush()

    except KeyboardInterrupt:
        print("\n  Прервано.")
    finally:
        for cam in cam_actors.values():
            cam.stop(); cam.destroy()
        lidar_sensor.stop(); lidar_sensor.destroy()
        vehicle.destroy()
        for npc in npcs:
            try: npc.destroy()
            except Exception: pass
        tm.set_synchronous_mode(False)
        csv_file.close()
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)
        print("   Синхронный режим отключён.")

    print(f"\nСохранено: {saved} строк  ({tick_idx} тиков)")
    print(f"   Кадров реальной езды: {tick_idx}  × 3 камеры = {saved}")
    print(f"   LiDAR-файлов: {tick_idx}")
    print("Распределение команд (по всем 3 камерам):")
    total = max(1, saved)
    for k, v in cmd_counts.items():
        bar = '█' * int(30 * v / total)
        print(f"   {CMD_NAMES[k]:8s}: {v:5d} ({100*v/total:.1f}%)  {bar}")


if __name__ == '__main__':
    import carla
    main()
