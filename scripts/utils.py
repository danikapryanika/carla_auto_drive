"""
Shared utilities for LIDAR BEV projection and navigation command inference.
Used in both data collection and inference so behaviour stays identical.
"""
import numpy as np

# ── LIDAR BEV parameters ────────────────────────────────────────────────────
LIDAR_RANGE   = 25.0    # metres in each direction from vehicle centre
BEV_SIZE      = 200     # pixels (200×200 grid)
BEV_RES       = (LIDAR_RANGE * 2) / BEV_SIZE   # 0.25 m / pixel
HEIGHT_SLICES = 4       # vertical bins; +1 intensity channel → 5 total channels
Z_RANGE       = (-3.0, 2.0)

# ── Command identifiers ──────────────────────────────────────────────────────
CMD_FOLLOW   = 0
CMD_LEFT     = 1
CMD_RIGHT    = 2
CMD_STRAIGHT = 3
CMD_NAMES    = {CMD_FOLLOW: "FOLLOW", CMD_LEFT: "LEFT",
                CMD_RIGHT: "RIGHT",  CMD_STRAIGHT: "STRAIGHT"}

# ── ImageNet normalisation (used with pretrained ResNet-18) ──────────────────
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = (224, 224)

MAX_SPEED_KMH = 60.0   # normalisation factor


def lidar_to_bev(pts: np.ndarray) -> np.ndarray:
    """
    Convert a LIDAR point cloud (N, 4) [x, y, z, intensity] to a
    bird's-eye-view tensor of shape (HEIGHT_SLICES+1, BEV_SIZE, BEV_SIZE).

    CARLA sensor frame: x=forward, y=right, z=up.
    BEV axes: axis-0 (row) = y (right→down), axis-1 (col) = x (forward→right).
    """
    x_lo, x_hi = -LIDAR_RANGE, LIDAR_RANGE
    y_lo, y_hi = -LIDAR_RANGE, LIDAR_RANGE
    z_lo, z_hi = Z_RANGE

    bev = np.zeros((HEIGHT_SLICES + 1, BEV_SIZE, BEV_SIZE), dtype=np.float32)

    if len(pts) == 0:
        return bev

    mask = (
        (pts[:, 0] > x_lo) & (pts[:, 0] < x_hi) &
        (pts[:, 1] > y_lo) & (pts[:, 1] < y_hi) &
        (pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)
    )
    pts = pts[mask]
    if len(pts) == 0:
        return bev

    col = np.floor((pts[:, 0] - x_lo) / BEV_RES).astype(np.int32)
    row = np.floor((pts[:, 1] - y_lo) / BEV_RES).astype(np.int32)
    col = np.clip(col, 0, BEV_SIZE - 1)
    row = np.clip(row, 0, BEV_SIZE - 1)

    z_step = (z_hi - z_lo) / HEIGHT_SLICES
    for k in range(HEIGHT_SLICES):
        z_bot = z_lo + k * z_step
        z_top = z_lo + (k + 1) * z_step
        ch = (pts[:, 2] >= z_bot) & (pts[:, 2] < z_top)
        np.maximum.at(bev[k], (row[ch], col[ch]), 1.0)

    # Intensity channel (normalised density)
    np.add.at(bev[HEIGHT_SLICES], (row, col), pts[:, 3])
    vmax = bev[HEIGHT_SLICES].max()
    if vmax > 0:
        bev[HEIGHT_SLICES] /= vmax

    return bev


def get_command(vehicle, world_map,
                lookahead_m: float = 28.0,
                junction_ahead_m: float = 22.0) -> int:
    """
    Infer the upcoming high-level navigation command.

    lookahead_m      — смотрим на waypoint ВПЕРЕДИ на это расстояние для определения
                       yaw-разницы. Должно быть > junction_ahead_m, чтобы точка
                       оказалась уже за перекрёстком и давала правильное направление.
    junction_ahead_m — считать «рядом с перекрёстком» если он ближе этого расстояния.
                       22 м при 30 км/ч → ~2.6 с реакции для поворота.
    """
    import carla

    loc = vehicle.get_location()
    wp  = world_map.get_waypoint(
        loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return CMD_FOLLOW

    vehicle_yaw = vehicle.get_transform().rotation.yaw
    next_wps = wp.next(lookahead_m)
    if not next_wps:
        return CMD_FOLLOW

    ahead_wp = next_wps[0]
    diff = ((ahead_wp.transform.rotation.yaw - vehicle_yaw) + 180) % 360 - 180

    # Перекрёсток: текущий WP внутри junction ИЛИ junction в пределах junction_ahead_m.
    # Проверяем несколько точек вдоль дороги — wp.next(d) может «проскочить» через
    # короткую junction-зону если d велико.
    near_junction = wp.is_junction
    if not near_junction:
        step = junction_ahead_m / 3.0
        cur  = wp
        for _ in range(3):
            nxt = cur.next(step)
            if not nxt:
                break
            cur = nxt[0]
            if cur.is_junction:
                near_junction = True
                break

    if near_junction:
        if diff < -20:
            return CMD_LEFT
        elif diff > 20:
            return CMD_RIGHT
        else:
            return CMD_STRAIGHT

    return CMD_FOLLOW


def obstacle_ahead(pts: np.ndarray,
                   max_dist: float = 8.0,
                   half_width: float = 1.5,
                   min_z: float = 0.2,
                   max_z: float = 2.2,
                   min_pts: int = 8) -> bool:
    """Safety check: returns True if >= min_pts LIDAR points are found
    directly ahead within max_dist metres and within the lane width."""
    if len(pts) == 0:
        return False
    mask = (
        (pts[:, 0] > 1.0) & (pts[:, 0] < max_dist) &
        (np.abs(pts[:, 1]) < half_width) &
        (pts[:, 2] > min_z) & (pts[:, 2] < max_z)
    )
    return int(mask.sum()) >= min_pts
