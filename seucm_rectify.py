#!/usr/bin/env python3
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def load_seucm_params(calibration_path: Path) -> Dict:
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    return data["rgb_calibration"]["SEUCM0"]


def load_anchor_params(params_path: Path) -> Dict:
    return json.loads(params_path.read_text(encoding="utf-8"))


def get_output_center(width: int, height: int) -> Tuple[float, float]:
    return (width - 1) / 2.0, (height - 1) / 2.0


def unproject_seucm_pixel(u: float, v: float, params: Dict) -> np.ndarray:
    fx = float(params["fx"])
    fy = float(params["fy"])
    eu = float(params["eu"])
    ev = float(params["ev"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])

    mx = (u - eu) / fx
    my = (v - ev) / fy
    rho2 = mx * mx + my * my

    a = 1.0 - (alpha * alpha) * beta * rho2
    b = -2.0 * (1.0 - alpha)
    c = 1.0 - 2.0 * alpha

    if abs(a) < 1e-10:
        scale = 1.0 if abs(b) < 1e-10 else (-c / b)
    else:
        disc = max(b * b - 4.0 * a * c, 0.0)
        sqrt_disc = math.sqrt(disc)
        roots = [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]
        positive_roots = [value for value in roots if value > 0.0]
        if not positive_roots:
            raise RuntimeError("Cannot unproject anchor pixel with SEUCM model")
        scale = max(positive_roots)

    ray = np.array([mx * scale, my * scale, 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    return ray


def rotation_from_z_to_ray(target_ray: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    target_ray = target_ray / np.linalg.norm(target_ray)

    cross = np.cross(z_axis, target_ray)
    sin_theta = np.linalg.norm(cross)
    cos_theta = float(np.dot(z_axis, target_ray))

    if sin_theta < 1e-12:
        if cos_theta > 0.0:
            return np.eye(3, dtype=np.float64)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
        )

    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - cos_theta) / (sin_theta * sin_theta))


def build_rotation(params: Dict, anchor_x: float, anchor_y: float) -> np.ndarray:
    anchor_ray = unproject_seucm_pixel(anchor_x, anchor_y, params)
    return rotation_from_z_to_ray(anchor_ray)


def get_anchor_norm(params: Dict, anchor_x: float, anchor_y: float) -> Tuple[float, float]:
    anchor_ray = unproject_seucm_pixel(anchor_x, anchor_y, params)
    if anchor_ray[2] <= 1e-10:
        raise RuntimeError("Anchor ray is not in front of the virtual camera")
    return float(anchor_ray[0] / anchor_ray[2]), float(anchor_ray[1] / anchor_ray[2])


def build_remap(
    params: Dict,
    focal_scale: float,
    rotate_anchor_x: float,
    rotate_anchor_y: float,
    shift_x: float,
    shift_y: float,
    roll_degrees: float,
    output_width: int,
    output_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx_src = float(params["fx"])
    fy_src = float(params["fy"])
    eu = float(params["eu"])
    ev = float(params["ev"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    out_cx, out_cy = get_output_center(output_width, output_height)

    fx_out = fx_src * focal_scale
    fy_out = fy_src * focal_scale

    xs = np.arange(output_width, dtype=np.float32) - out_cx
    ys = np.arange(output_height, dtype=np.float32) - out_cy
    x_grid, y_grid = np.meshgrid(xs, ys)

    roll_radians = math.radians(float(roll_degrees))
    cos_roll = math.cos(roll_radians)
    sin_roll = math.sin(roll_radians)

    pre_rot_x = cos_roll * x_grid + sin_roll * y_grid
    pre_rot_y = -sin_roll * x_grid + cos_roll * y_grid

    x_grid = (pre_rot_x - float(shift_x)) / fx_out
    y_grid = (pre_rot_y - float(shift_y)) / fy_out

    rotation = build_rotation(params, rotate_anchor_x, rotate_anchor_y)
    src_x = rotation[0, 0] * x_grid + rotation[0, 1] * y_grid + rotation[0, 2]
    src_y = rotation[1, 0] * x_grid + rotation[1, 1] * y_grid + rotation[1, 2]
    src_z = rotation[2, 0] * x_grid + rotation[2, 1] * y_grid + rotation[2, 2]

    valid = src_z > 1e-6
    norm_x = np.empty_like(src_x, dtype=np.float32)
    norm_y = np.empty_like(src_y, dtype=np.float32)
    norm_x[valid] = src_x[valid] / src_z[valid]
    norm_y[valid] = src_y[valid] / src_z[valid]
    norm_x[~valid] = 0.0
    norm_y[~valid] = 0.0

    denom = alpha * np.sqrt(beta * (norm_x * norm_x + norm_y * norm_y) + 1.0) + (1.0 - alpha)
    map_x = fx_src * (norm_x / denom) + eu
    map_y = fy_src * (norm_y / denom) + ev
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def fits_inside(
    params: Dict,
    focal_scale: float,
    rotate_anchor_x: float,
    rotate_anchor_y: float,
    shift_x: float,
    shift_y: float,
    roll_degrees: float,
    output_width: int,
    output_height: int,
) -> bool:
    width = int(params["w"])
    height = int(params["h"])
    map_x, map_y, valid = build_remap(
        params,
        focal_scale,
        rotate_anchor_x,
        rotate_anchor_y,
        shift_x,
        shift_y,
        roll_degrees,
        output_width,
        output_height,
    )
    if not bool(valid.all()):
        return False
    return (
        float(map_x.min()) >= 0.0
        and float(map_x.max()) <= (width - 1)
        and float(map_y.min()) >= 0.0
        and float(map_y.max()) <= (height - 1)
    )


def find_full_view_scale(
    params: Dict,
    rotate_anchor_x: float,
    rotate_anchor_y: float,
    shift_x: float,
    shift_y: float,
    roll_degrees: float,
    output_width: int,
    output_height: int,
) -> float:
    low = 0.01
    high = 1.0

    while fits_inside(
        params, low, rotate_anchor_x, rotate_anchor_y, shift_x, shift_y, roll_degrees, output_width, output_height
    ):
        low *= 0.5
        if low < 1e-6:
            return low

    while not fits_inside(
        params, high, rotate_anchor_x, rotate_anchor_y, shift_x, shift_y, roll_degrees, output_width, output_height
    ):
        high *= 1.25
        if high > 100.0:
            raise RuntimeError("Cannot find a valid focal scale")

    for _ in range(32):
        mid = (low + high) / 2.0
        if fits_inside(
            params, mid, rotate_anchor_x, rotate_anchor_y, shift_x, shift_y, roll_degrees, output_width, output_height
        ):
            high = mid
        else:
            low = mid

    return high


def remap_frame(
    frame: np.ndarray,
    params: Dict,
    rotate_anchor_x: float,
    rotate_anchor_y: float,
    shift_x: float,
    shift_y: float,
    roll_degrees: float,
    focal_scale: float,
    output_width: int,
    output_height: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    map_x, map_y, valid = build_remap(
        params,
        focal_scale,
        rotate_anchor_x,
        rotate_anchor_y,
        shift_x,
        shift_y,
        roll_degrees,
        output_width,
        output_height,
    )
    output = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    stats = {
        "valid_ratio": float(valid.mean()),
        "min_x": float(map_x[valid].min()) if bool(valid.any()) else -1.0,
        "max_x": float(map_x[valid].max()) if bool(valid.any()) else -1.0,
        "min_y": float(map_y[valid].min()) if bool(valid.any()) else -1.0,
        "max_y": float(map_y[valid].max()) if bool(valid.any()) else -1.0,
    }
    return output, stats


def extract_view_outline(
    params: Dict,
    rotate_anchor_x: float,
    rotate_anchor_y: float,
    shift_x: float,
    shift_y: float,
    roll_degrees: float,
    focal_scale: float,
    output_width: int,
    output_height: int,
    step: int = 8,
):
    map_x, map_y, valid = build_remap(
        params,
        focal_scale,
        rotate_anchor_x,
        rotate_anchor_y,
        shift_x,
        shift_y,
        roll_degrees,
        output_width,
        output_height,
    )

    height, width = map_x.shape
    border_indices = []

    for x in range(0, width, step):
        border_indices.append((0, x))
    border_indices.append((0, width - 1))

    for y in range(step, height, step):
        border_indices.append((y, width - 1))
    border_indices.append((height - 1, width - 1))

    for x in range(width - 1 - step, -1, -step):
        border_indices.append((height - 1, x))
    border_indices.append((height - 1, 0))

    for y in range(height - 1 - step, 0, -step):
        border_indices.append((y, 0))

    outline = []
    for y, x in border_indices:
        if valid[y, x]:
            outline.append((float(map_x[y, x]), float(map_y[y, x])))
        else:
            outline.append(None)
    return outline


def save_anchor_params(params_path: Path, payload: Dict) -> None:
    params_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
