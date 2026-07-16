"""
src/collision_predictor.py

Forward-projects each tracked actor's constant-velocity trajectory over a
configurable horizon and checks it against the ego vehicle's (padded)
footprint at the origin of each frame's local sensor coordinate system,
producing per-frame collision warnings with Time-to-Collision (TTC) and a
deterministic risk severity score for downstream planning/AEB consumers.

Consumes:  data/processed/tracked_objects.json  (from src/tracker.py)
Produces:  data/processed/collision_predictions.json

Invocation (from repo root):
    python src/collision_predictor.py --config configs/pipeline_config.yaml
"""

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("collision_predictor")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    logger.info("Loading pipeline config from %s", config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def get_collision_params(cfg: dict) -> dict:
    cp_cfg = cfg.get("collision_prediction", {})
    output_cfg = cfg.get("output", {})

    return {
        "input_path": output_cfg.get(
            "tracked_objects_path", "./data/processed/tracked_objects.json"
        ),
        "output_path": output_cfg.get(
            "collision_predictions_path",
            "./data/processed/collision_predictions.json",
        ),
        "prediction_horizon": float(cp_cfg.get("prediction_horizon", 3.0)),
        "time_step": float(cp_cfg.get("time_step", 0.1)),
        "safety_buffer_x": float(cp_cfg.get("safety_buffer_x", 1.0)),
        "safety_buffer_y": float(cp_cfg.get("safety_buffer_y", 0.5)),
        "risk_score_threshold": float(cp_cfg.get("risk_score_threshold", 0.6)),
        "ego_length": float(cp_cfg.get("ego_length", 4.5)),  # local y-axis (length/heading)
        "ego_width": float(cp_cfg.get("ego_width", 2.0)),    # local x-axis (width/lateral)
        # How aggressively safety buffers widen per missed update on a
        # predicted-only (coasting) track, to account for accumulating drift.
        "drift_degradation_factor": float(
            cp_cfg.get("drift_degradation_factor", 0.5)
        ),
        # Hard cap on track staleness before it is excluded entirely rather
        # than just discounted (protects against wildly stale coasts).
        "max_time_since_update_considered": int(
            cp_cfg.get("max_time_since_update_considered", 5)
        ),
    }


# --------------------------------------------------------------------------
# Oriented Bounding Box geometry (2D, top-down / BEV)
# --------------------------------------------------------------------------
def obb_corners(cx: float, cy: float, half_width: float, half_length: float,
                yaw: float) -> np.ndarray:
    """
    Returns the 4 corners (4x2 array) of a 2D oriented box centered at (cx, cy).
    
    Convention Alignment Fix:
    - local_corners maps local x-axis to width and local y-axis to length.
    - Yaw is rotated correctly relative to heading.
    """
    local_corners = np.array([
        [half_width, half_length],
        [half_width, -half_length],
        [-half_width, -half_length],
        [-half_width, half_length],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world_corners = local_corners @ rot.T
    world_corners[:, 0] += cx
    world_corners[:, 1] += cy
    return world_corners


def _project_polygon(corners: np.ndarray, axis: np.ndarray) -> tuple:
    projections = corners @ axis
    return projections.min(), projections.max()


def obb_intersect(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """
    Separating Axis Theorem (SAT) test for two convex quadrilaterals.
    Returns True if the boxes overlap (no separating axis found).
    """
    for corners in (corners_a, corners_b):
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            
            # FIX: Zero-division guard for coincident or corrupted corner states
            if norm < 1e-9:
                continue
            axis = axis / norm

            min_a, max_a = _project_polygon(corners_a, axis)
            min_b, max_b = _project_polygon(corners_b, axis)

            if max_a < min_b or max_b < min_a:
                return False  # Separating axis found -> no intersection.

    return True


# --------------------------------------------------------------------------
# Per-track collision evaluation
# --------------------------------------------------------------------------
def evaluate_track_collision(track: dict, params: dict) -> Optional[dict]:
    """
    Rolls out a single track's constant-velocity trajectory and checks it
    against the (padded) ego footprint at each discrete time step. Returns a
    collision-warning dict if an intersection is found within the horizon,
    otherwise None.
    """
    time_since_update = int(track.get("time_since_update", 0))
    is_predicted_only = bool(track.get("is_predicted_only", False))

    if time_since_update > params["max_time_since_update_considered"]:
        logger.debug(
            "Track %s excluded: time_since_update=%d exceeds cap.",
            track.get("track_id"), time_since_update,
        )
        return None

    # Drift-degradation heuristic: widen safety buffers the longer a track
    # has been coasting on prediction-only updates.
    degradation_scale = 1.0
    if is_predicted_only or time_since_update > 0:
        degradation_scale = 1.0 + params["drift_degradation_factor"] * time_since_update

    safety_buffer_x = params["safety_buffer_x"] * degradation_scale
    safety_buffer_y = params["safety_buffer_y"] * degradation_scale

    # Ego dimensions: length along y-axis, width along x-axis
    ego_half_width = params["ego_width"] / 2.0 + safety_buffer_y
    ego_half_length = params["ego_length"] / 2.0 + safety_buffer_x
    ego_corners = obb_corners(0.0, 0.0, ego_half_width, ego_half_length, 0.0)

    # size = [dx, dy, dz] = [width, length, height]
    dx, dy, _dz = track["size"]
    track_half_width = dx / 2.0
    track_half_length = dy / 2.0

    x0, y0 = track["x"], track["y"]
    vx, vy = track.get("vx", 0.0), track.get("vy", 0.0)
    yaw = track.get("yaw", 0.0)

    horizon = params["prediction_horizon"]
    dt = params["time_step"]
    n_steps = int(round(horizon / dt))

    for step in range(n_steps + 1):
        t = step * dt
        x_t = x0 + vx * t
        y_t = y0 + vy * t

        track_corners = obb_corners(
            x_t, y_t, track_half_width, track_half_length, yaw
        )

        if obb_intersect(ego_corners, track_corners):
            risk_score = max(0.0, 1.0 - (t / horizon)) if horizon > 0 else 1.0
            return {
                "track_id": track.get("track_id"),
                "time_to_collision": round(float(t), 3),
                "predicted_intersection_point": [round(float(x_t), 3), round(float(y_t), 3)],
                "risk_score": round(float(risk_score), 3),
                "is_critical_hazard": bool(risk_score >= params["risk_score_threshold"]),
            }

    return None


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run(config_path: str):
    cfg = load_config(config_path)
    params = get_collision_params(cfg)

    logger.info(
        "Collision prediction params: horizon=%.2fs, step=%.2fs, "
        "buffer_x=%.2fm, buffer_y=%.2fm, risk_threshold=%.2f",
        params["prediction_horizon"], params["time_step"],
        params["safety_buffer_x"], params["safety_buffer_y"],
        params["risk_score_threshold"],
    )

    input_path = Path(params["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Tracked objects file not found: {input_path}")

    logger.info("Loading tracked objects from %s", input_path)
    with open(input_path, "r") as f:
        tracked_frames = json.load(f)

    # Group by scene for clean, isolated processing/logging.
    scene_buckets = defaultdict(list)
    for frame in tracked_frames:
        scene_name = frame.get("scene_name", "unknown_scene")
        scene_buckets[scene_name].append(frame)

    output_frames = []
    total_warnings = 0
    total_critical = 0

    for scene_id, frames in scene_buckets.items():
        frames = sorted(frames, key=lambda fr: fr["timestamp"])
        logger.info(
            "Evaluating collision risk for sequence: %s (%d frames)",
            scene_id, len(frames),
        )

        for frame in frames:
            collision_warnings = []
            for track in frame.get("tracked_objects", []):
                warning = evaluate_track_collision(track, params)
                if warning is not None:
                    collision_warnings.append(warning)
                    total_warnings += 1
                    if warning["is_critical_hazard"]:
                        total_critical += 1

            output_frames.append(
                {
                    "frame_id": frame["frame_id"],
                    "scene_name": scene_id,
                    "timestamp": frame["timestamp"],
                    "collision_warnings": collision_warnings,
                }
            )

    out_path = Path(params["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_frames, f, indent=2)

    logger.info(
        "Wrote %d frames (%d collision warnings, %d flagged critical) to %s",
        len(output_frames), total_warnings, total_critical, out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constant-velocity collision risk prediction over tracked actors."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()

    run(args.config)