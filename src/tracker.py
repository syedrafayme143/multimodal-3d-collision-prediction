"""
src/tracker.py

3D Multi-Object Tracking (MOT) over CenterPoint detections produced by
src/run_detector.py. Consumes data/processed/raw_detections.json and exports
data/processed/tracked_objects.json with persistent track IDs and estimated
3D velocities, ensuring strict scene-level isolation for collision prediction.

Converts coordinates from Global UTM space to local LiDAR frame space using 
nuScenes calibration logs to ensure data validation compatibility.

Invocation (from repo root):
    python src/tracker.py --config configs/pipeline_config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("tracker")

STATE_DIM = 10
MEAS_DIM = 7

# State layout: 0:x 1:y 2:z 3:vx 4:vy 5:vz 6:dx 7:dy 8:dz 9:yaw
POS_IDX = [0, 1, 2]
VEL_IDX = [3, 4, 5]
SIZE_IDX = [6, 7, 8]
YAW_IDX = 9


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def wrap_angle(angle: float) -> float:
    """Wraps an angle in radians to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def transform_global_to_local(detection: dict, ego_pose: dict, calibrated_sensor: dict) -> dict:
    """
    Transforms a global UTM detection into the local LiDAR coordinate frame[cite: 2].
    """
    # 1. Extract position vector
    global_pos = np.array([detection["x"], detection["y"], detection["z"]])
    
    # Global to Ego Transformation: v_ego = inv(q_ego) * (v_global - t_ego)[cite: 2]
    q_ego_inv = Quaternion(ego_pose["rotation"]).inverse
    ego_pos = q_ego_inv.rotate(global_pos - np.array(ego_pose["translation"]))
    
    # Ego to LiDAR Transformation: v_sensor = inv(q_cs) * (v_ego - t_cs)[cite: 2]
    q_cs_inv = Quaternion(calibrated_sensor["rotation"]).inverse
    local_pos = q_cs_inv.rotate(ego_pos - np.array(calibrated_sensor["translation"]))
    
    # 2. Extract and transform yaw orientation quaternion[cite: 2]
    global_quat = Quaternion(axis=[0, 0, 1], angle=detection["yaw"])
    local_quat = q_cs_inv * q_ego_inv * global_quat
    
    # Extract the local yaw angle by mapping a forward vector
    v_forward = local_quat.rotate(np.array([1.0, 0.0, 0.0]))
    local_yaw = np.arctan2(v_forward[1], v_forward[0])
    
    # Return modified local detection copy
    local_detection = detection.copy()
    local_detection["x"] = float(local_pos[0])
    local_detection["y"] = float(local_pos[1])
    local_detection["z"] = float(local_pos[2])
    local_detection["yaw"] = float(wrap_angle(local_yaw))
    
    return local_detection


# --------------------------------------------------------------------------
# Config Loading
# --------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    logger.info("Loading pipeline config from %s", config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def get_tracking_params(cfg: dict) -> dict:
    tracking_cfg = cfg.get("tracking", {})
    output_cfg = cfg.get("output", {})

    return {
        "input_path": output_cfg.get(
            "raw_detections_path", "./data/processed/raw_detections.json"
        ),
        "output_path": output_cfg.get(
            "tracked_objects_path", "./data/processed/tracked_objects.json"
        ),
        "max_age": int(tracking_cfg.get("max_age", 3)),
        "min_hits": int(tracking_cfg.get("min_hits", 2)),
        "distance_threshold": float(tracking_cfg.get("distance_threshold", 4.0)),
        "process_noise_std": float(tracking_cfg.get("process_noise_std", 1.0)),
        "measurement_noise_std": float(tracking_cfg.get("measurement_noise_std", 0.5)),
    }


# --------------------------------------------------------------------------
# Single track wrapping a filterpy KalmanFilter[cite: 1]
# --------------------------------------------------------------------------
class Track:
    _next_id = 1

    def __init__(self, detection: dict, process_noise_std: float,
                 measurement_noise_std: float):
        self.id = Track._next_id
        Track._next_id += 1

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.process_noise_std = process_noise_std

        self.last_class = detection.get("class", "car")
        self.last_score = float(detection.get("score", 0.0))

        self.kf = self._build_kf(process_noise_std, measurement_noise_std)

        x, y, z = detection["x"], detection["y"], detection["z"]
        dx, dy, dz = detection["size"]
        yaw = wrap_angle(detection["yaw"])

        self.kf.x = np.zeros((STATE_DIM, 1))
        self.kf.x[POS_IDX, 0] = [x, y, z]
        self.kf.x[VEL_IDX, 0] = [0.0, 0.0, 0.0]
        self.kf.x[SIZE_IDX, 0] = [dx, dy, dz]
        self.kf.x[YAW_IDX, 0] = yaw

    @staticmethod
    def _build_kf(process_noise_std: float, measurement_noise_std: float) -> KalmanFilter:
        kf = KalmanFilter(dim_x=STATE_DIM, dim_z=MEAS_DIM)
        kf.F = np.eye(STATE_DIM)

        H = np.zeros((MEAS_DIM, STATE_DIM))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        H[3, 6] = 1.0  # dx
        H[4, 7] = 1.0  # dy
        H[5, 8] = 1.0  # dz
        H[6, 9] = 1.0  # yaw
        kf.H = H

        kf.R = np.eye(MEAS_DIM) * (measurement_noise_std ** 2)
        kf.P = np.eye(STATE_DIM) * 10.0
        kf.P[np.ix_(VEL_IDX, VEL_IDX)] *= 100.0
        kf.Q = np.eye(STATE_DIM) * (process_noise_std ** 2)

        return kf

    def predict(self, dt: float):
        F = np.eye(STATE_DIM)
        F[0, 3] = dt  
        F[1, 4] = dt  
        F[2, 5] = dt  
        self.kf.F = F

        self.kf.Q = np.eye(STATE_DIM) * (self.process_noise_std ** 2) * max(dt, 1e-3)
        self.kf.predict()
        self.kf.x[YAW_IDX, 0] = wrap_angle(self.kf.x[YAW_IDX, 0])

        self.age += 1
        self.time_since_update += 1

    def update(self, detection: dict):
        x, y, z = detection["x"], detection["y"], detection["z"]
        dx, dy, dz = detection["size"]
        raw_yaw = detection["yaw"]

        pred_yaw = self.kf.x[YAW_IDX, 0]
        aligned_yaw = pred_yaw + wrap_angle(raw_yaw - pred_yaw)

        z_meas = np.array([[x], [y], [z], [dx], [dy], [dz], [aligned_yaw]])
        self.kf.update(z_meas)
        self.kf.x[YAW_IDX, 0] = wrap_angle(self.kf.x[YAW_IDX, 0])

        self.time_since_update = 0
        self.hits += 1
        self.last_class = detection.get("class", self.last_class)
        self.last_score = float(detection.get("score", self.last_score))

    @property
    def position(self) -> np.ndarray:
        return self.kf.x[POS_IDX, 0]

    def to_output_dict(self) -> dict:
        state = self.kf.x[:, 0]
        x, y, z = state[POS_IDX]
        vx, vy, vz = state[VEL_IDX]
        dx, dy, dz = state[SIZE_IDX]
        yaw = state[YAW_IDX]

        return {
            "track_id": self.id,
            "class": self.last_class,
            "score": round(self.last_score, 4),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "size": [float(dx), float(dy), float(dz)],
            "yaw": float(yaw),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "time_since_update": int(self.time_since_update),
            "is_predicted_only": bool(self.time_since_update > 0),
        }


# --------------------------------------------------------------------------
# Tracker manager: association + lifecycle across frames[cite: 1]
# --------------------------------------------------------------------------
class Tracker3D:
    def __init__(self, max_age: int, min_hits: int, distance_threshold: float,
                 process_noise_std: float, measurement_noise_std: float):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.tracks = []

    def _associate(self, detections: list):
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        track_positions = np.array([t.position for t in self.tracks])
        det_positions = np.array([[d["x"], d["y"], d["z"]] for d in detections])

        diff = track_positions[:, None, :] - det_positions[None, :, :]
        cost_matrix = np.linalg.norm(diff, axis=2)

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matches = []
        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] <= self.distance_threshold:
                matches.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

    def step(self, detections: list, dt: float) -> list:
        for track in self.tracks:
            track.predict(dt)

        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        for det_idx in unmatched_dets:
            new_track = Track(
                detections[det_idx],
                self.process_noise_std,
                self.measurement_noise_std,
            )
            self.tracks.append(new_track)

        # Death step
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Filter tracks passing min_hits threshold
        active = [t for t in self.tracks if t.hits >= self.min_hits]
        return [t.to_output_dict() for t in active]


# --------------------------------------------------------------------------
# Main pipeline[cite: 1]
# --------------------------------------------------------------------------
def run(config_path: str):
    cfg = load_config(config_path)
    params = get_tracking_params(cfg)

    # Initialize NuScenes Instance to acquire localized transform frames[cite: 2]
    nusc_cfg = cfg.get("nuscenes", {})
    logger.info("Initializing NuScenes interface for tracking frame transformations...")
    nusc = NuScenes(
        version=nusc_cfg.get("version", "v1.0-mini"),
        dataroot=nusc_cfg.get("dataroot", "./data/raw_nuscenes"),
        verbose=False
    )

    logger.info(
        "Tracking params: max_age=%d, min_hits=%d, distance_threshold=%.2fm",
        params["max_age"], params["min_hits"], params["distance_threshold"],
    )

    input_path = Path(params["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Raw detections file not found: {input_path}")

    logger.info("Loading detections from %s", input_path)
    with open(input_path, "r") as f:
        raw_frames = json.load(f)

    # Categorize frames by their nuScenes scene origin[cite: 1]
    scene_buckets = defaultdict(list)
    for frame in raw_frames:
        scene_name = frame.get("scene_name", "scene-0061" if "0061" in frame["frame_id"] else "scene-0103")
        scene_buckets[scene_name].append(frame)

    output_frames = []

    # Process each scene sequence in total isolation[cite: 1]
    for scene_id, frames in scene_buckets.items():
        logger.info("Starting fresh isolated tracker context for sequence: %s", scene_id)
        
        # Sort current scene frames chronologically[cite: 1]
        frames = sorted(frames, key=lambda fr: fr["timestamp"])
        logger.info("Sequence %s contains %d frames.", scene_id, len(frames))

        # Instantiate a completely fresh isolated tracker instance[cite: 1]
        tracker = Tracker3D(
            max_age=params["max_age"],
            min_hits=params["min_hits"],
            distance_threshold=params["distance_threshold"],
            process_noise_std=params["process_noise_std"],
            measurement_noise_std=params["measurement_noise_std"],
        )

        prev_timestamp_us = None

        for frame in frames:
            timestamp_us = frame["timestamp"]
            frame_id = frame["frame_id"]

            # Acquire vehicle transformation metrics relative to coordinate origin[cite: 2]
            sample = nusc.get("sample", frame_id)
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = nusc.get("sample_data", lidar_token)
            ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
            calibrated_sensor = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

            # Map raw detections from Global UTM space into Local LiDAR frame space[cite: 2]
            local_detections = []
            for det in frame.get("detections", []):
                local_det = transform_global_to_local(det, ego_pose, calibrated_sensor)
                local_detections.append(local_det)

            if prev_timestamp_us is None:
                dt = 0.0  
            else:
                dt = (timestamp_us - prev_timestamp_us) / 1e6  
                if dt <= 0:
                    logger.warning(
                        "Non-positive dt (%.6fs) inside %s; clamping to 1e-3s.",
                        dt, scene_id,
                    )
                    dt = 1e-3

            tracked_objects = tracker.step(local_detections, dt)

            output_frames.append(
                {
                    "frame_id": frame_id,
                    "scene_name": scene_id,
                    "timestamp": timestamp_us,
                    "tracked_objects": tracked_objects,
                }
            )

            prev_timestamp_us = timestamp_us

    # --- Save output --------------------------------------------------------
    out_path = Path(params["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_frames, f, indent=2)

    total_tracked = sum(len(fr["tracked_objects"]) for fr in output_frames)
    logger.info(
        "Wrote %d frames (%d tracked instances, %d total IDs allocated) to %s",
        len(output_frames), total_tracked, Track._next_id - 1, out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D multi-object tracking over CenterPoint detections."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()

    run(args.config)