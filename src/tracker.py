"""
src/tracker.py

3D Multi-Object Tracking (MOT) over CenterPoint detections produced by
src/run_detector.py. Consumes data/processed/raw_detections.json and exports
data/processed/tracked_objects.json with persistent track IDs and estimated
3D velocities, for downstream collision-prediction consumers.

Invocation (from repo root):
    python src/tracker.py --config configs/pipeline_config.yaml

State space per track (Kalman Filter, filterpy):
    x = [x, y, z, vx, vy, vz, dx, dy, dz, yaw]^T   (10-dim)
Measurement per detection:
    z = [x, y, z, dx, dy, dz, yaw]^T                (7-dim)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

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


def angle_residual(a, b):
    """
    Custom residual function for the Kalman Filter update step.
    Computes the measurement residual y = z - Hx while cleanly wrapping
    the yaw angle difference (index 6) to prevent wrap-around bugs.
    """
    res = a - b
    res[6, 0] = wrap_angle(res[6, 0])
    return res


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
        "min_hits": int(tracking_cfg.get("min_hits", 1)),
        "distance_threshold": float(tracking_cfg.get("distance_threshold", 4.0)),
        "process_noise_std": float(tracking_cfg.get("process_noise_std", 1.0)),
        "measurement_noise_std": float(tracking_cfg.get("measurement_noise_std", 0.5)),
    }


# --------------------------------------------------------------------------
# Single track wrapping a filterpy KalmanFilter
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

        # F is dt-dependent; initialize as identity, updated every predict().
        kf.F = np.eye(STATE_DIM)

        # H maps state -> measurement [x, y, z, dx, dy, dz, yaw]
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

        # Modest initial uncertainty; velocities start with higher uncertainty
        # since they are unobserved at initialization.
        kf.P = np.eye(STATE_DIM) * 10.0
        kf.P[np.ix_(VEL_IDX, VEL_IDX)] *= 100.0

        kf.Q = np.eye(STATE_DIM) * (process_noise_std ** 2)

        return kf

    def predict(self, dt: float):
        F = np.eye(STATE_DIM)
        F[0, 3] = dt  # x = x + vx * dt
        F[1, 4] = dt  # y = y + vy * dt
        F[2, 5] = dt  # z = z + vz * dt
        self.kf.F = F

        # FIX: Scaling process noise correctly with dt while preserving config-defined noise std
        self.kf.Q = np.eye(STATE_DIM) * (self.process_noise_std ** 2) * max(dt, 1e-3)

        self.kf.predict()
        
        # FIX: Explicitly normalize predicted yaw angle state to keep it bounded
        self.kf.x[YAW_IDX, 0] = wrap_angle(self.kf.x[YAW_IDX, 0])

        self.age += 1
        self.time_since_update += 1

    def update(self, detection: dict):
        x, y, z = detection["x"], detection["y"], detection["z"]
        dx, dy, dz = detection["size"]
        yaw = wrap_angle(detection["yaw"])

        z_meas = np.array([[x], [y], [z], [dx], [dy], [dz], [yaw]])
        
        # FIX: Pass custom residual function to resolve yaw wrap-around spikes
        self.kf.update(z_meas, residual_fn=angle_residual)
        
        # FIX: Explicitly normalize yaw state after Kalman adjustment
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
        }


# --------------------------------------------------------------------------
# Tracker manager: association + lifecycle across frames
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
        """
        Hungarian assignment on a 3D Euclidean-distance cost matrix between
        predicted track centers and detection centers. Matches whose distance
        exceeds `distance_threshold` are rejected (gated out).

        Returns (matches, unmatched_track_idx, unmatched_det_idx)
        where matches is a list of (track_idx, det_idx) pairs.
        """
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        track_positions = np.array([t.position for t in self.tracks])
        det_positions = np.array(
            [[d["x"], d["y"], d["z"]] for d in detections]
        )

        # Pairwise Euclidean distance matrix (num_tracks x num_dets).
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
        """
        Advances all tracks by dt, associates them with the current frame's
        detections, updates matches, ages/deletes misses, and spawns new
        tracks for unmatched detections. Returns the list of currently active
        tracks (already updated) as output dicts.
        """
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

        # Death: drop tracks coasting past max_age.
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_age
        ]

        active = [t for t in self.tracks if t.hits >= self.min_hits]
        return [t.to_output_dict() for t in active]


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run(config_path: str):
    cfg = load_config(config_path)
    params = get_tracking_params(cfg)

    logger.info(
        "Tracking params: max_age=%d, min_hits=%d, distance_threshold=%.2fm",
        params["max_age"], params["min_hits"], params["distance_threshold"],
    )

    input_path = Path(params["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Raw detections file not found: {input_path}")

    logger.info("Loading detections from %s", input_path)
    with open(input_path, "r") as f:
        frames = json.load(f)

    # Ensure strict temporal order (dt calculation depends on it).
    frames = sorted(frames, key=lambda fr: fr["timestamp"])
    logger.info("Loaded %d frames.", len(frames))

    tracker = Tracker3D(
        max_age=params["max_age"],
        min_hits=params["min_hits"],
        distance_threshold=params["distance_threshold"],
        process_noise_std=params["process_noise_std"],
        measurement_noise_std=params["measurement_noise_std"],
    )

    output_frames = []
    prev_timestamp_us = None

    for frame in frames:
        timestamp_us = frame["timestamp"]

        if prev_timestamp_us is None:
            dt = 0.0  # First frame: no motion to integrate yet.
        else:
            dt = (timestamp_us - prev_timestamp_us) / 1e6  # microseconds -> seconds
            if dt <= 0:
                logger.warning(
                    "Non-positive dt (%.6fs) between frames at ts=%d; clamping to 1e-3s.",
                    dt, timestamp_us,
                )
                dt = 1e-3

        tracked_objects = tracker.step(frame.get("detections", []), dt)

        output_frames.append(
            {
                "frame_id": frame["frame_id"],
                "timestamp": timestamp_us,
                "tracked_objects": tracked_objects,
            }
        )

        prev_timestamp_us = timestamp_us

    out_path = Path(params["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_frames, f, indent=2)

    total_tracked = sum(len(fr["tracked_objects"]) for fr in output_frames)
    logger.info(
        "Wrote %d frames (%d tracked-object instances, %d unique track IDs) to %s",
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