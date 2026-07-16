"""
src/visualizer.py

Renders an animated Bird's-Eye-View (BEV) simulation of the full perception ->
tracking -> collision-prediction pipeline over nuScenes-mini, and exports it
as an .mp4 video.

Consumes:
    - Raw LIDAR_TOP point clouds through nuscenes-devkit
    - data/processed/tracked_objects.json
    - data/processed/collision_predictions.json

Produces:
    - data/processed/pipeline_simulation.mp4

Invocation from repo root:
    python src/visualizer.py --config configs/pipeline_config.yaml

Coordinate convention:
    - Ego vehicle is drawn at origin (0, 0).
    - x-axis = forward/backward direction.
    - y-axis = left/right direction.
    - Track size = [dx, dy, dz] = [length, width, height].
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless rendering; must be before pyplot import.

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("visualizer")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load YAML pipeline configuration."""

    logger.info("Loading pipeline config from %s", config_path)

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    return cfg or {}


def get_visualizer_params(cfg: dict) -> dict:
    """Read visualizer, nuScenes, output, and ego-vehicle settings."""

    vis_cfg = cfg.get("visualizer", {})
    nusc_cfg = cfg.get("nuscenes", {})
    output_cfg = cfg.get("output", {})
    collision_cfg = cfg.get("collision_prediction", {})

    params = {
        "dataroot": nusc_cfg.get("dataroot", "./data/raw_nuscenes"),
        "version": nusc_cfg.get("version", "v1.0-mini"),

        "tracked_objects_path": output_cfg.get(
            "tracked_objects_path",
            "./data/processed/tracked_objects.json",
        ),
        "collision_predictions_path": output_cfg.get(
            "collision_predictions_path",
            "./data/processed/collision_predictions.json",
        ),

        "video_output_path": vis_cfg.get(
            "video_output_path",
            "./data/processed/pipeline_simulation.mp4",
        ),

        "canvas_xlim": tuple(vis_cfg.get("canvas_xlim", [-50.0, 50.0])),
        "canvas_ylim": tuple(vis_cfg.get("canvas_ylim", [-50.0, 50.0])),
        "pointcloud_downsample_rate": int(
            vis_cfg.get("pointcloud_downsample_rate", 5)
        ),
        "fps": int(vis_cfg.get("fps", 10)),
        "dpi": int(vis_cfg.get("dpi", 120)),
        "colormap": vis_cfg.get("colormap", "tab20"),
        "velocity_arrow_scale": float(
            vis_cfg.get("velocity_arrow_scale", 1.0)
        ),

        # Ego dimensions.
        # x-axis = length direction, y-axis = width direction.
        "ego_length": float(collision_cfg.get("ego_length", 4.5)),
        "ego_width": float(collision_cfg.get("ego_width", 2.0)),

        # Safety buffer around ego vehicle.
        "safety_buffer_x": float(collision_cfg.get("safety_buffer_x", 1.0)),
        "safety_buffer_y": float(collision_cfg.get("safety_buffer_y", 0.5)),
    }

    validate_visualizer_params(params)
    return params


def validate_visualizer_params(params: dict) -> None:
    """Validate visualizer config values."""

    if len(params["canvas_xlim"]) != 2:
        raise ValueError("canvas_xlim must contain exactly two values.")

    if len(params["canvas_ylim"]) != 2:
        raise ValueError("canvas_ylim must contain exactly two values.")

    if params["canvas_xlim"][0] >= params["canvas_xlim"][1]:
        raise ValueError("canvas_xlim lower bound must be smaller than upper bound.")

    if params["canvas_ylim"][0] >= params["canvas_ylim"][1]:
        raise ValueError("canvas_ylim lower bound must be smaller than upper bound.")

    if params["pointcloud_downsample_rate"] < 1:
        raise ValueError("pointcloud_downsample_rate must be >= 1.")

    if params["fps"] < 1:
        raise ValueError("fps must be >= 1.")

    if params["dpi"] < 50:
        raise ValueError("dpi is too small. Use at least 50.")

    if params["ego_length"] <= 0 or params["ego_width"] <= 0:
        raise ValueError("ego_length and ego_width must be positive.")

    if params["safety_buffer_x"] < 0 or params["safety_buffer_y"] < 0:
        raise ValueError("safety buffers cannot be negative.")


# --------------------------------------------------------------------------
# Data loading / merging
# --------------------------------------------------------------------------
def load_json(path: str) -> list:
    """Load a JSON file and require it to contain a list."""

    json_path = Path(path)

    if not json_path.exists():
        raise FileNotFoundError(f"Required pipeline output not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in JSON file: {json_path}")

    return data


def merge_pipeline_frames(tracked_frames: list, collision_frames: list) -> list:
    """
    Merge tracked_objects.json and collision_predictions.json by frame_id.

    The tracked frame order is preserved.
    """

    collision_by_frame = {fr["frame_id"]: fr for fr in collision_frames}

    merged = []
    missing_collision_count = 0

    for tracked_frame in tracked_frames:
        frame_id = tracked_frame["frame_id"]
        collision_frame = collision_by_frame.get(frame_id)

        if collision_frame is None:
            missing_collision_count += 1
            collision_warnings = []
        else:
            collision_warnings = collision_frame.get("collision_warnings", [])

        merged.append(
            {
                "frame_id": frame_id,
                "scene_name": tracked_frame.get("scene_name", "unknown_scene"),
                "timestamp": tracked_frame["timestamp"],
                "tracked_objects": tracked_frame.get("tracked_objects", []),
                "collision_warnings": collision_warnings,
            }
        )

    if missing_collision_count > 0:
        logger.warning(
            "%d tracked frames did not have matching collision prediction frames.",
            missing_collision_count,
        )

    return merged


def load_lidar_points_2d(
    nusc: NuScenes,
    frame_id: str,
    downsample_rate: int,
) -> np.ndarray | None:
    """
    Load and downsample top-down LIDAR_TOP points for one nuScenes sample token.

    Returns:
        Nx2 array containing x and y coordinates, or None if loading fails.
    """

    try:
        sample = nusc.get("sample", frame_id)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)

        pcd_path = os.path.join(nusc.dataroot, lidar_data["filename"])
        point_cloud = LidarPointCloud.from_file(pcd_path)

    except Exception as exc:  # Keep video rendering resilient.
        logger.warning(
            "Could not load LIDAR_TOP for frame %s (%s). "
            "Skipping point cloud for this frame.",
            frame_id,
            exc,
        )
        return None

    points_xy = point_cloud.points[:2, :]
    step = max(1, downsample_rate)

    return points_xy[:, ::step].T


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def add_obb_patch(
    ax,
    cx: float,
    cy: float,
    length: float,
    width: float,
    yaw: float,
    **kwargs,
):
    """
    Add an oriented rectangle in BEV.

    length is along local x-axis.
    width is along local y-axis.
    yaw rotates the box around its center.
    """

    rect = patches.Rectangle(
        (cx - length / 2.0, cy - width / 2.0),
        length,
        width,
        **kwargs,
    )

    transform = mtransforms.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
    rect.set_transform(transform)
    ax.add_patch(rect)

    return rect


def get_track_color(track_id: int, scene_color_map: dict, colormap_name: str):
    """Assign a stable color to each track ID within one scene."""

    colormap = plt.get_cmap(colormap_name)
    n_colors = colormap.N if hasattr(colormap, "N") else 20

    if track_id not in scene_color_map:
        color_index = len(scene_color_map) % n_colors
        scene_color_map[track_id] = colormap(
            color_index / max(n_colors - 1, 1)
        )

    return scene_color_map[track_id]


# --------------------------------------------------------------------------
# Frame renderer
# --------------------------------------------------------------------------
class BEVRenderer:
    """Renders one BEV frame at a time."""

    def __init__(self, nusc: NuScenes, params: dict):
        self.nusc = nusc
        self.params = params
        self.scene_color_map = {}
        self.current_scene = None

        self.fig, self.ax = plt.subplots(figsize=(9, 9))

    def _reset_for_new_scene(self, scene_name: str) -> None:
        """Reset color mapping and scene state when a new scene starts."""

        logger.info("Resetting BEV state for new sequence: %s", scene_name)
        self.scene_color_map = {}
        self.current_scene = scene_name

    def render_frame(self, frame_idx: int, frame: dict):
        """Render one animation frame."""

        if frame["scene_name"] != self.current_scene:
            self._reset_for_new_scene(frame["scene_name"])

        ax = self.ax
        ax.clear()

        ax.set_xlim(*self.params["canvas_xlim"])
        ax.set_ylim(*self.params["canvas_ylim"])
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.set_title(
            f"{frame['scene_name']} | frame {frame_idx} | "
            f"t={frame['timestamp']}",
            color="white",
        )
        ax.tick_params(colors="white")

        # 1. LiDAR point cloud background.
        points_2d = load_lidar_points_2d(
            self.nusc,
            frame["frame_id"],
            self.params["pointcloud_downsample_rate"],
        )

        if points_2d is not None and len(points_2d) > 0:
            ax.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                s=1.0,
                c="lightgray",
                alpha=0.4,
                zorder=1,
            )

        # 2. Safety buffer zone.
        any_critical = any(
            warning.get("is_critical_hazard", False)
            for warning in frame["collision_warnings"]
        )

        if any_critical and frame_idx % 2 == 0:
            zone_color = "red"
            zone_alpha = 0.35
        elif any_critical:
            zone_color = "red"
            zone_alpha = 0.15
        else:
            zone_color = "green"
            zone_alpha = 0.15

        zone_length = self.params["ego_length"] + 2 * self.params["safety_buffer_x"]
        zone_width = self.params["ego_width"] + 2 * self.params["safety_buffer_y"]

        add_obb_patch(
            ax,
            cx=0.0,
            cy=0.0,
            length=zone_length,
            width=zone_width,
            yaw=0.0,
            facecolor=zone_color,
            edgecolor=zone_color,
            alpha=zone_alpha,
            zorder=2,
        )

        # 3. Ego vehicle.
        add_obb_patch(
            ax,
            cx=0.0,
            cy=0.0,
            length=self.params["ego_length"],
            width=self.params["ego_width"],
            yaw=0.0,
            facecolor="blue",
            edgecolor="cyan",
            alpha=0.95,
            zorder=5,
        )

        # 4. Tracked objects and velocity arrows.
        warnings_by_id = {
            warning["track_id"]: warning
            for warning in frame["collision_warnings"]
            if "track_id" in warning
        }

        for track in frame["tracked_objects"]:
            track_id = track["track_id"]

            color = get_track_color(
                track_id,
                self.scene_color_map,
                self.params["colormap"],
            )

            size = track.get("size", [0.0, 0.0, 0.0])
            if len(size) < 2:
                logger.debug("Skipping track %s due to invalid size.", track_id)
                continue

            # Correct convention:
            # size = [dx, dy, dz] = [length, width, height]
            length = float(size[0])
            width = float(size[1])

            x = float(track["x"])
            y = float(track["y"])
            yaw = float(track.get("yaw", 0.0))

            add_obb_patch(
                ax,
                cx=x,
                cy=y,
                length=length,
                width=width,
                yaw=yaw,
                facecolor=color,
                edgecolor="white",
                alpha=0.85,
                zorder=4,
            )

            # Velocity arrow.
            vx = float(track.get("vx", 0.0))
            vy = float(track.get("vy", 0.0))
            scale = self.params["velocity_arrow_scale"]

            if abs(vx) > 1e-3 or abs(vy) > 1e-3:
                ax.arrow(
                    x,
                    y,
                    vx * scale,
                    vy * scale,
                    head_width=0.6,
                    head_length=0.8,
                    fc=color,
                    ec=color,
                    length_includes_head=True,
                    zorder=6,
                )

            # Track ID label.
            ax.text(
                x,
                y + width / 2.0 + 1.0,
                str(track_id),
                color=color,
                fontsize=8,
                ha="center",
                zorder=7,
            )

            # 5. Hazard overlay.
            warning = warnings_by_id.get(track_id)

            if warning is not None:
                ix, iy = warning["predicted_intersection_point"]

                ax.plot(
                    [x, ix],
                    [y, iy],
                    linestyle=":",
                    color="red",
                    linewidth=1.5,
                    zorder=6,
                )

                ax.scatter(
                    [ix],
                    [iy],
                    marker="x",
                    c="red",
                    s=40,
                    zorder=6,
                )

                hud_text = (
                    f"ID {warning['track_id']}\n"
                    f"TTC {warning['time_to_collision']:.2f}s\n"
                    f"Risk {warning['risk_score']:.2f}"
                )

                ax.text(
                    x + 2.0,
                    y + 2.0,
                    hud_text,
                    color="white",
                    fontsize=7,
                    zorder=8,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": (
                            "red"
                            if warning["is_critical_hazard"]
                            else "orange"
                        ),
                        "alpha": 0.85,
                    },
                )

        return ax.patches + ax.lines + ax.texts + ax.collections


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run(config_path: str) -> None:
    """Run BEV video rendering."""

    cfg = load_config(config_path)
    params = get_visualizer_params(cfg)

    logger.info(
        "Visualizer params: canvas_x=%s, canvas_y=%s, downsample=1/%d, fps=%d",
        params["canvas_xlim"],
        params["canvas_ylim"],
        params["pointcloud_downsample_rate"],
        params["fps"],
    )

    tracked_frames = load_json(params["tracked_objects_path"])
    collision_frames = load_json(params["collision_predictions_path"])

    frames = merge_pipeline_frames(tracked_frames, collision_frames)

    if not frames:
        raise ValueError("No frames found to render. Check upstream outputs.")

    frames = sorted(
        frames,
        key=lambda frame: (frame["scene_name"], frame["timestamp"]),
    )

    logger.info("Merged %d frames for animation.", len(frames))

    logger.info(
        "Initializing NuScenes(version=%s, dataroot=%s) for point cloud playback",
        params["version"],
        params["dataroot"],
    )

    nusc = NuScenes(
        version=params["version"],
        dataroot=params["dataroot"],
        verbose=False,
    )

    renderer = BEVRenderer(nusc, params)

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg writer is not available. Install ffmpeg first, for example:\n"
            "  apt-get install ffmpeg\n"
            "or\n"
            "  conda install -c conda-forge ffmpeg"
        )

    def update(frame_idx: int):
        return renderer.render_frame(frame_idx, frames[frame_idx])

    anim = animation.FuncAnimation(
        renderer.fig,
        update,
        frames=len(frames),
        blit=False,
    )

    output_path = Path(params["video_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(total=len(frames), desc="Rendering BEV simulation")

    def progress_callback(current_frame: int, total_frames: int):
        progress_bar.n = current_frame + 1
        progress_bar.refresh()

    writer = animation.FFMpegWriter(fps=params["fps"])

    anim.save(
        str(output_path),
        writer=writer,
        dpi=params["dpi"],
        progress_callback=progress_callback,
    )

    progress_bar.close()
    plt.close(renderer.fig)

    logger.info(
        "Wrote BEV simulation video (%d frames) to %s",
        len(frames),
        output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animated BEV visualizer for the perception/tracking/collision pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline_config.yaml",
    )

    args = parser.parse_args()
    run(args.config)