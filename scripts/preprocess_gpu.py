#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.signal import find_peaks
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cuconvolve
    from cupyx.scipy.ndimage import zoom as cuzoom

    GPU_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depends on local CUDA environment
    cp = None
    cuconvolve = None
    cuzoom = None
    GPU_IMPORT_ERROR = exc


@dataclass(slots=True)
class IOConfig:
    input_dir: Path
    output_dir: Path


@dataclass(slots=True)
class RuntimeConfig:
    opencv_num_threads: int
    parallel_njobs: int


@dataclass(slots=True)
class VideoConfig:
    fps: float
    microm_per_pixel: float | None


@dataclass(slots=True)
class ROIConfig:
    ball_radius: int
    roi_width: int
    roi_height: int
    margin: int
    prominence_value: float
    roi_ys: list[int]


@dataclass(slots=True)
class BeatConfig:
    th_denoise_opt: float
    th_max_noise_ratio: float
    th_frame: int
    th_frame_inter: int


@dataclass(slots=True)
class TensorConfig:
    size_x: int
    size_t: int


@dataclass(slots=True)
class ConvConfig:
    nt: int
    nx: int
    ny: int


@dataclass(slots=True)
class ArtifactConfig:
    save_artifacts: bool
    save_overlay_movies: bool
    save_raw_flow_xy: bool


@dataclass(slots=True)
class AppConfig:
    io: IOConfig
    runtime: RuntimeConfig
    video: VideoConfig
    roi: ROIConfig
    beat: BeatConfig
    tensor: TensorConfig
    conv: ConvConfig
    artifacts: ArtifactConfig


@dataclass(slots=True)
class ROIBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass(slots=True)
class VideoContext:
    video_path: Path
    output_dir: Path
    preview_image: np.ndarray
    fps: float


@dataclass(slots=True)
class DatasetPaths:
    root: Path
    repo_root: Path
    tensors_dir: Path
    artifacts_dir: Path
    manifest_path: Path


@dataclass(slots=True)
class FlowArtifacts:
    flow_x_full: np.ndarray
    flow_y_full: np.ndarray
    flow_mag_full: np.ndarray
    flow_ang_full: np.ndarray
    flow_x_resized: np.ndarray
    flow_y_resized: np.ndarray
    velocity: np.ndarray


@dataclass(slots=True)
class BeatDetection:
    threshold: float
    start_frames: np.ndarray
    end_frames: np.ndarray
    heart_rate_bpm: float | None

    @property
    def n_beats(self) -> int:
        return int(self.start_frames.size)


MANIFEST_COLUMNS = [
    "sample_id",
    "split",
    "path",
    "class",
    "class_id",
    "cond_raw",
    "roi_id",
    "beat_id",
]


def require_gpu() -> None:
    if GPU_IMPORT_ERROR is not None:
        raise ImportError(
            "CuPy is required for scripts/preprocess_gpu.py. "
            "Install one of the CUDA-specific packages listed in "
            "requirements_preprocess_gpu.txt."
        ) from GPU_IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU optical-flow preprocessing for BeatVecNet."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new samples even if a beat has already been exported. "
        "Default behavior is skip-existing.",
    )
    return parser.parse_args()


def parse_optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_config(config_path: str | Path) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    runtime_raw = raw["runtime"]
    artifacts_raw = raw.get("artifacts", {})

    cfg = AppConfig(
        io=IOConfig(
            input_dir=Path(raw["io"]["input_dir"]).expanduser(),
            output_dir=Path(raw["io"]["output_dir"]).expanduser(),
        ),
        runtime=RuntimeConfig(
            opencv_num_threads=int(
                runtime_raw.get(
                    "opencv_num_threads",
                    runtime_raw.get("num_worker_threads", 1),
                )
            ),
            parallel_njobs=int(
                runtime_raw.get(
                    "parallel_njobs",
                    runtime_raw.get("njobs", 1),
                )
            ),
        ),
        video=VideoConfig(
            fps=float(raw["video"]["fps"]),
            microm_per_pixel=parse_optional_float(raw["video"]["microm_per_pixel"]),
        ),
        roi=ROIConfig(
            ball_radius=int(raw["roi"]["ball_radius"]),
            roi_width=int(raw["roi"]["roi_width"]),
            roi_height=int(raw["roi"]["roi_height"]),
            margin=int(raw["roi"]["margin"]),
            prominence_value=float(raw["roi"]["prominence_value"]),
            roi_ys=[int(v) for v in raw["roi"]["roi_ys"]],
        ),
        beat=BeatConfig(
            th_denoise_opt=float(raw["beat"]["th_denoise_opt"]),
            th_max_noise_ratio=float(raw["beat"]["th_max_noise_ratio"]),
            th_frame=int(raw["beat"]["th_frame"]),
            th_frame_inter=int(raw["beat"]["th_frame_inter"]),
        ),
        tensor=TensorConfig(
            size_x=int(raw["tensor"]["size_x"]),
            size_t=int(raw["tensor"]["size_t"]),
        ),
        conv=ConvConfig(
            nt=int(raw["conv"]["nt"]),
            nx=int(raw["conv"]["nx"]),
            ny=int(raw["conv"]["ny"]),
        ),
        artifacts=ArtifactConfig(
            save_artifacts=bool(artifacts_raw.get("save_artifacts", False)),
            save_overlay_movies=bool(
                artifacts_raw.get("save_overlay_movies", False)
            ),
            save_raw_flow_xy=bool(
                artifacts_raw.get("save_raw_flow_xy", False)
            ),
        ),
    )

    if cfg.runtime.opencv_num_threads < 1:
        raise ValueError("runtime.opencv_num_threads must be >= 1")
    if cfg.runtime.parallel_njobs == 0:
        raise ValueError("runtime.parallel_njobs must not be 0")
    if cfg.conv.nt < 1 or cfg.conv.nt % 2 == 0:
        raise ValueError("conv.nt must be a positive odd number")

    return cfg


def discover_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    return sorted(input_dir.rglob("*.avi"))


def build_dataset_paths(output_root: Path) -> DatasetPaths:
    return DatasetPaths(
        root=output_root,
        repo_root=REPO_ROOT,
        tensors_dir=output_root / "tensors",
        artifacts_dir=output_root / "artifacts",
        manifest_path=output_root / "manifest.csv",
    )


def initialize_dataset_paths(dataset_paths: DatasetPaths) -> None:
    dataset_paths.root.mkdir(parents=True, exist_ok=True)
    dataset_paths.tensors_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)


def safe_relative_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def load_existing_manifest(dataset_paths: DatasetPaths) -> pd.DataFrame:
    if not dataset_paths.manifest_path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    df = pd.read_csv(dataset_paths.manifest_path)
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[MANIFEST_COLUMNS].copy()


def next_sample_index(
    dataset_paths: DatasetPaths,
    existing_manifest: pd.DataFrame,
) -> int:
    max_index = 0

    tensor_pattern = re.compile(r"^s(\d+)\.npy$")
    for tensor_path in dataset_paths.tensors_dir.glob("s*.npy"):
        match = tensor_pattern.match(tensor_path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    sample_pattern = re.compile(r"^s(\d+)$")
    if "sample_id" in existing_manifest.columns:
        for sample_id in existing_manifest["sample_id"].dropna().astype(str):
            match = sample_pattern.match(sample_id)
            if match:
                max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def write_manifest(
    dataset_paths: DatasetPaths,
    existing_manifest: pd.DataFrame,
    new_rows: list[dict[str, object]],
) -> None:
    new_manifest = pd.DataFrame(new_rows, columns=MANIFEST_COLUMNS)
    if existing_manifest.empty:
        final_manifest = new_manifest
    elif new_manifest.empty:
        final_manifest = existing_manifest
    else:
        final_manifest = pd.concat(
            [existing_manifest, new_manifest],
            ignore_index=True,
            sort=False,
        )

    if final_manifest.empty:
        final_manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)
    else:
        for column in MANIFEST_COLUMNS:
            if column not in final_manifest.columns:
                final_manifest[column] = pd.NA
        final_manifest = final_manifest[MANIFEST_COLUMNS]
        final_manifest = final_manifest.drop_duplicates(
            subset=["sample_id"],
            keep="first",
        ).reset_index(drop=True)

    final_manifest.to_csv(dataset_paths.manifest_path, index=False)


def tensor_manifest_path(tensor_path: Path, dataset_paths: DatasetPaths) -> str:
    try:
        return tensor_path.relative_to(dataset_paths.repo_root).as_posix()
    except ValueError:
        return tensor_path.relative_to(dataset_paths.root).as_posix()


def velocity_scale(video_cfg: VideoConfig, fps: float) -> float:
    if video_cfg.microm_per_pixel is None:
        return fps
    return video_cfg.microm_per_pixel * fps


def velocity_unit(video_cfg: VideoConfig) -> str:
    if video_cfg.microm_per_pixel is None:
        return "pixel/s"
    return "um/s"


def read_sample_marker(beat_dir: Path) -> str | None:
    marker_path = beat_dir / "sample_id.txt"
    if not marker_path.exists():
        return None
    sample_id = marker_path.read_text(encoding="utf-8").strip()
    return sample_id or None


def write_sample_marker(beat_dir: Path, sample_id: str) -> None:
    (beat_dir / "sample_id.txt").write_text(f"{sample_id}\n", encoding="utf-8")


def build_manifest_row(
    sample_id: str,
    tensor_path: Path,
    dataset_paths: DatasetPaths,
    roi_id: int,
    beat_id: int,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": "",
        "path": tensor_manifest_path(tensor_path, dataset_paths),
        "class": "",
        "class_id": None,
        "cond_raw": "",
        "roi_id": roi_id,
        "beat_id": beat_id,
    }


def infer_shooting_time(video_path: Path) -> str:
    parts = video_path.stem.rsplit("_", maxsplit=2)
    if len(parts) == 3:
        return parts[0]
    return video_path.stem


def build_video_output_dir(
    video_path: Path,
    cfg: AppConfig,
    dataset_paths: DatasetPaths,
) -> Path:
    relative_parent = safe_relative_path(video_path.parent, cfg.io.input_dir)
    return dataset_paths.artifacts_dir / relative_parent / infer_shooting_time(video_path)


def read_video_fps(video_path: Path, fallback: float) -> float:
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_ANY)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 0 else fallback


def load_preview_image(video_path: Path) -> np.ndarray:
    bmp_candidates = sorted(video_path.parent.glob("*.bmp"))
    if bmp_candidates:
        image = cv2.imread(str(bmp_candidates[0]), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_ANY)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"failed to read preview frame: {video_path}")

    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def build_video_context(
    video_path: Path,
    cfg: AppConfig,
    dataset_paths: DatasetPaths,
) -> VideoContext:
    output_dir = build_video_output_dir(video_path, cfg, dataset_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    return VideoContext(
        video_path=video_path,
        output_dir=output_dir,
        preview_image=load_preview_image(video_path),
        fps=read_video_fps(video_path, fallback=cfg.video.fps),
    )


def moving_average(signal: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(signal, kernel, mode="same")


def detect_rois(preview_image: np.ndarray, cfg: AppConfig) -> list[ROIBox]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.roi.ball_radius, cfg.roi.ball_radius),
    )
    background = cv2.morphologyEx(preview_image, cv2.MORPH_OPEN, kernel)
    subtract_image = cv2.subtract(preview_image, background)

    pixel_sum = subtract_image.sum(axis=0)
    averaged = moving_average(pixel_sum, cfg.roi.roi_width)
    peaks, _ = find_peaks(
        averaged,
        distance=cfg.roi.roi_width,
        prominence=cfg.roi.prominence_value,
    )

    image_height, image_width = subtract_image.shape
    rois: list[ROIBox] = []
    for peak in peaks:
        start_x = max(int(peak) - cfg.roi.roi_width // 2, 0)
        end_x = start_x + cfg.roi.roi_width
        if start_x < cfg.roi.margin:
            continue
        if end_x > image_width - cfg.roi.margin:
            continue

        for start_y in cfg.roi.roi_ys:
            if start_y < 0:
                continue
            if start_y + cfg.roi.roi_height > image_height:
                continue
            rois.append(
                ROIBox(
                    x=start_x,
                    y=int(start_y),
                    width=cfg.roi.roi_width,
                    height=cfg.roi.roi_height,
                )
            )

    return rois


def save_roi_reports(
    preview_image: np.ndarray,
    rois: Sequence[ROIBox],
    video_output_dir: Path,
    cfg: AppConfig,
) -> None:
    if not cfg.artifacts.save_artifacts:
        return

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.roi.ball_radius, cfg.roi.ball_radius),
    )
    background = cv2.morphologyEx(preview_image, cv2.MORPH_OPEN, kernel)
    subtract_image = cv2.subtract(preview_image, background)

    pixel_sum = subtract_image.sum(axis=0)
    averaged = moving_average(pixel_sum, cfg.roi.roi_width)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pixel_sum, label="Pixel Intensity Sum")
    ax.plot(averaged, label="Moving Average", linestyle="--")
    ax.set_title("Pixel Intensity Sum and Moving Average")
    ax.set_xlabel("X-axis Position")
    ax.set_ylabel("Sum of Pixel Intensities")
    ax.legend()
    fig.savefig(video_output_dir / "detected_patterns.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(preview_image, cmap="gray")
    for roi in rois:
        rect = plt.Rectangle(
            (roi.x, roi.y),
            roi.width,
            roi.height,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        ax.add_patch(rect)
    ax.set_title("Original Image with ROIs Marked")
    ax.axis("off")
    fig.savefig(video_output_dir / "ROI_merge.pdf", bbox_inches="tight")
    plt.close(fig)

    roi_df = pd.DataFrame(
        {
            "ROI X": [roi.x for roi in rois],
            "ROI Y": [roi.y for roi in rois],
            "ROI W": [roi.width for roi in rois],
            "ROI H": [roi.height for roi in rois],
        }
    )
    roi_df.to_csv(video_output_dir / "roi_coordinates.csv", index=False)


def extract_roi_movies(
    video_path: Path,
    rois: Sequence[ROIBox],
    output_dir: Path,
    fps: float,
    save_allbeats_movie: bool,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    roi_dirs = [output_dir / f"ROI {idx + 1}" for idx in range(len(rois))]
    for roi_dir in roi_dirs:
        roi_dir.mkdir(parents=True, exist_ok=True)

    writers: list[cv2.VideoWriter | None] = []
    for roi, roi_dir in zip(rois, roi_dirs):
        if save_allbeats_movie:
            writers.append(
                cv2.VideoWriter(
                    str(roi_dir / "video_allbeats.mp4"),
                    fourcc,
                    fps,
                    (roi.width, roi.height),
                )
            )
        else:
            writers.append(None)
    buffers: list[list[np.ndarray]] = [[] for _ in rois]

    progress = tqdm(
        total=frame_count if frame_count > 0 else None,
        desc=f"Exporting {video_path.name}",
        unit="frame",
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame.ndim == 3:
                frame_color = frame
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            for idx, roi in enumerate(rois):
                cropped_color = frame_color[roi.y : roi.y2, roi.x : roi.x2]
                cropped_gray = frame_gray[roi.y : roi.y2, roi.x : roi.x2]
                if writers[idx] is not None:
                    writers[idx].write(cropped_color)
                buffers[idx].append(np.ascontiguousarray(cropped_gray))

            progress.update(1)
    finally:
        progress.close()
        cap.release()
        for writer in writers:
            if writer is not None:
                writer.release()

    roi_stacks: list[np.ndarray] = []
    for buffer, roi in zip(buffers, rois):
        if buffer:
            roi_stacks.append(np.stack(buffer, axis=0))
        else:
            roi_stacks.append(
                np.empty((0, roi.height, roi.width), dtype=np.uint8)
            )
    return roi_stacks


def calc_optical_flow_pair(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame,
        next_frame,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.1,
        0,
    )
    return np.transpose(flow.astype(np.float32, copy=False), (2, 0, 1))


def compute_optical_flow_stack(roi_frames: np.ndarray, njobs: int) -> np.ndarray:
    if roi_frames.shape[0] < 2:
        raise ValueError("at least two frames are required to compute optical flow")

    if njobs == 1:
        flows = [
            calc_optical_flow_pair(roi_frames[i], roi_frames[i + 1])
            for i in tqdm(
                range(roi_frames.shape[0] - 1),
                desc="Optical flow",
                unit="pair",
            )
        ]
    else:
        flows = Parallel(n_jobs=njobs, backend="threading")(
            delayed(calc_optical_flow_pair)(roi_frames[i], roi_frames[i + 1])
            for i in range(roi_frames.shape[0] - 1)
        )
    return np.stack(flows, axis=0)


def align_temporal_padding(arr: "cp.ndarray", kernel_t: int) -> "cp.ndarray":
    if kernel_t == 1:
        return arr

    half_window = (kernel_t - 1) // 2
    if half_window == 0:
        return arr

    trimmed = arr[half_window:-half_window]
    padding = cp.zeros((kernel_t - 1, *arr.shape[1:]), dtype=arr.dtype)
    return cp.concatenate((padding, trimmed), axis=0)


def smooth_threshold_and_resize(
    flow_raw: np.ndarray,
    cfg: AppConfig,
    fps: float,
) -> FlowArtifacts:
    require_gpu()

    kernel = cp.ones(
        (cfg.conv.nt, cfg.conv.nx, cfg.conv.ny),
        dtype=cp.float32,
    ) / float(cfg.conv.nt * cfg.conv.nx * cfg.conv.ny)
    flow_gpu = cp.asarray(flow_raw, dtype=cp.float32)

    flow_x_full = cuconvolve(flow_gpu[:, 0], kernel, mode="constant")
    flow_y_full = cuconvolve(flow_gpu[:, 1], kernel, mode="constant")
    flow_x_full = align_temporal_padding(flow_x_full, cfg.conv.nt)
    flow_y_full = align_temporal_padding(flow_y_full, cfg.conv.nt)

    flow_mag_full = cp.sqrt(flow_x_full**2 + flow_y_full**2)
    keep_mask = flow_mag_full >= cfg.beat.th_denoise_opt
    flow_x_full = cp.where(keep_mask, flow_x_full, 0)
    flow_y_full = cp.where(keep_mask, flow_y_full, 0)
    flow_mag_full = cp.sqrt(flow_x_full**2 + flow_y_full**2)
    flow_ang_full = cp.arctan2(flow_y_full, flow_x_full)

    resize_scale = cfg.tensor.size_x / float(flow_x_full.shape[2])
    flow_x_resized = cuzoom(flow_x_full, (1, 1, resize_scale), order=1)
    flow_y_resized = cuzoom(flow_y_full, (1, 1, resize_scale), order=1)

    velocity = (
        cp.mean(flow_mag_full, axis=(1, 2)).get()
        * velocity_scale(cfg.video, fps)
    )

    return FlowArtifacts(
        flow_x_full=cp.asnumpy(flow_x_full),
        flow_y_full=cp.asnumpy(flow_y_full),
        flow_mag_full=cp.asnumpy(flow_mag_full),
        flow_ang_full=cp.asnumpy(flow_ang_full),
        flow_x_resized=cp.asnumpy(flow_x_resized),
        flow_y_resized=cp.asnumpy(flow_y_resized),
        velocity=np.asarray(velocity, dtype=np.float32),
    )


def merge_close_candidates(
    candidate_indices: np.ndarray,
    max_index_gap: int,
) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    group_start = int(candidate_indices[0])
    group_end = int(candidate_indices[0])

    for idx in candidate_indices[1:]:
        idx = int(idx)
        if idx - group_end < max_index_gap:
            group_end = idx
            continue
        groups.append((group_start, group_end))
        group_start = idx
        group_end = idx

    groups.append((group_start, group_end))
    return groups


def empty_detection(threshold: float) -> BeatDetection:
    return BeatDetection(
        threshold=float(threshold),
        start_frames=np.array([], dtype=int),
        end_frames=np.array([], dtype=int),
        heart_rate_bpm=None,
    )


def detect_beats(velocity: np.ndarray, cfg: AppConfig, fps: float) -> BeatDetection:
    hist, bin_edges = np.histogram(velocity, bins=100)
    threshold_idx = min(int(np.argmax(hist)) + 1, len(bin_edges) - 1)
    threshold = float(bin_edges[threshold_idx] + 1e-3)

    low_motion = np.flatnonzero(velocity < threshold)
    if low_motion.size == 0:
        return empty_detection(threshold)

    if low_motion[0] > 0:
        low_motion = np.insert(low_motion, 0, 0)
    if low_motion[-1] < len(velocity) - 1:
        low_motion = np.append(low_motion, len(velocity) - 1)

    gaps = np.diff(low_motion)
    candidate_gaps = np.flatnonzero(gaps > cfg.beat.th_frame)
    if candidate_gaps.size == 0:
        return empty_detection(threshold)

    merged = merge_close_candidates(candidate_gaps, cfg.beat.th_frame_inter)
    start_frames = np.array([low_motion[start] for start, _ in merged], dtype=int)
    end_frames = np.array(
        [low_motion[end] + gaps[end] + 1 for _, end in merged],
        dtype=int,
    )

    edge_margin = max(cfg.conv.nt - 1, 1)
    if start_frames.size and start_frames[0] < edge_margin:
        start_frames = start_frames[1:]
        end_frames = end_frames[1:]
    if start_frames.size and end_frames[-1] > len(velocity) - edge_margin:
        start_frames = start_frames[:-1]
        end_frames = end_frames[:-1]

    if start_frames.size == 0:
        return empty_detection(threshold)

    keep_mask = []
    eps = 1e-6
    min_duration = cfg.beat.th_frame * 2 + cfg.beat.th_frame_inter
    for start, end in zip(start_frames, end_frames):
        if end <= start:
            keep_mask.append(False)
            continue
        noise_ratio = float(np.max(velocity[start:end]) / max(threshold, eps))
        duration = int(end - start)
        keep_mask.append(
            noise_ratio > cfg.beat.th_max_noise_ratio and duration > min_duration
        )

    keep = np.asarray(keep_mask, dtype=bool)
    start_frames = start_frames[keep]
    end_frames = end_frames[keep]
    if start_frames.size == 0:
        return empty_detection(threshold)

    heart_rate_bpm: float | None = None
    if start_frames.size > 1:
        mean_interval = float(np.mean(np.diff(start_frames)))
        if mean_interval > 0:
            heart_rate_bpm = 60.0 / (mean_interval / fps)

    return BeatDetection(
        threshold=threshold,
        start_frames=start_frames,
        end_frames=end_frames,
        heart_rate_bpm=heart_rate_bpm,
    )


def draw_flow_overlay(
    frame_gray: np.ndarray,
    flow_xy: np.ndarray,
    step: int = 16,
    scale: float = 50.0,
) -> np.ndarray:
    h, w = frame_gray.shape[:2]
    y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1)
    fx, fy = flow_xy[y, x].T * scale
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2).astype(np.int32)
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    for line in lines:
        cv2.arrowedLine(
            vis,
            pt1=tuple(line[0]),
            pt2=tuple(line[1]),
            color=(38, 163, 233),
            thickness=1,
            tipLength=0.2,
        )
    return vis


def resize_time_axis(arr: np.ndarray, size_t: int) -> np.ndarray:
    if arr.shape[0] == size_t:
        return arr.astype(np.float32, copy=False)

    scale = size_t / float(arr.shape[0])
    resized = ndimage.zoom(arr, (scale, 1, 1), order=1)
    if resized.shape[0] != size_t:
        resized = resized[:size_t]
        if resized.shape[0] < size_t:
            pad_len = size_t - resized.shape[0]
            pad = np.repeat(resized[-1:], pad_len, axis=0)
            resized = np.concatenate([resized, pad], axis=0)
    return np.asarray(resized, dtype=np.float32)


def build_model_input_tensor(flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    x_cht = np.transpose(flow_x, (2, 1, 0))
    y_cht = np.transpose(flow_y, (2, 1, 0))
    tensor = np.concatenate([x_cht, y_cht], axis=0)
    return np.asarray(tensor, dtype=np.float32)


def save_detected_beats_plot(
    velocity: np.ndarray,
    detection: BeatDetection,
    fps: float,
    out_path: Path,
    velocity_unit_label: str,
) -> None:
    time_axis = np.arange(len(velocity), dtype=np.float32) / fps
    fig_width = max(6.0, float(max(detection.n_beats, 1) * 2))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.plot(time_axis, velocity, color="C1", alpha=0.6, label="velocity")
    ax.axhline(
        detection.threshold,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="threshold",
    )
    for start, end in zip(detection.start_frames, detection.end_frames):
        ax.axvspan(start / fps, end / fps, color="red", alpha=0.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"velocity [{velocity_unit_label}]")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_angle_reports(
    flow_ang: np.ndarray,
    flow_mag: np.ndarray,
    velocity: np.ndarray,
    fps: float,
    out_dir: Path,
    velocity_unit_label: str,
) -> None:
    tpoint, ypoint, xpoint = flow_ang.shape
    time_flat = np.repeat(np.arange(tpoint, dtype=np.float32) / fps, ypoint * xpoint)
    angle_flat = np.rad2deg(flow_ang.reshape(-1))
    mag_flat = flow_mag.reshape(-1)
    valid = mag_flat > 0

    time_valid = time_flat[valid]
    angle_valid = angle_flat[valid]
    angle_bins = np.arange(-180, 190, 10)
    time_bins = np.arange(tpoint + 1, dtype=np.float32) / fps

    fig, ax = plt.subplots()
    hist = ax.hist2d(time_valid, angle_valid, bins=(time_bins, angle_bins), cmap=cm.jet)
    ax.set_title("2D-histogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degree)")
    fig.colorbar(hist[3], ax=ax)
    fig.savefig(out_dir / "2dhist_angle.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(velocity), dtype=np.float32) / fps, velocity, color="black")
    ax.grid(True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Speed ({velocity_unit_label})")
    fig.savefig(out_dir / "1dtimec_mag.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(angle_valid, bins=angle_bins, color="gray")
    ax.grid(True)
    ax.set_xlabel("Angle (degree)")
    fig.savefig(out_dir / "1dhist_mag.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        2,
        gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1, 2]},
        figsize=(10, 8),
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    axes[0, 0].plot(
        np.arange(len(velocity), dtype=np.float32) / fps,
        velocity,
        color="black",
    )
    axes[0, 0].set_xlim([0, tpoint / fps])
    axes[0, 0].set_xticks([])
    axes[0, 0].set_ylabel(f"Speed ({velocity_unit_label})")
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    axes[0, 1].set_axis_off()

    combined = axes[1, 0].hist2d(
        time_valid,
        angle_valid,
        bins=(time_bins, angle_bins),
        cmap=cm.jet,
    )
    axes[1, 0].set_xlim([0, tpoint / fps])
    axes[1, 0].set_ylim([-180, 180])
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Angle (degree)")

    axes[1, 1].hist(angle_valid, bins=angle_bins, color="gray", orientation="horizontal")
    axes[1, 1].set_ylim([-180, 180])
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel("frequency")
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    fig.colorbar(combined[3], ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(out_dir / "2dhist_combined.pdf", bbox_inches="tight")
    plt.close(fig)


def save_beat_outputs(
    beat_index: int,
    roi_frames: np.ndarray,
    flow: FlowArtifacts,
    detection: BeatDetection,
    fps: float,
    cfg: AppConfig,
    roi_dir: Path,
    dataset_paths: DatasetPaths,
    roi_id: int,
    sample_id: str,
) -> dict[str, object]:
    start = int(detection.start_frames[beat_index])
    end = int(detection.end_frames[beat_index])
    beat_dir = roi_dir / f"beat_{beat_index + 1}"
    beat_dir.mkdir(parents=True, exist_ok=True)
    write_sample_marker(beat_dir, sample_id)

    raw_x = np.asarray(flow.flow_x_resized[start:end], dtype=np.float32)
    raw_y = np.asarray(flow.flow_y_resized[start:end], dtype=np.float32)
    resampled_x = resize_time_axis(raw_x, cfg.tensor.size_t)
    resampled_y = resize_time_axis(raw_y, cfg.tensor.size_t)
    model_input = build_model_input_tensor(resampled_x, resampled_y)

    tensor_path = dataset_paths.tensors_dir / f"{sample_id}.npy"
    np.save(tensor_path, model_input)

    if cfg.artifacts.save_artifacts:
        beat_velocity = flow.velocity[start:end]
        np.save(beat_dir / "v.npy", np.asarray(beat_velocity, dtype=np.float32))

        if cfg.artifacts.save_raw_flow_xy:
            np.save(beat_dir / f"flow_x_beat_{beat_index + 1}.npy", raw_x)
            np.save(beat_dir / f"flow_y_beat_{beat_index + 1}.npy", raw_y)
            np.save(
                beat_dir / f"flow_x_beat_{beat_index + 1}_t_{cfg.tensor.size_t}.npy",
                resampled_x,
            )
            np.save(
                beat_dir / f"flow_y_beat_{beat_index + 1}_t_{cfg.tensor.size_t}.npy",
                resampled_y,
            )

        if cfg.artifacts.save_overlay_movies:
            overlay_path = beat_dir / f"video_beat{beat_index + 1}.mp4"
            writer = cv2.VideoWriter(
                str(overlay_path),
                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                max(fps / 10.0, 1.0),
                (roi_frames.shape[2], roi_frames.shape[1]),
            )
            try:
                for frame_idx in range(start, end):
                    flow_xy = np.stack(
                        [flow.flow_x_full[frame_idx], flow.flow_y_full[frame_idx]],
                        axis=-1,
                    )
                    writer.write(draw_flow_overlay(roi_frames[frame_idx], flow_xy))
            finally:
                writer.release()

        cv2.imwrite(str(beat_dir / f"video_beat{beat_index + 1}.png"), roi_frames[start])
        save_angle_reports(
            flow_ang=flow.flow_ang_full[start:end],
            flow_mag=flow.flow_mag_full[start:end],
            velocity=beat_velocity,
            fps=fps,
            out_dir=beat_dir,
            velocity_unit_label=velocity_unit(cfg.video),
        )

    return build_manifest_row(
        sample_id=sample_id,
        tensor_path=tensor_path,
        dataset_paths=dataset_paths,
        roi_id=roi_id,
        beat_id=beat_index + 1,
    )


def process_roi(
    roi_frames: np.ndarray,
    roi_dir: Path,
    fps: float,
    cfg: AppConfig,
    dataset_paths: DatasetPaths,
    video_path: Path,
    roi_id: int,
    sample_index: int,
    append_mode: bool,
    known_sample_ids: set[str],
) -> tuple[int, list[dict[str, object]]]:
    if roi_frames.shape[0] < 2:
        print(f"  {roi_dir.name}: skipped, not enough frames")
        return sample_index, []

    print(f"  {roi_dir.name}: calculating optical flow")
    flow_raw = compute_optical_flow_stack(roi_frames, cfg.runtime.parallel_njobs)

    print(f"  {roi_dir.name}: smoothing and denoising on GPU")
    flow = smooth_threshold_and_resize(flow_raw, cfg, fps=fps)

    if cfg.artifacts.save_artifacts:
        np.save(roi_dir / "v_alltime.npy", flow.velocity)
    detection = detect_beats(flow.velocity, cfg, fps=fps)
    if cfg.artifacts.save_artifacts and detection.heart_rate_bpm is not None:
        np.save(
            roi_dir / "heart_rate.npy",
            np.asarray(detection.heart_rate_bpm, dtype=np.float32),
        )

    if cfg.artifacts.save_artifacts:
        save_detected_beats_plot(
            velocity=flow.velocity,
            detection=detection,
            fps=fps,
            out_path=roi_dir / "detected_beats.pdf",
            velocity_unit_label=velocity_unit(cfg.video),
        )

    if detection.n_beats == 0:
        print(f"  {roi_dir.name}: no beats detected")
        del flow_raw, flow
        return sample_index, []

    print(f"  {roi_dir.name}: detected {detection.n_beats} beats")
    manifest_rows: list[dict[str, object]] = []
    for beat_index in range(detection.n_beats):
        beat_dir = roi_dir / f"beat_{beat_index + 1}"
        if not append_mode:
            existing_sample_id = read_sample_marker(beat_dir)
            if existing_sample_id is not None:
                tensor_path = dataset_paths.tensors_dir / f"{existing_sample_id}.npy"
                if tensor_path.exists():
                    if existing_sample_id not in known_sample_ids:
                        manifest_rows.append(
                            build_manifest_row(
                                sample_id=existing_sample_id,
                                tensor_path=tensor_path,
                                dataset_paths=dataset_paths,
                                roi_id=roi_id,
                                beat_id=beat_index + 1,
                            )
                        )
                        known_sample_ids.add(existing_sample_id)
                    print(
                        f"  {roi_dir.name}: beat {beat_index + 1} already exported "
                        f"as {existing_sample_id}, skipping"
                    )
                    continue

        sample_id = f"s{sample_index:04d}"
        manifest_rows.append(
            save_beat_outputs(
                beat_index=beat_index,
                roi_frames=roi_frames,
                flow=flow,
                detection=detection,
                fps=fps,
                cfg=cfg,
                roi_dir=roi_dir,
                dataset_paths=dataset_paths,
                roi_id=roi_id,
                sample_id=sample_id,
            )
        )
        known_sample_ids.add(sample_id)
        sample_index += 1

    del flow_raw, flow
    gc.collect()
    return sample_index, manifest_rows


def process_video(
    video_path: Path,
    cfg: AppConfig,
    dataset_paths: DatasetPaths,
    sample_index: int,
    append_mode: bool,
    known_sample_ids: set[str],
) -> tuple[int, list[dict[str, object]]]:
    context = build_video_context(video_path, cfg, dataset_paths)
    print(f"\nProcessing {video_path}")
    print(f"Output directory: {context.output_dir}")

    rois = detect_rois(context.preview_image, cfg)
    save_roi_reports(context.preview_image, rois, context.output_dir, cfg)
    if not rois:
        print("No ROIs detected.")
        return sample_index, []

    print(f"Detected {len(rois)} ROIs")
    roi_frames_list = extract_roi_movies(
        video_path=context.video_path,
        rois=rois,
        output_dir=context.output_dir,
        fps=context.fps,
        save_allbeats_movie=cfg.artifacts.save_artifacts,
    )

    manifest_rows: list[dict[str, object]] = []
    for idx, roi_frames in enumerate(roi_frames_list):
        roi_dir = context.output_dir / f"ROI {idx + 1}"
        sample_index, roi_rows = process_roi(
            roi_frames=roi_frames,
            roi_dir=roi_dir,
            fps=context.fps,
            cfg=cfg,
            dataset_paths=dataset_paths,
            video_path=context.video_path,
            roi_id=idx + 1,
            sample_index=sample_index,
            append_mode=append_mode,
            known_sample_ids=known_sample_ids,
        )
        manifest_rows.extend(roi_rows)

    return sample_index, manifest_rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    require_gpu()

    dataset_paths = build_dataset_paths(cfg.io.output_dir)
    initialize_dataset_paths(dataset_paths)
    cv2.setNumThreads(cfg.runtime.opencv_num_threads)

    videos = discover_videos(cfg.io.input_dir)
    if not videos:
        print(f"No .avi files found under {cfg.io.input_dir}")
        return

    print(f"Found {len(videos)} video(s) in {cfg.io.input_dir}")
    existing_manifest = load_existing_manifest(dataset_paths)
    sample_index = next_sample_index(dataset_paths, existing_manifest)
    start_sample_index = sample_index
    known_sample_ids = set(existing_manifest["sample_id"].dropna().astype(str))
    new_rows: list[dict[str, object]] = []

    for video_path in videos:
        sample_index, video_rows = process_video(
            video_path=video_path,
            cfg=cfg,
            dataset_paths=dataset_paths,
            sample_index=sample_index,
            append_mode=args.append,
            known_sample_ids=known_sample_ids,
        )
        new_rows.extend(video_rows)

    write_manifest(dataset_paths, existing_manifest, new_rows)

    print(
        f"Saved {sample_index - start_sample_index} new tensor sample(s) "
        f"to {dataset_paths.tensors_dir}"
    )
    print(f"Updated manifest: {dataset_paths.manifest_path}")
    print("\nAll analysis completed.")


if __name__ == "__main__":
    main()
