#!/usr/bin/env python
"""Convert locally collected rollout data to a LeRobot HF dataset and push it.

The generated layout mirrors datasets such as:

  data/train-00000-of-00001.parquet
  meta_data/info.json
  meta_data/stats.safetensors
  meta_data/episode_data_index.safetensors
  videos/observation.images.<camera>_episode_000000.mp4

Edit the defaults in build_arg_parser(), then run this file directly from the IDE.
Command line arguments can still override those defaults.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import imageio.v2 as imageio
import numpy as np
import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
LEROBOT_ROOT = ROOT.parent / "lerobot"
for path in (ROOT, LEROBOT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lerobot.common.datasets.utils import calculate_episode_data_index, flatten_dict  # noqa: E402
from lerobot.common.datasets.video_utils import VideoFrame  # noqa: E402
from lerobot.common.datasets.video_utils import encode_video_frames  # noqa: E402
from lerobot.common.datasets.push_dataset_to_hub.utils import (  # noqa: E402
    get_default_encoding,
    save_images_concurrently,
)

DEFAULT_ENCODING = get_default_encoding()
_ENCODING_CHECKED = False


def ffmpeg_has_encoder(vcodec: str) -> bool:
    """Return True if the local ffmpeg build exposes the requested encoder."""
    if not vcodec or shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and vcodec in result.stdout


def get_runtime_encoding() -> dict[str, Any]:
    """Use LeRobot's default encoding, with a local ffmpeg-compatible fallback."""
    global DEFAULT_ENCODING, _ENCODING_CHECKED

    if _ENCODING_CHECKED:
        return DEFAULT_ENCODING

    encoding = dict(DEFAULT_ENCODING)
    vcodec = str(encoding.get("vcodec", ""))
    if vcodec and not ffmpeg_has_encoder(vcodec):
        if ffmpeg_has_encoder("libx264"):
            encoding.update({"vcodec": "libx264", "pix_fmt": "yuv420p", "g": 2, "crf": 23})
            logging.warning(
                "当前 ffmpeg 不支持 LeRobot 默认视频编码器 %s，已自动改用 %s。",
                vcodec,
                encoding["vcodec"],
            )
        else:
            raise RuntimeError(
                f"当前 ffmpeg 不支持 LeRobot 默认视频编码器 {vcodec!r}，也未检测到 libx264。"
                "请安装带 libsvtav1 或 libx264 的 ffmpeg，或手动修改 DEFAULT_ENCODING。"
            )

    DEFAULT_ENCODING = encoding
    _ENCODING_CHECKED = True
    return DEFAULT_ENCODING


@dataclass
class EpisodeRecord:
    source_dir: Path
    source_name: str
    info: dict[str, Any]
    arrays: dict[str, np.ndarray]


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=4, ensure_ascii=False)


def resolve_raw_dir(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    if (raw_dir / "metadata.json").exists() and (raw_dir / "episodes").exists():
        return raw_dir

    candidates = [
        path
        for path in raw_dir.glob("collect_*")
        if (path / "metadata.json").exists() and (path / "episodes").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find a collect run under {raw_dir}. Expected metadata.json and episodes/."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    logging.info("Using latest collect run: %s", candidates[0])
    return candidates[0]


def local_dataset_dir(raw_dir: Path, output_dir: str | Path, repo_id: str | None) -> Path:
    output_dir = Path(output_dir)
    if repo_id:
        name = repo_id.split("/", 1)[-1]
    else:
        name = raw_dir.name
    return output_dir / name


def list_episode_dirs(raw_dir: Path) -> list[Path]:
    episodes_dir = raw_dir / "episodes"
    episode_dirs = [
        path
        for path in episodes_dir.glob("episode_*")
        if path.is_dir() and not path.name.endswith(".tmp") and (path / "arrays.npz").exists()
    ]
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* folders with arrays.npz found in {episodes_dir}.")
    return sorted(episode_dirs)


def load_episode(episode_dir: Path) -> EpisodeRecord:
    info = read_json(episode_dir / "info.json")
    with np.load(episode_dir / "arrays.npz", allow_pickle=False) as npz:
        arrays = {key: np.asarray(npz[key]) for key in npz.files}
    return EpisodeRecord(
        source_dir=episode_dir,
        source_name=episode_dir.name,
        info=info,
        arrays=arrays,
    )


def select_episodes(
    raw_dir: Path,
    success_only: bool,
    max_episodes: int | None,
) -> list[EpisodeRecord]:
    records = []
    for episode_dir in list_episode_dirs(raw_dir):
        record = load_episode(episode_dir)
        if success_only and not bool(record.info.get("success", False)):
            continue
        records.append(record)
        if max_episodes is not None and len(records) >= max_episodes:
            break

    if not records:
        raise RuntimeError("No episodes selected. Check --success-only and --max-episodes.")
    return records


def infer_state_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
    preferred = ("observation_state", "obs__agent_pos", "obs__observation__state")
    for key in preferred:
        if key in arrays:
            return np.asarray(arrays[key], dtype=np.float32)

    obs_keys = [key for key in arrays if key.startswith("obs__")]
    if obs_keys:
        logging.warning("Using fallback state key %s.", obs_keys[0])
        return np.asarray(arrays[obs_keys[0]], dtype=np.float32)

    raise KeyError("Cannot find observation state in arrays.npz.")


def infer_cameras(records: list[EpisodeRecord], explicit_cameras: str | None) -> list[str]:
    if explicit_cameras:
        return [item.strip() for item in explicit_cameras.split(",") if item.strip()]

    cameras = set()
    for record in records:
        image_dirs = record.info.get("image_observation_dirs", {})
        for key, rel_path in image_dirs.items():
            if key.startswith("pixels."):
                cameras.add(key.split(".", 1)[1])
            else:
                cameras.add(Path(rel_path).name)

        videos_dir = record.source_dir / "videos"
        if videos_dir.exists():
            cameras.update(path.stem for path in videos_dir.glob("*.mp4"))

    if not cameras:
        raise RuntimeError("No cameras found in episode info or videos/ folders.")
    return sorted(cameras)


def frame_count_for(record: EpisodeRecord) -> int:
    arrays = record.arrays
    state = infer_state_array(arrays)
    action = np.asarray(arrays["action"])
    frame_count = min(len(state), len(action))
    for optional in ("timestamp", "terminated", "truncated", "frame_index"):
        if optional in arrays:
            frame_count = min(frame_count, len(arrays[optional]))
    if frame_count <= 0:
        raise RuntimeError(f"{record.source_dir} has no frames.")
    return int(frame_count)


def copy_or_encode_video(
    record: EpisodeRecord,
    camera: str,
    episode_index: int,
    videos_dir: Path,
    fps: int,
    overwrite: bool,
) -> str:
    video_name = f"observation.images.{camera}_episode_{episode_index:06d}.mp4"
    dst = videos_dir / video_name
    if dst.exists() and not overwrite:
        return f"videos/{video_name}"

    images_dir = record.source_dir / "images" / camera
    frame_paths = sorted(images_dir.glob("*.jpg"))
    if frame_paths:
        # 对齐 LeRobot 官方 push_dataset_to_hub 流程：
        # 先生成 frame_%06d.png，再用 encode_video_frames 写入带短 GOP 的训练友好视频。
        frames = np.stack([imageio.imread(frame_path) for frame_path in frame_paths], axis=0)
        encoding = get_runtime_encoding()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_imgs_dir = Path(tmp_dir) / "frames"
            save_images_concurrently(frames, tmp_imgs_dir)
            encode_video_frames(
                tmp_imgs_dir,
                dst,
                fps,
                overwrite=True,
                **encoding,
            )
        return f"videos/{video_name}"

    src = record.source_dir / "videos" / f"{camera}.mp4"
    if src.exists():
        shutil.copy2(src, dst)
        return f"videos/{video_name}"

    raise FileNotFoundError(f"Missing both {src} and image frames under {images_dir}.")


def make_hf_features(cameras: list[str], state_dim: int, action_dim: int) -> datasets.Features:
    features: dict[str, Any] = {}
    for camera in cameras:
        features[f"observation.images.{camera}"] = VideoFrame()
    features["observation.state"] = datasets.Sequence(
        datasets.Value("float32"), length=state_dim
    )
    features["action"] = datasets.Sequence(datasets.Value("float32"), length=action_dim)
    features["episode_index"] = datasets.Value("int64")
    features["frame_index"] = datasets.Value("int64")
    features["timestamp"] = datasets.Value("float32")
    features["next.done"] = datasets.Value("bool")
    features["index"] = datasets.Value("int64")
    return datasets.Features(features)


def build_rows_and_videos(
    records: list[EpisodeRecord],
    cameras: list[str],
    videos_dir: Path,
    fps: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    global_index = 0

    videos_dir.mkdir(parents=True, exist_ok=True)
    for episode_index, record in enumerate(records):
        frame_count = frame_count_for(record)
        state = infer_state_array(record.arrays)[:frame_count].astype(np.float32, copy=False)
        action = record.arrays["action"][:frame_count].astype(np.float32, copy=False)
        timestamps = record.arrays.get("timestamp")
        if timestamps is None:
            timestamps = np.arange(frame_count, dtype=np.float32) / float(fps)
        timestamps = np.asarray(timestamps[:frame_count], dtype=np.float32)

        terminated = np.asarray(record.arrays.get("terminated", np.zeros(frame_count)), dtype=bool)
        truncated = np.asarray(record.arrays.get("truncated", np.zeros(frame_count)), dtype=bool)
        done_flags = np.logical_or(terminated[:frame_count], truncated[:frame_count])
        done_flags[-1] = True

        video_paths = {
            camera: copy_or_encode_video(
                record=record,
                camera=camera,
                episode_index=episode_index,
                videos_dir=videos_dir,
                fps=fps,
                overwrite=overwrite,
            )
            for camera in cameras
        }

        all_states.append(state)
        all_actions.append(action)
        for frame_index in range(frame_count):
            timestamp = float(timestamps[frame_index])
            row: dict[str, Any] = {
                "observation.state": state[frame_index].tolist(),
                "action": action[frame_index].tolist(),
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "timestamp": timestamp,
                "next.done": bool(done_flags[frame_index]),
                "index": int(global_index),
            }
            for camera in cameras:
                row[f"observation.images.{camera}"] = {
                    "path": video_paths[camera],
                    "timestamp": timestamp,
                }
            rows.append(row)
            global_index += 1

        logging.info(
            "Converted %s -> episode_index=%06d frames=%d success=%s reward=%.2f",
            record.source_name,
            episode_index,
            frame_count,
            bool(record.info.get("success", False)),
            float(record.info.get("reward", 0.0)),
        )

    return rows, all_states, all_actions


def vector_stats(values: np.ndarray) -> dict[str, torch.Tensor]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "mean": torch.from_numpy(array.mean(axis=0).astype(np.float32)),
        "std": torch.from_numpy(np.maximum(array.std(axis=0), 1e-6).astype(np.float32)),
        "min": torch.from_numpy(array.min(axis=0).astype(np.float32)),
        "max": torch.from_numpy(array.max(axis=0).astype(np.float32)),
    }


def scalar_column(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def find_image_frames(records: list[EpisodeRecord], camera: str) -> list[Path]:
    frames: list[Path] = []
    for record in records:
        frames.extend(sorted((record.source_dir / "images" / camera).glob("*.jpg")))
    return frames


def image_stats(
    records: list[EpisodeRecord],
    camera: str,
    max_frames: int,
) -> dict[str, torch.Tensor]:
    frame_paths = find_image_frames(records, camera)
    if not frame_paths or max_frames == 0:
        mean = torch.full((3, 1, 1), 0.5, dtype=torch.float32)
        std = torch.full((3, 1, 1), 0.25, dtype=torch.float32)
        return {
            "mean": mean,
            "std": std,
            "min": torch.zeros((3, 1, 1), dtype=torch.float32),
            "max": torch.ones((3, 1, 1), dtype=torch.float32),
        }

    if max_frames > 0 and len(frame_paths) > max_frames:
        indices = np.linspace(0, len(frame_paths) - 1, num=max_frames, dtype=np.int64)
        frame_paths = [frame_paths[int(idx)] for idx in indices]

    count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)
    channel_min = np.full(3, np.inf, dtype=np.float64)
    channel_max = np.full(3, -np.inf, dtype=np.float64)

    for frame_path in frame_paths:
        image = np.asarray(imageio.imread(frame_path), dtype=np.float32) / 255.0
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        image = image[..., :3]
        flat = image.reshape(-1, 3)
        count += flat.shape[0]
        channel_sum += flat.sum(axis=0)
        channel_sumsq += np.square(flat).sum(axis=0)
        channel_min = np.minimum(channel_min, flat.min(axis=0))
        channel_max = np.maximum(channel_max, flat.max(axis=0))

    mean = channel_sum / max(1, count)
    var = np.maximum(channel_sumsq / max(1, count) - np.square(mean), 1e-12)
    std = np.sqrt(var)
    reshape = lambda value: torch.from_numpy(value.astype(np.float32)).view(3, 1, 1)
    return {
        "mean": reshape(mean),
        "std": reshape(std),
        "min": reshape(channel_min),
        "max": reshape(channel_max),
    }


def build_stats(
    rows: list[dict[str, Any]],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    records: list[EpisodeRecord],
    cameras: list[str],
    max_image_stat_frames: int,
) -> dict[str, dict[str, torch.Tensor]]:
    state_values = np.concatenate(states, axis=0)
    action_values = np.concatenate(actions, axis=0)
    stats: dict[str, dict[str, torch.Tensor]] = {
        "observation.state": vector_stats(state_values),
        "action": vector_stats(action_values),
        "episode_index": vector_stats(scalar_column([row["episode_index"] for row in rows])),
        "frame_index": vector_stats(scalar_column([row["frame_index"] for row in rows])),
        "timestamp": vector_stats(scalar_column([row["timestamp"] for row in rows])),
        "next.done": vector_stats(scalar_column([float(row["next.done"]) for row in rows])),
        "index": vector_stats(scalar_column([row["index"] for row in rows])),
    }
    for camera in cameras:
        stats[f"observation.images.{camera}"] = image_stats(
            records=records,
            camera=camera,
            max_frames=max_image_stat_frames,
        )
    return stats


def infer_fps(records: list[EpisodeRecord], metadata: dict[str, Any], override_fps: int | None) -> int:
    if override_fps is not None:
        return int(override_fps)
    if "fps" in metadata:
        return int(metadata["fps"])
    for record in records:
        if "fps" in record.info:
            return int(record.info["fps"])
    return 25


def build_info(fps: int, raw_dir: Path, records: list[EpisodeRecord], cameras: list[str]) -> dict[str, Any]:
    return {
        "codebase_version": "v1.6",
        "fps": int(fps),
        "video": 1,
        "encoding": DEFAULT_ENCODING,
        "source_raw_dir": str(raw_dir),
        "total_episodes": int(len(records)),
        "camera_keys": [f"observation.images.{camera}" for camera in cameras],
    }


def write_dataset_card(local_dir: Path, raw_dir: Path, repo_id: str | None) -> None:
    dataset_name = repo_id or raw_dir.name
    text = (
        "---\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "---\n"
        f"This dataset was converted from `{raw_dir}` for LeRobot training.\n"
        f"Dataset name: `{dataset_name}`.\n"
    )
    (local_dir / "README.md").write_text(text, encoding="utf-8")


def write_gitattributes(local_dir: Path) -> None:
    content = (
        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n"
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
    )
    (local_dir / ".gitattributes").write_text(content, encoding="utf-8")


def build_local_dataset(args: argparse.Namespace) -> Path:
    raw_dir = resolve_raw_dir(args.raw_dir)
    local_dir = Path(args.local_dir) if args.local_dir else local_dataset_dir(
        raw_dir, args.output_dir, args.repo_id
    )
    if local_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{local_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(local_dir)

    data_dir = local_dir / "data"
    meta_data_dir = local_dir / "meta_data"
    videos_dir = local_dir / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(raw_dir / "metadata.json") if (raw_dir / "metadata.json").exists() else {}
    records = select_episodes(
        raw_dir=raw_dir,
        success_only=args.success_only,
        max_episodes=args.max_episodes,
    )
    cameras = infer_cameras(records, args.cameras)
    fps = infer_fps(records, metadata, args.fps)
    logging.info("Selected %d episode(s), cameras=%s, fps=%d.", len(records), cameras, fps)

    rows, states, actions = build_rows_and_videos(
        records=records,
        cameras=cameras,
        videos_dir=videos_dir,
        fps=fps,
        overwrite=args.overwrite,
    )

    state_dim = int(states[0].shape[-1])
    action_dim = int(actions[0].shape[-1])
    # 将数据转换为 Hugging Face Dataset 格式并保存为 parquet 文件；视频文件已在 build_rows_and_videos 中处理好了。
    hf_dataset = datasets.Dataset.from_list(
        rows,
        features=make_hf_features(cameras=cameras, state_dim=state_dim, action_dim=action_dim),
    )
    parquet_path = data_dir / "train-00000-of-00001.parquet"
    hf_dataset.to_parquet(str(parquet_path))

    episode_data_index = calculate_episode_data_index(hf_dataset)
    save_file(episode_data_index, meta_data_dir / "episode_data_index.safetensors")
    stats = build_stats(
        rows=rows,
        states=states,
        actions=actions,
        records=records,
        cameras=cameras,
        max_image_stat_frames=args.max_image_stat_frames,
    )
    save_file(flatten_dict(stats), meta_data_dir / "stats.safetensors")

    info = build_info(fps=fps, raw_dir=raw_dir, records=records, cameras=cameras)
    write_json(meta_data_dir / "info.json", info)
    write_dataset_card(local_dir=local_dir, raw_dir=raw_dir, repo_id=args.repo_id)
    write_gitattributes(local_dir)

    logging.info("Wrote parquet: %s", parquet_path)
    logging.info("Wrote metadata: %s", meta_data_dir)
    logging.info("Wrote videos: %s", videos_dir)
    return local_dir


def push_dataset(local_dir: Path, args: argparse.Namespace) -> None:
    if not args.repo_id:
        raise ValueError("--repo-id is required when --push is set.")

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        commit_message=args.commit_message,
    )
    logging.info("Pushed dataset to https://huggingface.co/datasets/%s", args.repo_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 collect_data.py 采集的数据转换为 LeRobot/Hugging Face 数据集格式并上传。"
    )

    parser.add_argument(
        "--raw-dir",
        default="outputs/4_data_collect/collect_SewNeedle-3Arms-v0_3Arms_2026-06-02_11-52-01",
        help="采集数据目录；也可以填 outputs/4_data_collect，脚本会自动选择最新的 collect_* 目录。",
    )
    parser.add_argument(
        "--repo-id",
        default="Dc-dc/collect_sim_insert_peg_3arms",
        help="Hugging Face 数据集仓库名，格式为 用户名/数据集名。",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hf_datasets",
        help="本地生成的 HF 数据集根目录；local-dir 为 None 时会在这里创建数据集文件夹。",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="指定本地输出目录；None 表示自动使用 output-dir/数据集名。",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="本地输出目录已存在时是否覆盖重建；可用 --no-overwrite 关闭。",
    )
    parser.add_argument(
        "--use-existing-local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否跳过转换，直接上传已经生成好的本地数据集目录。",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多转换多少条 episode；None 表示转换全部 episode。",
    )
    parser.add_argument(
        "--success-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否只转换成功轨迹，即 info.json 中 success=true 的 episode。",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="指定要写入数据集的相机，用逗号分隔；None 表示使用采集数据中保存的全部相机。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="覆盖数据集帧率；None 表示读取采集时记录的 fps。",
    )
    parser.add_argument(
        "--max-image-stat-frames",
        type=int,
        default=500,
        help="每个相机最多抽样多少帧计算图像统计；设为 0 则使用占位统计，速度更快。",
    )
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否上传到 Hugging Face；可用 --no-push 只生成本地数据集。",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否创建/上传为私有数据集仓库。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token；None 表示使用 huggingface-cli login 保存的 token。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="上传目标分支或版本；None 表示默认 main。",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload LeRobot collected dataset",
        help="上传到 Hub 时显示的提交信息。",
    )
    parser.add_argument(
        "--http-proxy",
        default="",
        help="HTTP 代理地址；留空表示不修改当前环境变量。",
    )
    parser.add_argument(
        "--https-proxy",
        default="",
        help="HTTPS 代理地址；留空表示不修改当前环境变量。",
    )
    return parser


def apply_runtime_env(args: argparse.Namespace) -> None:
    if args.http_proxy:
        import os

        os.environ["http_proxy"] = args.http_proxy
    if args.https_proxy:
        import os

        os.environ["https_proxy"] = args.https_proxy

# 如果 --use-existing-local，直接使用 local-dir；否则先构建本地数据集。
def resolve_local_dir_from_args(args: argparse.Namespace) -> Path:
    raw_dir = resolve_raw_dir(args.raw_dir) # 解析 raw_dir 以确定默认 local_dir 的位置，但如果 --local-dir 已指定，则直接使用它。
    return Path(args.local_dir) if args.local_dir else local_dataset_dir(
        raw_dir, args.output_dir, args.repo_id
    )


def run_from_args(args: argparse.Namespace) -> None:
    apply_runtime_env(args)

    # 如果 --use-existing-local，直接使用 local-dir；否则先构建本地数据集。
    if args.use_existing_local:
        local_dir = resolve_local_dir_from_args(args)
        if not local_dir.exists():
            raise FileNotFoundError(
                f"{local_dir} does not exist. Disable --use-existing-local to build it first."
            )
        logging.info("Using existing local dataset: %s", local_dir)
    else:
        local_dir = build_local_dataset(args)

    # 如果 --push，上传到 Hugging Face；否则提示本地数据集已准备好。
    if args.push:
        push_dataset(local_dir, args)
    else:
        logging.info("Local dataset is ready. Add --push with --repo-id to upload it.")


def push_data_folder_to_hf(
    raw_dir: str,
    repo_id: str,
    output_dir: str = "outputs/hf_datasets",
    local_dir: str | None = None,
    overwrite: bool = True,
    use_existing_local: bool = False,
    max_episodes: int | None = None,
    success_only: bool = False,
    cameras: str | None = None,
    fps: int | None = None,
    max_image_stat_frames: int = 500,
    push: bool = True,
    private: bool = False,
    token: str | None = None,
    revision: str | None = None,
    commit_message: str = "Upload LeRobot collected dataset",
    http_proxy: str = "",
    https_proxy: str = "",
) -> None:
    """Use explicit Python variables instead of command line arguments."""
    init_logging()
    args = argparse.Namespace(
        raw_dir=raw_dir,
        repo_id=repo_id,
        output_dir=output_dir,
        local_dir=local_dir,
        overwrite=overwrite,
        use_existing_local=use_existing_local,
        max_episodes=max_episodes,
        success_only=success_only,
        cameras=cameras,
        fps=fps,
        max_image_stat_frames=max_image_stat_frames,
        push=push,
        private=private,
        token=token,
        revision=revision,
        commit_message=commit_message,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
    )
    run_from_args(args)


def main() -> None:
    init_logging()
    parser = build_arg_parser()
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    # collect_data.py 采集数据目录；也可以填 outputs/4_data_collect 自动选择最新 collect_*。
    RAW_DIR = "outputs/4_data_collect/collect_SewNeedle-3Arms-v0_3Arms_2026-06-02_15-01-25"

    # Hugging Face 数据集仓库，格式必须是 用户名/数据集名。
    HF_REPO_ID = "Dc-dc/collect_sim_sew_needle_3arms"

    # 本地 HF 数据集生成目录。
    OUTPUT_DIR = "outputs/hf_datasets"

    # 如果只想上传已生成好的本地HF目录，设为 True。
    USE_EXISTING_LOCAL = False

    # 本地 HF数据目录 已存在时是否覆盖重建。
    OVERWRITE = True

    # 是否上传到 Hugging Face；False 表示只生成本地数据集。
    PUSH_TO_HUB = True

    # 是否只转换成功轨迹。
    SUCCESS_ONLY = False

    # 最多转换多少条 episode；None 表示全部转换。
    MAX_EPISODES = None



    push_data_folder_to_hf(
        raw_dir=RAW_DIR,
        repo_id=HF_REPO_ID,
        output_dir=OUTPUT_DIR,
        overwrite=OVERWRITE,
        use_existing_local=USE_EXISTING_LOCAL,
        max_episodes=MAX_EPISODES,
        success_only=SUCCESS_ONLY,
        push=PUSH_TO_HUB,
    )
