#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import decord
import numpy as np
import torch
from decord import VideoReader
from diffusers import AutoencoderKL
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


LATENT_SCALING_FACTOR = 0.18215


def stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def infer_dataset_type(dataset_root: Path) -> str:
    video_dir = dataset_root / "video"
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing video directory: {video_dir}")

    names = [p.name for p in video_dir.iterdir() if p.is_file()]
    if any(name.endswith("_frontview.mp4") for name in names):
        return "fixed_robotics"
    if any(name.endswith("_front_color.mp4") for name in names):
        return "physics"
    raise ValueError(
        f"Unable to infer dataset type from {video_dir}. "
        "Pass --dataset-type explicitly."
    )


def resolve_view(dataset_type: str, view: str) -> Tuple[str, str]:
    if dataset_type == "fixed_robotics":
        view_alias = {
            "front": "frontview",
            "frontview": "frontview",
            "side": "sideview",
            "sideview": "sideview",
            "bird": "birdview",
            "birdview": "birdview",
            "agent": "agentview",
            "agentview": "agentview",
            "eye": "robot0_eye_in_hand",
            "robot0_eye_in_hand": "robot0_eye_in_hand",
        }
        video_view = view_alias.get(view, view)
        latent_subdir = view.replace("view", "")
        if video_view == "robot0_eye_in_hand":
            latent_subdir = "robot0_eye_in_hand"
        elif latent_subdir not in {"front", "side", "bird", "agent"}:
            latent_subdir = video_view
        return video_view, latent_subdir

    view_alias = {
        "front": "front",
        "left": "left",
        "right": "right",
        "bird": "bird",
    }
    video_view = view_alias.get(view, view)
    return video_view, video_view


def build_video_path(dataset_root: Path, dataset_type: str, uuid: str, video_view: str) -> Path:
    video_dir = dataset_root / "video"
    if dataset_type == "fixed_robotics":
        return video_dir / f"{uuid}_{video_view}.mp4"
    return video_dir / f"{uuid}_{video_view}_color.mp4"


def choose_indices(
    total_frames: int,
    num_frames: int,
    interval: int,
    max_start_frame: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_frames_needed = (num_frames - 1) * interval + 1
    start_upper = max(0, min(max_start_frame, total_frames - 1))
    start = int(rng.integers(0, start_upper + 1)) if start_upper > 0 else 0
    end = min(start + total_frames_needed, total_frames)

    if end - start < total_frames_needed:
        indices = np.linspace(start, max(start, end - 1), num_frames).astype(np.int64)
    else:
        indices = np.array([start + i * interval for i in range(num_frames)], dtype=np.int64)
    return indices


def resize_video_tensor(arr: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = image_size
    resized_frames: List[torch.Tensor] = []
    for t in range(arr.shape[0]):
        frame = arr[t]
        _, h, w = frame.shape
        aspect_ratio_in = w / h
        aspect_ratio_out = target_w / target_h

        if aspect_ratio_in > aspect_ratio_out:
            new_h = target_h
            new_w = int(w * target_h / h)
        else:
            new_w = target_w
            new_h = int(h * target_w / w)

        resized = TF.resize(
            frame,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        cropped = TF.center_crop(resized, output_size=[target_h, target_w])
        resized_frames.append(cropped)
    return torch.stack(resized_frames, dim=0)


def load_frames(
    video_path: Path,
    num_frames: int,
    interval: int,
    max_start_frame: int,
    image_size: Tuple[int, int],
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, np.ndarray]:
    decord.bridge.set_bridge("torch")
    vr = VideoReader(str(video_path), height=-1, width=-1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"Empty video: {video_path}")

    indices = choose_indices(
        total_frames=total_frames,
        num_frames=num_frames,
        interval=interval,
        max_start_frame=max_start_frame,
        rng=rng,
    )
    frames = vr.get_batch(indices)
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames)
    frames = frames.permute(0, 3, 1, 2).contiguous()
    frames = resize_video_tensor(frames, image_size=image_size)
    return frames, indices


def encode_latents(
    vae: AutoencoderKL,
    frames: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    frames = frames.float() / 255.0
    frames = frames * 2.0 - 1.0
    latents: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            batch = frames[start:start + batch_size].to(device=device, dtype=dtype)
            latent = vae.encode(batch).latent_dist.sample()
            latent = latent * LATENT_SCALING_FACTOR
            latents.append(latent.cpu())
    return torch.cat(latents, dim=0)


def collect_uuids(meta_dir: Path) -> Iterable[str]:
    for json_path in sorted(meta_dir.glob("*.json")):
        stem = json_path.stem
        uuid = stem[:-5] if stem.endswith("_meta") else stem
        npz_path = meta_dir / f"{uuid}.npz"
        if npz_path.exists():
            yield uuid


def default_latent_root(dataset_root: Path, dataset_type: str, latent_subdir: str) -> Path:
    return dataset_root / "latents" / latent_subdir


def save_latent_file(
    output_path: Path,
    latents: torch.Tensor,
    frame_numbers: Sequence[int],
    source_video: Path,
    dataset_type: str,
    view: str,
    clip_idx: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latents": latents.contiguous(),
        "frame_numbers": [int(x) for x in frame_numbers],
        "source_video": str(source_video),
        "dataset_type": dataset_type,
        "view": view,
        "clip_idx": int(clip_idx),
        "latent_scaling_factor": LATENT_SCALING_FACTOR,
    }
    torch.save(payload, output_path)


def preprocess_dataset(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    meta_dir = dataset_root / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Missing meta directory: {meta_dir}")

    dataset_type = args.dataset_type
    if dataset_type == "auto":
        dataset_type = infer_dataset_type(dataset_root)

    video_view, latent_subdir = resolve_view(dataset_type, args.view)
    latent_root = (
        Path(args.latent_root).resolve()
        if args.latent_root
        else default_latent_root(dataset_root, dataset_type, latent_subdir)
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" and not args.force_fp32 else torch.float32

    vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=dtype).to(device).eval()

    image_size = (args.image_size, args.image_size)
    uuids = list(collect_uuids(meta_dir))
    if args.num_shards > 1:
        uuids = [uuid for idx, uuid in enumerate(uuids) if idx % args.num_shards == args.shard_index]
    if args.max_samples is not None:
        uuids = uuids[:args.max_samples]

    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "dataset_type": dataset_type,
                "view": video_view,
                "latent_root": str(latent_root),
                "num_frames": args.num_frames,
                "interval": args.interval,
                "max_start_frame": args.max_start_frame,
                "image_size": args.image_size,
                "clips_per_video": args.clips_per_video,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "num_samples": len(uuids),
            },
            indent=2,
        )
    )

    for sample_idx, uuid in enumerate(uuids, start=1):
        video_path = build_video_path(dataset_root, dataset_type, uuid, video_view)
        if not video_path.exists():
            print(f"[skip] missing video: {video_path}")
            continue

        for clip_idx in range(args.clips_per_video):
            filename = f"{uuid}.pt" if args.clips_per_video == 1 else f"{uuid}_{clip_idx}.pt"
            output_path = latent_root / filename
            if output_path.exists() and not args.overwrite:
                print(f"[skip] exists: {output_path}")
                continue

            seed = args.seed + stable_int(f"{uuid}:{clip_idx}:{video_view}")
            rng = np.random.default_rng(seed)

            frames, frame_numbers = load_frames(
                video_path=video_path,
                num_frames=args.num_frames,
                interval=args.interval,
                max_start_frame=args.max_start_frame,
                image_size=image_size,
                rng=rng,
            )
            latents = encode_latents(
                vae=vae,
                frames=frames,
                device=device,
                dtype=dtype,
                batch_size=args.encode_batch_size,
            )
            save_latent_file(
                output_path=output_path,
                latents=latents,
                frame_numbers=frame_numbers,
                source_video=video_path,
                dataset_type=dataset_type,
                view=video_view,
                clip_idx=clip_idx,
            )
            print(
                f"[{sample_idx}/{len(uuids)}] saved {output_path} "
                f"shape={tuple(latents.shape)} frames={frame_numbers.tolist()}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute CRL latents with stabilityai/sd-vae-ft-mse."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing meta/ and video/.")
    parser.add_argument(
        "--dataset-type",
        default="auto",
        choices=["auto", "physics", "fixed_robotics", "mobile_robotics"],
        help="Dataset layout preset.",
    )
    parser.add_argument(
        "--view",
        default="front",
        help="Video view to encode. physics: front/left/right/bird; fixed_robotics: front/side/bird/agent/robot0_eye_in_hand.",
    )
    parser.add_argument(
        "--latent-root",
        default=None,
        help="Directory to write .pt latent files. Default follows training-side layout.",
    )
    parser.add_argument(
        "--vae-path",
        default=str((Path(__file__).resolve().parents[2] / "pretrained_models").resolve()),
        help="Local path or HF model id for the image VAE.",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-start-frame", type=int, default=2)
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input resolution for SD-VAE. 512 gives latent shape [T,4,64,64].",
    )
    parser.add_argument("--clips-per-video", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--force-fp32", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.shard_index < 0 or args.num_shards <= 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    return args


if __name__ == "__main__":
    preprocess_dataset(parse_args())
