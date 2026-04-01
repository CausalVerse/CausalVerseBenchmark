import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.optim as optim
import torchvision
from diffusers import AutoencoderKL, AutoencoderKLCogVideoX
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_modules.data_factory import DataFactory
from lib.ivae import LatentFrameIVAE, build_auxiliary_variables

LATENT_SCALING_FACTOR = 0.18215
LOCAL_SD_VAE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models"))
_VIDEO_WRITE_WARNED = False

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def setup_distributed() -> tuple[torch.device, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0

def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()

def write_video(save_path, frames, fps=8):
    global _VIDEO_WRITE_WARNED

    if hasattr(torchvision.io, "write_video"):
        torchvision.io.write_video(
            save_path,
            frames,
            fps=fps,
            video_codec="libx264",
            options={"crf": "18"},
        )
        return True

    if imageio is not None:
        frame_array = frames.detach().cpu().numpy()
        imageio.mimwrite(
            save_path,
            frame_array,
            fps=fps,
            codec="libx264",
            quality=8
        )
        return True

    if not _VIDEO_WRITE_WARNED:
        print(
            "[warn] video export unavailable: torchvision.io.write_video is missing "
            "and imageio is not installed. Install imageio imageio-ffmpeg to enable mp4 export."
        )
        _VIDEO_WRITE_WARNED = True
    return False

def get_current_beta(args, epoch: int) -> float:
    if args.beta_warmup_epochs <= 0:
        return args.beta
    progress = min(epoch / args.beta_warmup_epochs, 1.0)
    return args.beta_start + progress * (args.beta - args.beta_start)

def train(args):
    set_seed(args.seed)
    device, local_rank = setup_distributed()
    tensorboard_dir = os.environ.get("TENSORBOARD_DIR", args.log_dir)
    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    video_dir = os.path.join(args.log_dir, "videos")

    writer = SummaryWriter(log_dir=tensorboard_dir) if is_main_process() else None

    factory = DataFactory()
    dataset = factory.create_dataset(
        name=args.dataset_name,
        dataset_type="video",
        dataset_path=args.dataset_path,
        split="",
        image_size=(args.height, args.width),
        num_frames=args.num_frames,
        clips_per_video=args.clips_per_video,
        sampling="uniform",
        only_latents=True,
        latent_view=args.latent_view,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
    )

    aux_dim = 1 + args.aux_noise_dim
    model = LatentFrameIVAE(
        channel=4 if not args.dataset_name.endswith("_vid") else 16,
        time=args.num_frames,
        aux_dim=aux_dim,
        z_dim=args.z_dim,
    ).to(device)
    if args.fp16:
        model = model.half()
    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if args.dataset_name.endswith("_vid"):
        decoder = AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX-2b",
            subfolder="vae",
            torch_dtype=torch.float32 if device.type == "cuda" else torch.float16,
        ).to(device).eval()
    else:
        decoder = AutoencoderKL.from_pretrained(
            LOCAL_SD_VAE_PATH,
            torch_dtype=torch.float32 if device.type == "cuda" else torch.float16,
        ).to(device).eval()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        current_beta = get_current_beta(args, epoch)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())

        for batch in pbar:
            x = batch["latent"].to(device)
            if not args.dataset_name.endswith("_vid"):
                x = x.permute(0, 2, 1, 3, 4).contiguous()

            x = x.float()
            if args.fp16:
                x = x.half()

            B, C, T, H, W = x.shape
            u = build_auxiliary_variables(
                batch_size=B,
                time_steps=T,
                noise_dim=args.aux_noise_dim,
                device=x.device,
                dtype=x.dtype,
                noise_scale=args.aux_noise_scale,
            )

            optimizer.zero_grad()
            loss, recon_loss, kl_loss, z, recon_x = (
                model.module.elbo(x, u, beta=current_beta)
                if isinstance(model, DDP)
                else model.elbo(x, u, beta=current_beta)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if writer is not None:
                writer.add_scalar("Loss/total", loss.item(), global_step)
                writer.add_scalar("Loss/reconstruction", recon_loss.item(), global_step)
                writer.add_scalar("Loss/KL", kl_loss.item(), global_step)
                writer.add_scalar("Loss/beta", current_beta, global_step)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            if is_main_process():
                pbar.set_postfix(
                    step=global_step,
                    beta=f"{current_beta:.5f}",
                    loss=f"{loss.item():.4f}",
                    recon=f"{recon_loss.item():.4f}",
                    kl=f"{kl_loss.item():.4f}",
                )

            if is_main_process() and global_step % args.save_interval == 0:
                model.eval()
                with torch.no_grad():
                    step_str = f"{global_step:08d}"
                    save_path = os.path.join(video_dir, f"step_{step_str}.mp4")
                    gt_save_path = os.path.join(video_dir, f"step_{step_str}_gt.mp4")
                    sample_recon = recon_x[0]

                    if not args.dataset_name.endswith("_vid"):
                        decoded_frames = []
                        sample_recon = sample_recon.permute(1, 0, 2, 3) / LATENT_SCALING_FACTOR
                        for t in range(sample_recon.shape[0]):
                            frame_latent = sample_recon[t].unsqueeze(0)
                            decoded_frame = decoder.decode(frame_latent).sample[0].float().cpu()
                            decoded_frame = (decoded_frame * 0.5 + 0.5).clamp(0, 1)
                            decoded_frames.append(decoded_frame)
                        decoded = torch.stack(decoded_frames, dim=1)
                        frames = (decoded * 255).byte().permute(1, 2, 3, 0)
                        write_video(save_path, frames, fps=8)

                        orig_latent = x[0].permute(1, 0, 2, 3) / LATENT_SCALING_FACTOR
                        gt_frames_list = []
                        for t in range(orig_latent.shape[0]):
                            frame_latent = orig_latent[t].unsqueeze(0)
                            gt_frame = decoder.decode(frame_latent).sample[0].float().cpu()
                            gt_frame = (gt_frame * 0.5 + 0.5).clamp(0, 1)
                            gt_frames_list.append(gt_frame)
                        gt_decoded = torch.stack(gt_frames_list, dim=1)
                        gt_frames = (gt_decoded * 255).byte().permute(1, 2, 3, 0)
                        write_video(gt_save_path, gt_frames, fps=8)
                    else:
                        sample_recon = sample_recon.unsqueeze(0)
                        decoded = decoder.decode(sample_recon).sample[0].float().cpu()
                        decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
                        frames = (decoded * 255).byte().permute(1, 2, 3, 0)
                        write_video(save_path, frames, fps=8)

                        orig_latent = x[0].permute(1, 0, 2, 3).unsqueeze(0)
                        gt_decoded = decoder.decode(orig_latent).sample[0].float().cpu()
                        gt_decoded = (gt_decoded * 0.5 + 0.5).clamp(0, 1)
                        gt_frames = (gt_decoded * 255).byte().permute(1, 2, 3, 0)
                        write_video(gt_save_path, gt_frames, fps=8)

                model.train()

            global_step += 1

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        if writer is not None:
            writer.add_scalar("Epoch/loss", avg_loss, epoch)
            writer.add_scalar("Epoch/reconstruction", avg_recon, epoch)
            writer.add_scalar("Epoch/KL", avg_kl, epoch)
            writer.add_scalar("Epoch/beta", current_beta, epoch)

        if is_main_process():
            print(f"[Epoch {epoch}] beta: {current_beta:.5f}, loss: {avg_loss:.4f}, recon: {avg_recon:.4f}, kl: {avg_kl:.4f}")
            ckpt_path = os.path.join(ckpt_dir, f"ivae_epoch{epoch}.pth")
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(state_dict, ckpt_path)

    if writer is not None:
        writer.close()
    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="fall")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="training-runs/ivae/debug")
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--clips_per_video", type=int, default=1)
    parser.add_argument("--latent_view", type=str, default="front")
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--beta_start", type=float, default=0.0)
    parser.add_argument("--beta_warmup_epochs", type=int, default=50)
    parser.add_argument("--aux_noise_dim", type=int, default=0, help="0 means time-only auxiliary variable.")
    parser.add_argument("--aux_noise_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
