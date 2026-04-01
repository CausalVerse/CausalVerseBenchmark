import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoencoderKLCogVideoX
from diffusers import AutoencoderKL
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from lib.tdrl import TDRLLatentVAE
from data_modules.data_factory import DataFactory
from torch.utils.tensorboard import SummaryWriter

LATENT_SCALING_FACTOR = 0.18215
LOCAL_SD_VAE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models"))
_VIDEO_WRITE_WARNED = False

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

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

def init_weights_random(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            

def analyze_and_optimize_network(model, dataset):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("[Analyze] Collecting activations on small data...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3: break

            x = batch["latent"].to(device)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.float()
            model(x)

    print("[Analyze] Optimizing network based on activation stats...")
    model.optimize_activations()

    print("[Analyze] Network optimization complete.")
    return model

def write_video(save_path, frames, fps=8):
    global _VIDEO_WRITE_WARNED

    if hasattr(torchvision.io, "write_video"):
        torchvision.io.write_video(
            save_path,
            frames,
            fps=fps,
            video_codec="libx264",
            options={"crf": "18"}
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
def train(args):
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
    model = TDRLLatentVAE(
        channel=4,
        time=args.num_frames,
        latent_dim=args.latent_dim,
        z_dim=args.z_dim,
        lag=args.lag,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        gamma=args.gamma,
        latent_recon_weight=args.latent_recon_weight,
        temporal_recon_weight=args.temporal_recon_weight,
    ).to(device)
    if args.fp16:
        model = model.half()
    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    
    if args.dataset_name.endswith("_vid"):
        cog_decoder = AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX-2b",
            subfolder="vae",
            torch_dtype=torch.float32 if device.type == "cuda" else torch.float16
        ).to(device).eval()
    else:
       img_decoder = AutoencoderKL.from_pretrained(
            LOCAL_SD_VAE_PATH,
            torch_dtype=torch.float32 if device.type == "cuda" else torch.float16
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
        total_loss, total_recon, total_latent_recon, total_temporal_recon, total_kld_normal, total_kld_future = 0, 0, 0, 0, 0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())
        for batch in pbar:
            x = batch["latent"].to(device)

            if not args.dataset_name.endswith("_vid"):
                x = x.permute(0, 2, 1, 3, 4).contiguous()

            x = x.float()
            if args.fp16:
                x = x.half()
            
            optimizer.zero_grad()
            outputs, losses = model(x)
            recon_x = outputs["x_recon"]
            loss = losses["total"]
            recon_loss = losses["recon"]
            latent_recon_loss = losses["latent_recon"]
            temporal_recon_loss = losses["temporal_recon"]
            kld_normal = losses["independence"]
            kld_future = losses["temporal"]

            loss.backward()
            optimizer.step()
            
            if writer is not None:
                writer.add_scalar("Loss/total", loss.item(), global_step)
                writer.add_scalar("Loss/recon", recon_loss.item(), global_step)
                writer.add_scalar("Loss/latent_recon", latent_recon_loss.item(), global_step)
                writer.add_scalar("Loss/temporal_recon", temporal_recon_loss.item(), global_step)
                writer.add_scalar("Loss/kld_normal", kld_normal.item(), global_step)
                writer.add_scalar("Loss/kld_future", kld_future.item(), global_step)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_latent_recon += latent_recon_loss.item()
            total_temporal_recon += temporal_recon_loss.item()
            total_kld_normal += kld_normal.item()
            total_kld_future += kld_future.item()

            if is_main_process():
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{loss.item():.4f}",
                    "recon_loss": f"{recon_loss.item():.4f}",
                    "latent_recon": f"{latent_recon_loss.item():.4f}",
                    "temporal_recon": f"{temporal_recon_loss.item():.4f}",
                    "kld_normal": f"{kld_normal.item():.4f}",
                    "kld_future": f"{kld_future.item():.4f}",
                })
            if is_main_process() and global_step % args.save_interval == 0:
                model.eval()
                with torch.no_grad():
                    step_str = f"{global_step:08d}"

                    save_path = os.path.join(video_dir, f"step_{step_str}.mp4")
                    gt_save_path = os.path.join(video_dir, f"step_{step_str}_gt.mp4")
                    sample_recon = recon_x[0]  
                       
                    if not args.dataset_name.endswith("_vid"):
                        img_decoder_dtype = next(img_decoder.parameters()).dtype
                        decoded_frames = []
                        sample_recon = sample_recon.permute(1, 0, 2, 3)
                        sample_recon = sample_recon / LATENT_SCALING_FACTOR  
                        for t in range(sample_recon.shape[0]):
                            frame_latent = sample_recon[t].unsqueeze(0).to(device=device, dtype=img_decoder_dtype)
                            decoded_frame = img_decoder.decode(frame_latent).sample
                            decoded_frame = decoded_frame[0].float().cpu()
                            decoded_frame = (decoded_frame * 0.5 + 0.5).clamp(0, 1)
                            decoded_frames.append(decoded_frame)
                        
                        decoded = torch.stack(decoded_frames, dim=1)
                        frames = (decoded * 255).byte().permute(1, 2, 3, 0)
                        
                        write_video(save_path, frames, fps=8)
                        print(f"[Step {global_step}] Saved reconstruction video (image-based): {save_path}")
                        
                        orig_latent = x[0]                        
                        orig_latent = orig_latent.permute(1, 0, 2, 3)
                        orig_latent = orig_latent / LATENT_SCALING_FACTOR  
                        gt_decoded_frames = []
                        for t in range(orig_latent.shape[0]):
                            frame_latent = orig_latent[t].unsqueeze(0).to(device=device, dtype=img_decoder_dtype)
                            gt_decoded_frame = img_decoder.decode(frame_latent).sample
                            gt_decoded_frame = gt_decoded_frame[0].float().cpu()
                            gt_decoded_frame = (gt_decoded_frame * 0.5 + 0.5).clamp(0, 1)
                            gt_decoded_frames.append(gt_decoded_frame)
                        
                        gt_decoded = torch.stack(gt_decoded_frames, dim=1)
                        gt_frames = (gt_decoded * 255).byte().permute(1, 2, 3, 0)
                        
                        write_video(gt_save_path, gt_frames, fps=8)
                        print(f"[Step {global_step}] Saved videos (image-based): {save_path} & {gt_save_path}")
                    else:
                        cog_decoder_dtype = next(cog_decoder.parameters()).dtype
                        sample_recon = sample_recon.unsqueeze(0)
                        sample_recon = sample_recon.to(device=device, dtype=cog_decoder_dtype)
                        decoded = cog_decoder.decode(sample_recon).sample
                        decoded = decoded[0].float().cpu()
                        decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
                        frames = (decoded * 255).byte().permute(1, 2, 3, 0)

                        save_path = os.path.join(video_dir, f"step_{step_str}.mp4")
                        write_video(save_path, frames, fps=8)
                        print(f"[Step {global_step}] Saved reconstruction video: {save_path}")
                        
                        
                        orig_latent = x[0]
                        if orig_latent.shape[1] != 16 and orig_latent.shape[0] == 16:
                            orig_latent = orig_latent.permute(1, 0, 2, 3)
                        orig_latent = orig_latent.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=cog_decoder_dtype)
                        
                        gt_decoded = cog_decoder.decode(orig_latent).sample
                        gt_decoded = gt_decoded[0].float().cpu()
                        gt_decoded = (gt_decoded * 0.5 + 0.5).clamp(0, 1)
                        gt_frames = (gt_decoded * 255).byte().permute(1, 2, 3, 0)
                        gt_save_path = os.path.join(video_dir, f"step_{step_str}_gt.mp4")
                        write_video(gt_save_path, gt_frames, fps=8)

                        print(f"[Step {global_step}] Saved videos: {save_path} & {gt_save_path}")
        
                model.train()

            global_step += 1
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_latent_recon = total_latent_recon / len(dataloader)
        avg_temporal_recon = total_temporal_recon / len(dataloader)
        avg_kld_normal = total_kld_normal / len(dataloader)
        avg_kld_future = total_kld_future / len(dataloader)
        
        
       
        if writer is not None:
            writer.add_scalar("Epoch/loss", avg_loss, epoch)
            writer.add_scalar("Epoch/recon", avg_recon, epoch)
            writer.add_scalar("Epoch/latent_recon", avg_latent_recon, epoch)
            writer.add_scalar("Epoch/temporal_recon", avg_temporal_recon, epoch)
            writer.add_scalar("Epoch/kld_normal", avg_kld_normal, epoch)
            writer.add_scalar("Epoch/kld_future", avg_kld_future, epoch)
        if is_main_process():
            print(f"[Epoch {epoch}] loss: {avg_loss:.4f}, recon: {avg_recon:.4f}, latent_recon: {avg_latent_recon:.4f}, temporal_recon: {avg_temporal_recon:.4f}, kld_normal: {avg_kld_normal:.4f}, avg_kld_future: {avg_kld_future:.4f}")
        if is_main_process():
            ckpt_path = os.path.join(ckpt_dir, f"vae_epoch{epoch}.pth")
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(state_dict, ckpt_path)
        
    if writer is not None:
        writer.close()
    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="fall_img", help="Name of the dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per clip")
    parser.add_argument("--model", type=str, default="", help="Choose model: ivae | add | caring | idol, if not set, would be referred as standard vae")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=2048, help="Dimensionality of frame-wise backbone latent")
    parser.add_argument("--z_dim", type=int, default=8, help="Dimensionality of causal latent code")
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.00025)
    parser.add_argument("--gamma", type=float, default=0.0075)
    parser.add_argument("--latent_recon_weight", type=float, default=0.5)
    parser.add_argument("--temporal_recon_weight", type=float, default=0.5)
    parser.add_argument("--log_dir", type=str, default="training-runs/baseline_vae/collision_simple")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval (in steps) to save reconstructed videos")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument("--clips_per_video", type=int, default=1, help="Number of clips per video")
    parser.add_argument("--latent_view", type=str, default="front")
    args = parser.parse_args()

    train(args)
