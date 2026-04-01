import argparse
import time

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from base_vae import LatentFrameVAE
from data_modules.data_factory import DataFactory
from lib.caring import CaRiNGModel
from lib.idol import IDOLLatentVAE
from lib.ivae import LatentFrameIVAE, build_auxiliary_variables
from lib.tcl import TCLLatentFrameVAE
from lib.tdrl import TDRLLatentVAE
from metrics.mcc import compute_mcc_topk
from metrics.r2 import compute_r2


def compute_mcc1(pred, gt, metric="R2"):
    pred = np.array(pred)
    gt = np.array(gt)

    if metric == "R2":
        return r2_score(gt, pred)
    if metric == "Pearson":
        import scipy.stats

        r = []
        for i in range(gt.shape[1]):
            corr, _ = scipy.stats.pearsonr(gt[:, i], pred[:, i])
            r.append(corr)
        return np.mean(r)
    raise ValueError("Unknown metric: choose from ['R2', 'Pearson'].")


def print_debug_stats(name, arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"[debug_mcc] {name} is empty with shape {arr.shape}")
        return
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
    var_per_dim = np.var(flat, axis=1)
    mean_per_dim = np.mean(flat, axis=1)
    nan_per_dim = np.isnan(flat).any(axis=1)
    inf_per_dim = np.isinf(flat).any(axis=1)
    zero_var_mask = var_per_dim < 1e-12

    print(f"[debug_mcc] {name} shape: {arr.shape}")
    print(f"[debug_mcc] {name} mean(first 8): {mean_per_dim[:8]}")
    print(f"[debug_mcc] {name} var(first 8): {var_per_dim[:8]}")
    print(f"[debug_mcc] {name} nan_dims: {np.where(nan_per_dim)[0].tolist()}")
    print(f"[debug_mcc] {name} inf_dims: {np.where(inf_per_dim)[0].tolist()}")
    print(f"[debug_mcc] {name} near_zero_var_dims: {np.where(zero_var_mask)[0].tolist()}")


def compute_fast_mcc_topk(latents, gts, k=4):
    eps = 1e-12

    lat_mean = latents.mean(axis=1, keepdims=True)
    gt_mean = gts.mean(axis=1, keepdims=True)
    lat_std = latents.std(axis=1, keepdims=True)
    gt_std = gts.std(axis=1, keepdims=True)

    lat_std[lat_std < eps] = eps
    gt_std[gt_std < eps] = eps

    lat_norm = (latents - lat_mean) / lat_std
    gt_norm = (gts - gt_mean) / gt_std

    corr = (gt_norm @ lat_norm.T) / latents.shape[1]
    abs_corr = np.abs(corr)
    best_per_gt = abs_corr.max(axis=1)
    k = min(k, best_per_gt.shape[0])
    topk = np.sort(best_per_gt)[::-1][:k]
    return float(np.mean(topk)), corr, best_per_gt


def build_model(args, device):
    if args.model == "ivae":
        return LatentFrameIVAE(
            channel=4,
            time=args.num_frames,
            aux_dim=1 + args.aux_noise_dim,
            z_dim=args.z_dim,
        ).to(device)
    if args.model == "tcl":
        return TCLLatentFrameVAE(
            channel=4,
            time=args.num_frames,
            z_dim=args.z_dim,
            segment_size=args.segment_size,
        ).to(device)
    if args.model == "caring":
        return CaRiNGModel(
            channel=4,
            time=args.num_frames,
            latent_dim=args.latent_dim,
            z_dim=args.z_dim,
            lag=args.lag,
            context_frames=args.context_frames,
            hidden_dim=args.hidden_dim,
            beta=args.beta,
            gamma=args.gamma,
        ).to(device)
    if args.model == "tdrl":
        return TDRLLatentVAE(
            channel=4,
            time=args.num_frames,
            latent_dim=args.latent_dim,
            z_dim=args.z_dim,
            lag=args.lag,
            hidden_dim=args.hidden_dim,
            beta=args.beta,
            gamma=args.gamma,
        ).to(device)
    if args.model == "idol":
        return IDOLLatentVAE(
            channel=4,
            time=args.num_frames,
            latent_dim=args.latent_dim,
            z_dim=args.z_dim,
            lag=args.lag,
            hidden_dim=args.hidden_dim,
            beta=args.beta,
            gamma=args.gamma,
            theta=args.theta,
        ).to(device)
    return LatentFrameVAE(channel=4, z_dim=args.z_dim).to(device)


def normalize_repr_mode(mode: str) -> str:
    aliases = {
        "full": "full",
        "frame_latents": "full",
        "agg": "agg",
        "causal_z": "agg",
        "mean": "agg",
    }
    if mode not in aliases:
        raise ValueError(f"Unknown repr_mode '{mode}'. Expected one of: {', '.join(sorted(aliases))}")
    return aliases[mode]


def aggregate_spatial_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.shape[-1] != 8 * 16 * 16:
        raise ValueError(f"Aggregation requires 2048-d latent, got shape {tuple(latent.shape)}")
    batch_size, time_steps, _ = latent.shape
    return latent.view(batch_size, time_steps, 8, 16, 16).mean(dim=[3, 4])


def select_gt_vectors(gt_vectors, repr_mode: str, dataset_name: str):
    """
    Select which GT dimensions participate in evaluation.

    This helper is intentionally the place to customize the GT slice used by
    both MCC and R2. In other words, if you want evaluation to focus on a
    subset of factors, change the returned rows here.

    `gt_vectors` is expected to be shaped as [D, N], where D is the GT
    dimensionality and N is the number of sampled frames.

    Example for the fixed robotics datasets:
    one possible 57D form commonly seen by the current evaluation code comes
    from the latent-only path, where
    `FixedRoboticsDataset.merge_content_and_style_as_sequence(...)` produces:

    - [0:4):   4 object-category slots from `encode_global_to_vector_simple`
    - [4:11):  `joint_pos` (7 dims)
    - [11:18): `joint_vel` (7 dims)
    - [18:21): `eef_pos` (3 dims)
    - [21:25): `eef_quat` (4 dims)
    - [25:27): `gripper_qpos` (2 dims)
    - [27:29): `gripper_qvel` (2 dims)
    - [29:41): `pos_objects` for up to 4 objects, flattened as 4 * 3 dims
    - [41:57): `rot_objects` for up to 4 objects, flattened as 4 * 4 dims

    This is only one possible GT layout used by the present dataset code. It
    omits other static information one might also want to evaluate, such as
    scene-level metadata. Therefore, a slice such as
    `gt_vectors[12:20, :]` is only a custom experiment-specific choice;
    it may even not be a clean semantic block. For example, in the
    robotics-kitchen layout above, it spans part of `joint_vel` plus
    part of `eef_pos`.

    Please customize this function according to the specific GT dimensions you
    want your evaluation to target.
    """
    if repr_mode != "agg":
        return gt_vectors

    if dataset_name.startswith("fixed_robotics"):
        return gt_vectors


    return gt_vectors[: min(8, gt_vectors.shape[0]), :]


def extract_pred_representation(args, model, x, u=None):
    repr_mode = normalize_repr_mode(args.repr_mode)

    if args.model == "ivae":
        _, _, _, zs_orig = model(x, u)
        representations = {
            "full": zs_orig,
            "agg": zs_orig,
        }
    elif args.model == "tcl":
        _, _, _, zs_orig, _, _ = model(x)
        representations = {
            "full": zs_orig,
            "agg": aggregate_spatial_latent(zs_orig),
        }
    elif args.model == "caring":
        caring_outputs, _ = model(x, random_sampling=False)
        representations = {
            "full": caring_outputs["frame_latents"],
            "agg": caring_outputs["causal_z"],
        }
    elif args.model == "tdrl":
        tdrl_outputs, _ = model(x, random_sampling=False)
        representations = {
            "full": tdrl_outputs["frame_latents"],
            "agg": tdrl_outputs["causal_z"],
        }
    elif args.model == "idol":
        batch_size, _, length, height, width = x.shape
        frame_first = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * length, model.channel, height, width)
        z_full, _, _ = model.latent_vae.encoder(frame_first)
        z_full = z_full.view(batch_size, length, model.latent_dim)
        representations = {
            "full": z_full,
            "agg": aggregate_spatial_latent(z_full),
        }
    else:
        _, _, _, zs_orig = model(x)
        representations = {
            "full": zs_orig,
            "agg": aggregate_spatial_latent(zs_orig),
        }

    if args.model == "ivae":
        representations["agg"] = aggregate_spatial_latent(zs_orig)

    if repr_mode not in representations:
        available = ", ".join(sorted(representations.keys()))
        raise ValueError(
            f"Model '{args.model}' does not support repr_mode='{repr_mode}'. "
            f"Available representations: {available}"
        )
    return representations[repr_mode]


def evaluate(args):
    t0 = time.time()
    args.repr_mode = normalize_repr_mode(args.repr_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    if args.fp16:
        model = model.half()
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    pred_vectors = []
    gt_vectors = []
    used_videos = 0
    max_videos = args.max_samples if args.max_samples > 0 else float("inf")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x = batch["latent"].to(device).float()
            B, T, C, H, W = x.shape

            if used_videos + B > max_videos:
                B = max_videos - used_videos
                if B <= 0:
                    break
                x = x[:B]
                batch["vector"] = batch["vector"][:B]

            if args.eval_frames is None or args.eval_frames >= T:
                selected_indices = list(range(T))
            else:
                selected_indices = np.linspace(0, T - 1, args.eval_frames).astype(int).tolist()

            x = x[:, selected_indices]
            vector = batch["vector"]
            if isinstance(vector, list):
                vector = torch.stack(vector, dim=0)
            selected_vectors = vector[:, selected_indices]

            B, T_sel, C, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous()

            u = None
            if args.model == "ivae":
                u = build_auxiliary_variables(
                    batch_size=B,
                    time_steps=T_sel,
                    noise_dim=args.aux_noise_dim,
                    device=x.device,
                    dtype=x.dtype,
                    noise_scale=args.aux_noise_scale,
                )
            pred_repr = extract_pred_representation(args, model, x, u=u)
            pred = pred_repr.cpu().numpy()

            for i in range(B):
                pred_vectors.append(pred[i])
                gt_vectors.append(selected_vectors[i].numpy())
            used_videos += B

    print(f"[timing] feature extraction finished in {time.time() - t0:.2f}s", flush=True)

    pred_vectors = np.concatenate(pred_vectors, axis=0)
    gt_vectors = np.concatenate(gt_vectors, axis=0)

    if pred_vectors.ndim > 2:
        pred_vectors = pred_vectors.reshape(-1, pred_vectors.shape[-1])

    pred_vectors_full = pred_vectors.T
    gt_vectors_full = gt_vectors.T
    gt_vectors_selected = select_gt_vectors(gt_vectors_full, args.repr_mode, args.dataset_name)

    hz = pred_vectors_full.T
    z_selected = gt_vectors_selected.T

    print(pred_vectors.shape, flush=True)
    print(gt_vectors.shape, flush=True)
    print(f"[eval] selected_gt_shape={z_selected.shape}", flush=True)

    t_mcc = time.time()
    pred_for_mcc = pred_vectors_full
    mcc, corr_matrix, best_per_gt = compute_fast_mcc_topk(pred_for_mcc, gt_vectors_selected, k=args.topk)
    if args.debug_mcc:
        print_debug_stats(f"pred_vectors_{args.repr_mode}", pred_for_mcc)
        print_debug_stats("gt_vectors_selected", gt_vectors_selected)
        print(f"[debug_mcc] corr_matrix shape: {corr_matrix.shape}")
        print(f"[debug_mcc] best_per_gt: {best_per_gt}")
        print(f"[debug_mcc] topk_used: {np.sort(best_per_gt)[::-1][:min(args.topk, best_per_gt.shape[0])]}")
    print(f"Mean Pearson Correlation (MCC) on {args.repr_mode}: {mcc:.4f}", flush=True)
    print(f"[timing] mcc finished in {time.time() - t_mcc:.2f}s", flush=True)

    if args.debug_mcc:
        print("[debug_mcc] skipping R2 because --debug_mcc is enabled", flush=True)
        return

    t_r2 = time.time()
    print(
        f"[timing] starting r2 on repr_mode={args.repr_mode} with hz shape={hz.shape} "
        f"and gt shape={z_selected.shape}",
        flush=True,
    )
    r2_score_val = compute_r2(z_selected, hz, select_mode=args.r2_select_mode)
    print(f"NonLinear R² Score: {r2_score_val:.4f}", flush=True)
    print(f"[timing] r2 finished in {time.time() - t_r2:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="fall_img")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--eval_frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--clips_per_video", type=int, default=1)
    parser.add_argument("--latent_view", type=str, default="front")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--z_dim", type=int, default=10)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--debug_mcc", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default="ivae",
        choices=["ivae", "tcl", "caring", "tdrl", "idol"],
    )
    parser.add_argument("--repr_mode", type=str, default="agg")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--r2_select_mode", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--aux_noise_dim", type=int, default=0)
    parser.add_argument("--aux_noise_scale", type=float, default=1.0)
    parser.add_argument("--segment_size", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=2048)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--context_frames", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.00025)
    parser.add_argument("--gamma", type=float, default=0.0075)
    parser.add_argument("--theta", type=float, default=0.02)
    args = parser.parse_args()
    evaluate(args)
