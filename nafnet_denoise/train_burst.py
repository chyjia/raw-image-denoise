"""Train the alignment-aware 16-frame BurstNAFNet in synthetic or real mode."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .burst_dataset import RealBurstDataset, SyntheticBurstDataset, VideoBurstSyntheticDataset
from .common import (
    RAW_MAX,
    decode_mono10,
    memmap_frames,
    model_output_to_raw,
    parse_exposure_ms,
    parse_geometry,
)
from .infer_burst import denoise_burst, prepare_aligned_burst
from .temporal_fusion import build_burst_nafnet
from .validate import aligned_temporal_trimmed_mean, central_roi


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def masked_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    return masked_mean(
        torch.sqrt((prediction - target).square() + epsilon**2),
        mask,
    )


def masked_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    return masked_charbonnier(pred_x, target_x, mask_x) + masked_charbonnier(
        pred_y,
        target_y,
        mask_y,
    )


def ssim_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, 1, padding)
    mu_y = F.avg_pool2d(target, window_size, 1, padding)
    sigma_x = torch.clamp(
        F.avg_pool2d(prediction.square(), window_size, 1, padding) - mu_x.square(),
        min=0.0,
    )
    sigma_y = torch.clamp(
        F.avg_pool2d(target.square(), window_size, 1, padding) - mu_y.square(),
        min=0.0,
    )
    covariance = (
        F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    )
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / torch.clamp(
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2),
        min=1e-8,
    )


def flow_smoothness(flows: torch.Tensor) -> torch.Tensor:
    dx = flows[..., :, 1:] - flows[..., :, :-1]
    dy = flows[..., 1:, :] - flows[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def burst_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    teacher_valid: torch.Tensor,
    aux: dict[str, torch.Tensor],
    teacher_weight: float,
    grad_weight: float,
    ssim_weight: float,
    mean_weight: float,
    flow_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pixel = masked_charbonnier(prediction, target, mask)
    gradient = masked_gradient_loss(prediction, target, mask)
    structural = masked_mean(1.0 - ssim_map(prediction, target), mask)
    teacher_mask = mask * teacher_valid[:, None, None, None]
    if float(teacher_valid.sum()) > 0:
        teacher_loss = masked_charbonnier(prediction, teacher, teacher_mask)
    else:
        teacher_loss = prediction.new_zeros(())
    flow_loss = flow_smoothness(aux["flows"])
    residual = (prediction - target) * mask
    mean_loss = (
        residual.sum(dim=(1, 2, 3))
        / torch.clamp(mask.sum(dim=(1, 2, 3)), min=1.0)
    ).abs().mean()
    total = (
        pixel
        + grad_weight * gradient
        + ssim_weight * structural
        + mean_weight * mean_loss
        + teacher_weight * teacher_loss
        + flow_weight * flow_loss
    )
    attention = aux["attention"].clamp_min(1e-8)
    entropy = -(attention * attention.log()).sum(dim=1).mean()
    return total, {
        "pixel": float(pixel.detach()),
        "gradient": float(gradient.detach()),
        "ssim_loss": float(structural.detach()),
        "mean": float(mean_loss.detach()),
        "teacher": float(teacher_loss.detach()),
        "flow": float(flow_loss.detach()),
        "attention_entropy": float(entropy.detach()),
    }


def build_dataset(args: argparse.Namespace, validation: bool = False):
    samples = args.validation_samples if validation else args.samples_per_epoch
    seed = 12345 if validation else args.seed
    if validation or args.dataset_mode == "real":
        manifest = args.validation_manifest if validation else args.manifest
        if manifest is None:
            raise ValueError("A real/validation manifest is required.")
        return RealBurstDataset(
            manifest,
            patch_size=args.patch_size,
            input_frames=args.input_frames,
            samples_per_epoch=samples,
            center_mode=args.center_mode,
            seed=seed,
        )
    common_kwargs = dict(
        patch_size=args.patch_size,
        input_frames=args.input_frames,
        samples_per_epoch=samples,
        noise_jitter=args.noise_jitter,
        exposure_ms_range=(args.exposure_ms_min, args.exposure_ms_max),
        black_level_dn=args.black_level_dn,
        black_drift_sigma=args.black_drift_sigma,
        fixed_pattern_sigma=args.fixed_pattern_sigma,
        row_noise_sigma=args.row_noise_sigma,
        column_noise_sigma=args.column_noise_sigma,
        dark_variance_per_s_range=(
            args.dark_variance_per_s_min,
            args.dark_variance_per_s_max,
        ),
        seed=seed,
    )
    if args.dataset_mode == "video":
        return VideoBurstSyntheticDataset(args.manifest, **common_kwargs)
    if args.dataset_mode == "mixed":
        if args.video_manifest is None:
            raise ValueError("--video-manifest is required for mixed mode.")
        pixelshift_samples = max(1, int(round(samples * (1.0 - args.video_fraction))))
        video_samples = max(1, samples - pixelshift_samples)
        pixelshift = SyntheticBurstDataset(
            args.manifest,
            max_shift=args.max_motion_shift,
            max_rotation=args.max_motion_rotation,
            motion_strength=args.motion_strength,
            **{**common_kwargs, "samples_per_epoch": pixelshift_samples},
        )
        video = VideoBurstSyntheticDataset(
            args.video_manifest,
            motion_blur_strength=args.motion_blur_strength,
            **{**common_kwargs, "samples_per_epoch": video_samples, "seed": seed + 17},
        )
        return torch.utils.data.ConcatDataset([pixelshift, video])
    return SyntheticBurstDataset(
        args.manifest,
        max_shift=args.max_motion_shift,
        max_rotation=args.max_motion_rotation,
        motion_strength=args.motion_strength,
        **common_kwargs,
    )


@torch.no_grad()
def validate_patches(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float]:
    dataset = build_dataset(args, validation=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    model.eval()
    maes = []
    ssims = []
    for batch in loader:
        burst = batch["burst"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(burst)
        pred_np = prediction.float().cpu().numpy()
        target_np = target.float().cpu().numpy()
        mask_np = mask.cpu().numpy()
        for index in range(pred_np.shape[0]):
            pred_dn = model_output_to_raw(pred_np[index, 0])
            target_dn = model_output_to_raw(target_np[index, 0])
            valid = mask_np[index, 0] > 0.5
            maes.append(float(np.mean(np.abs(pred_dn[valid] - target_dn[valid]))))
        ssims.append(
            float(masked_mean(ssim_map(prediction.float(), target), mask))
        )
    model.train()
    return float(np.mean(maes)), float(np.mean(ssims))


@torch.no_grad()
def validate_full_frames(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float]:
    """Validate through the same aligned tiled path used for deployment."""
    import json

    if args.validation_manifest is None:
        raise ValueError("A validation manifest is required.")
    entries = json.loads(args.validation_manifest.read_text(encoding="utf-8"))
    maes: list[float] = []
    ssims: list[float] = []
    model.eval()
    for entry in entries:
        width, height, _fps = parse_geometry(entry["name"])
        frames = memmap_frames(Path(entry["path"]), width, height)
        frame_index = (
            int(entry["anchor_index"])
            if args.validation_frame_index < 0
            else min(args.validation_frame_index, frames.shape[0] - 1)
        )
        ys, xs = central_roi(height, width)
        temporal = aligned_temporal_trimmed_mean(
            frames,
            frame_index,
            ys,
            xs,
            frames.shape[0],
            args.validation_temporal_window,
            trim_fraction=0.10,
            exclude_frame_index=frame_index,
        )
        aligned = prepare_aligned_burst(frames, frame_index, args.input_frames)
        prediction = denoise_burst(
            model,
            aligned,
            parse_exposure_ms(entry["name"]),
            device,
            tile_size=args.validation_tile_size,
        )
        maes.append(float(np.mean(np.abs(prediction - temporal))))
        prediction_tensor = torch.from_numpy(prediction[None, None]).to(device) / RAW_MAX
        temporal_tensor = torch.from_numpy(temporal[None, None]).to(device) / RAW_MAX
        ssims.append(float(ssim_map(prediction_tensor, temporal_tensor).mean()))
    model.train()
    return float(np.mean(maes)), float(np.mean(ssims))


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    best: dict[str, float],
    validation_metrics: tuple[float, float] | None = None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best": best,
            "validation": validation_metrics,
            "args": vars(args) | {"out_dir": str(args.out_dir)},
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-mode",
        choices=("synthetic", "real", "video", "mixed"),
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, default=None)
    parser.add_argument(
        "--video-fraction",
        type=float,
        default=0.5,
        help="Fraction of mixed-mode samples drawn from the DAVIS video burst set.",
    )
    parser.add_argument("--motion-blur-strength", type=float, default=1.0)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-frames", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-epoch", type=int, default=4000)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument(
        "--center-mode",
        choices=("anchor", "random"),
        default="random",
        help="Use fixed cached anchors for the alignment control run, or random centers.",
    )
    parser.add_argument("--validation-frame-index", type=int, default=10)
    parser.add_argument("--validation-temporal-window", type=int, default=16)
    parser.add_argument("--validation-tile-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-weight", type=float, default=0.20)
    parser.add_argument("--ssim-weight", type=float, default=0.10)
    parser.add_argument("--mean-weight", type=float, default=0.50)
    parser.add_argument("--flow-weight", type=float, default=1e-3)
    parser.add_argument("--teacher-weight-start", type=float, default=0.15)
    parser.add_argument("--teacher-weight-end", type=float, default=0.05)
    parser.add_argument("--noise-jitter", type=float, default=0.15)
    parser.add_argument("--max-motion-shift", type=float, default=2.0)
    parser.add_argument("--max-motion-rotation", type=float, default=0.25)
    parser.add_argument("--motion-strength", type=float, default=1.0)
    parser.add_argument("--exposure-ms-min", type=float, default=10.0)
    parser.add_argument("--exposure-ms-max", type=float, default=400.0)
    parser.add_argument("--black-level-dn", type=float, default=60.0)
    parser.add_argument("--black-drift-sigma", type=float, default=0.0)
    parser.add_argument("--fixed-pattern-sigma", type=float, default=0.0)
    parser.add_argument("--row-noise-sigma", type=float, default=0.0)
    parser.add_argument("--column-noise-sigma", type=float, default=0.0)
    parser.add_argument("--dark-variance-per-s-min", type=float, default=0.0)
    parser.add_argument("--dark-variance-per-s-max", type=float, default=6.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--reset-best",
        action="store_true",
        help="Ignore saved best metrics when fine-tuning under a new validation protocol.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for BurstNAFNet training.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    model = build_burst_nafnet(
        width=args.width,
        input_frames=args.input_frames,
        use_checkpoint=not args.no_checkpoint,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    global_step = 0
    best = {"mae": float("inf"), "ssim": float("-inf"), "composite": float("inf")}
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("step", 0))
        if not args.reset_best:
            best.update(state.get("best", {}))
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"BurstNAFNet: {parameter_count / 1e6:.2f} M parameters, "
        f"{args.input_frames} frames, width={args.width}, patch={args.patch_size}"
    )
    train_log = args.out_dir / "train_log.csv"
    validation_log = args.out_dir / "validation_log.csv"
    if not train_log.exists():
        train_log.write_text(
            "epoch,step,total,pixel,gradient,ssim_loss,mean,teacher,flow,"
            "attention_entropy,teacher_weight,sec,max_mem_gb\n",
            encoding="utf-8",
        )
    if not validation_log.exists():
        validation_log.write_text(
            "epoch,step,mae_dn,ssim,composite\n",
            encoding="utf-8",
        )

    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        progress = epoch / max(args.epochs - 1, 1)
        teacher_weight = (
            args.teacher_weight_start * (1.0 - progress)
            + args.teacher_weight_end * progress
        )
        totals = []
        latest: dict[str, float] = {}
        for micro_step, batch in enumerate(loader, start=1):
            burst = batch["burst"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            teacher_valid = batch["teacher_valid"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction, aux = model(burst, return_aux=True)
                total, latest = burst_loss(
                    prediction,
                    target,
                    teacher,
                    mask,
                    teacher_valid,
                    aux,
                    teacher_weight,
                    args.grad_weight,
                    args.ssim_weight,
                    args.mean_weight,
                    args.flow_weight,
                )
            (total / args.accum_steps).backward()
            totals.append(float(total.detach()))
            if micro_step % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        elapsed = time.time() - epoch_start
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        mean_total = float(np.mean(totals))
        print(
            f"epoch {epoch}: total={mean_total:.5f} pixel={latest['pixel']:.5f} "
            f"grad={latest['gradient']:.5f} ssim={latest['ssim_loss']:.5f} "
            f"mean={latest['mean']:.5f} teacher={latest['teacher']:.5f} "
            f"{elapsed:.1f}s mem={max_mem:.2f}GB"
        )
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{global_step},{mean_total:.7f},{latest['pixel']:.7f},"
                f"{latest['gradient']:.7f},{latest['ssim_loss']:.7f},"
                f"{latest['mean']:.7f},{latest['teacher']:.7f},{latest['flow']:.7f},"
                f"{latest['attention_entropy']:.7f},{teacher_weight:.6f},"
                f"{elapsed:.2f},{max_mem:.3f}\n"
            )

        metrics = None
        should_validate = (
            args.validation_manifest is not None
            and ((epoch + 1) % args.validation_every == 0 or stop)
        )
        if should_validate:
            mae, ssim = validate_full_frames(model, args, device)
            composite = mae / 1023.0 + (1.0 - ssim)
            metrics = (mae, ssim)
            print(
                f"validation: MAE={mae:.4f} DN SSIM={ssim:.6f} "
                f"composite={composite:.6f}"
            )
            with validation_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch},{global_step},{mae:.6f},{ssim:.8f},{composite:.8f}\n"
                )
            candidates = (
                ("mae", mae, args.out_dir / "best_mae.pt", mae < best["mae"]),
                ("ssim", ssim, args.out_dir / "best_ssim.pt", ssim > best["ssim"]),
                (
                    "composite",
                    composite,
                    args.out_dir / "best.pt",
                    composite < best["composite"],
                ),
            )
            for name, value, path, improved in candidates:
                if improved:
                    best[name] = value
                    save_checkpoint(
                        path,
                        model,
                        optimizer,
                        epoch + 1,
                        global_step,
                        args,
                        best,
                        metrics,
                    )

        save_checkpoint(
            args.out_dir / "last.pt",
            model,
            optimizer,
            epoch + 1,
            global_step,
            args,
            best,
            metrics,
        )
        if (epoch + 1) % 20 == 0:
            save_checkpoint(
                args.out_dir / f"epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                epoch + 1,
                global_step,
                args,
                best,
                metrics,
            )
        if stop:
            print("Reached max steps.")
            break


if __name__ == "__main__":
    main()
