"""Train NAFNet Mono10 denoiser on PixelShift200 with PTC-calibrated synthetic noise.

The noise model uses the current grayscale-chart PTC calibration:
    variance_dn2 = 0.10618515 * mean_dn + 2.00145577

Usage:
    python -m nafnet_denoise.build_dataset \\
        --input-dir "D:/denoise/素材/PixelShift200_train" \\
        --cache-dir nafnet_denoise/cache

    python -m nafnet_denoise.train \\
        --manifest nafnet_denoise/cache/manifest.json \\
        --out-dir nafnet_denoise/checkpoints \\
        --patch-size 128 --micro-batch 4 --accum-steps 4 --epochs 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PixelShiftPatchDataset
from .nafnet import build_nafnet


def build_optimizer(model: torch.nn.Module, lr: float):
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer, "bitsandbytes.PagedAdamW8bit"
    except Exception as error:  # noqa: BLE001
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer, f"torch.AdamW (fallback: {error})"


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    return torch.sqrt((pred - target).square() + epsilon**2).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return charbonnier_loss(pred_dx, target_dx) + charbonnier_loss(pred_dy, target_dy)


def ssim_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Differentiable SSIM for normalized VST images in approximately [0, 1]."""
    padding = window_size // 2
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    pred_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding)
    target_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=padding)
    cross = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding)
    var_pred = torch.clamp(pred_sq - mu_pred.square(), min=0.0)
    var_target = torch.clamp(target_sq - mu_target.square(), min=0.0)
    covariance = cross - mu_pred * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_pred * mu_target + c1) * (2 * covariance + c2)
    denominator = (mu_pred.square() + mu_target.square() + c1) * (
        var_pred + var_target + c2
    )
    return (numerator / torch.clamp(denominator, min=1e-8)).mean()


def denoise_loss(
    pred,
    target,
    grad_weight: float,
    ssim_weight: float,
    mean_weight: float,
):
    pixel = charbonnier_loss(pred, target)
    grad = gradient_loss(pred, target)
    ssim = ssim_index(pred, target)
    structural = 1.0 - ssim
    mean = (pred.mean(dim=(1, 2, 3)) - target.mean(dim=(1, 2, 3))).abs().mean()
    total = pixel + grad_weight * grad + ssim_weight * structural + mean_weight * mean
    return total, pixel, grad, structural, mean


def validate_real_sequences(
    model: torch.nn.Module,
    validation_dir: Path,
    frame_index: int,
    input_frames: int,
    device: torch.device,
    tile: int,
) -> tuple[float, float]:
    """Evaluate against held-out real RAW bursts and a temporal proxy target."""
    from .common import (
        decode_mono10,
        memmap_frames,
        parse_exposure_ms,
        parse_geometry,
    )
    from .infer import denoise_frame
    from .validate import aligned_temporal_trimmed_mean, central_roi

    files = sorted(validation_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise ValueError(f"No validation RAW videos found in {validation_dir}")

    was_training = model.training
    model.eval()
    maes: list[float] = []
    ssims: list[float] = []
    for path in files:
        frame_width, frame_height, _fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, frame_width, frame_height)
        if frame_index >= frames.shape[0]:
            raise ValueError(
                f"{path.name}: validation frame {frame_index} exceeds "
                f"available range 0-{frames.shape[0] - 1}"
            )
        ys, xs = central_roi(frame_height, frame_width)
        temporal = aligned_temporal_trimmed_mean(
            frames,
            frame_index,
            ys,
            xs,
            frames.shape[0],
            window=16,
            trim_fraction=0.10,
            exclude_frame_index=frame_index,
        )
        output = denoise_frame(
            model,
            frames,
            frame_index,
            exposure_ms,
            device,
            input_frames=input_frames,
            tile=tile,
        )
        maes.append(float(np.mean(np.abs(output - temporal))))
        output_tensor = torch.from_numpy(output[None, None]).to(device) / 1023.0
        target_tensor = torch.from_numpy(temporal[None, None]).to(device) / 1023.0
        with torch.no_grad():
            ssims.append(float(ssim_index(output_tensor, target_tensor)))

    if was_training:
        model.train()
    return float(np.mean(maes)), float(np.mean(ssims))


def save_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    validation_mae: float,
    validation_ssim: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "validation_mae_dn": validation_mae,
            "validation_ssim": validation_ssim,
            "args": vars(args) | {"out_dir": str(args.out_dir)},
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("nafnet_denoise/checkpoints"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patches-per-epoch", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-weight", type=float, default=0.20)
    parser.add_argument("--ssim-weight", type=float, default=0.10)
    parser.add_argument("--mean-weight", type=float, default=0.10)
    parser.add_argument(
        "--noise-jitter",
        type=float,
        default=0.15,
        help="Relative PTC slope/intercept jitter; 0.15 means ±15%%.",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--input-frames", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-ram-cache", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(r"D:\denoise\素材\训练素材\不参加训练，用来验证训练模型效果"),
    )
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-frame-index", type=int, default=10)
    parser.add_argument("--validation-tile", type=int, default=512)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device. Use cpu for smoke tests without a compatible GPU build.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = PixelShiftPatchDataset(
        args.manifest,
        patch_size=args.patch_size,
        patches_per_epoch=args.patches_per_epoch,
        input_frames=args.input_frames,
        noise_jitter=args.noise_jitter,
        cache_images_in_ram=not args.no_ram_cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_nafnet(width=args.width, input_frames=args.input_frames).to(device)
    optimizer, optimizer_name = build_optimizer(model, args.lr)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Input frames: {args.input_frames}")
    print(f"Model parameters: {param_count / 1e6:.2f} M")
    print(f"Optimizer: {optimizer_name}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    start_epoch = 0
    global_step = 0
    best_mae = float("inf")
    best_ssim = float("-inf")
    best_composite = float("inf")
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.lr
        start_epoch = state.get("epoch", 0)
        global_step = state.get("step", 0)
        best_mae = state.get("best_mae_dn", best_mae)
        best_ssim = state.get("best_ssim", best_ssim)
        best_composite = state.get("best_composite", best_composite)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_path = args.out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,step,total,charbonnier,grad,ssim_loss,mean,sec,max_mem_gb\n",
            encoding="utf-8",
        )
    validation_log = args.out_dir / "validation_log.csv"
    if not validation_log.exists():
        validation_log.write_text(
            "epoch,step,mae_dn,ssim,composite\n",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        running = 0.0
        micro_index = 0
        last = {
            "total": 0.0,
            "pixel": 0.0,
            "grad": 0.0,
            "ssim_loss": 0.0,
            "mean": 0.0,
        }

        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=device.type == "cuda",
            ):
                pred = model(inputs)
                total, pixel, grad, structural, mean = denoise_loss(
                    pred,
                    targets,
                    args.grad_weight,
                    args.ssim_weight,
                    args.mean_weight,
                )
            (total / args.accum_steps).backward()
            micro_index += 1
            running += float(total.detach())
            last = {
                "total": float(total.detach()),
                "pixel": float(pixel.detach()),
                "grad": float(grad.detach()),
                "ssim_loss": float(structural.detach()),
                "mean": float(mean.detach()),
            }

            if micro_index % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    break

        elapsed = time.time() - epoch_start
        if device.type == "cuda":
            max_mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        else:
            max_mem = 0.0
        steps = max(micro_index, 1)
        print(
            f"epoch {epoch}: total={running / steps:.5f} charb={last['pixel']:.5f} "
            f"grad={last['grad']:.5f} ssim_loss={last['ssim_loss']:.5f} "
            f"mean={last['mean']:.5f} "
            f"{elapsed:.1f}s peak_mem={max_mem:.2f} GB"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{global_step},{running / steps:.6f},{last['pixel']:.6f},"
                f"{last['grad']:.6f},{last['ssim_loss']:.6f},{last['mean']:.6f},"
                f"{elapsed:.2f},{max_mem:.3f}\n"
            )

        should_validate = (
            not args.no_validation
            and (
                (epoch + 1) % args.validation_every == 0
                or epoch + 1 == args.epochs
                or (args.max_steps and global_step >= args.max_steps)
            )
        )
        if should_validate:
            validation_mae, validation_ssim = validate_real_sequences(
                model,
                args.validation_dir,
                args.validation_frame_index,
                args.input_frames,
                device,
                args.validation_tile,
            )
            composite = validation_mae / 1023.0 + (1.0 - validation_ssim)
            print(
                f"validation: MAE={validation_mae:.4f} DN "
                f"SSIM={validation_ssim:.6f} composite={composite:.6f}"
            )
            with validation_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch},{global_step},{validation_mae:.6f},"
                    f"{validation_ssim:.8f},{composite:.8f}\n"
                )
            if validation_mae < best_mae:
                best_mae = validation_mae
                save_model_checkpoint(
                    args.out_dir / "best_mae.pt",
                    model,
                    epoch + 1,
                    global_step,
                    args,
                    validation_mae,
                    validation_ssim,
                )
            if validation_ssim > best_ssim:
                best_ssim = validation_ssim
                save_model_checkpoint(
                    args.out_dir / "best_ssim.pt",
                    model,
                    epoch + 1,
                    global_step,
                    args,
                    validation_mae,
                    validation_ssim,
                )
            if composite < best_composite:
                best_composite = composite
                save_model_checkpoint(
                    args.out_dir / "best.pt",
                    model,
                    epoch + 1,
                    global_step,
                    args,
                    validation_mae,
                    validation_ssim,
                )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "best_mae_dn": best_mae,
                "best_ssim": best_ssim,
                "best_composite": best_composite,
                "args": vars(args) | {"out_dir": str(args.out_dir)},
            },
            args.out_dir / "last.pt",
        )
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), args.out_dir / f"epoch_{epoch + 1:03d}.pt")

        if args.max_steps and global_step >= args.max_steps:
            print("Reached max-steps; stopping.")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
