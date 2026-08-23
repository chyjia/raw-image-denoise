"""Train the conditional Restormer denoiser on the VST-domain patch dataset.

Memory strategy for an 8 GB GPU:
* per-block gradient checkpointing (recompute activations instead of storing);
* bf16 mixed precision autocast;
* gradient accumulation to reach a large effective batch with small micro-batches;
* clean references cached in system RAM and streamed with pinned memory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import common  # noqa: F401  (kept for consistent module resolution)
from dataset import VstDenoiseDataset
from restormer import Restormer


def image_gradients(x: torch.Tensor):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mean_weight: float,
    gradient_weight: float,
    flat_smooth_weight: float,
    flat_edge_scale: float,
    flat_gradient_gate: float = 0.0,
):
    pixel = F.l1_loss(pred, target)

    pred_mean = pred.mean(dim=(1, 2, 3))
    target_mean = target.mean(dim=(1, 2, 3))
    mean_loss = (pred_mean - target_mean).abs().mean()

    tdx, tdy = image_gradients(target)
    pdx, pdy = image_gradients(pred)
    grad_loss = F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)

    flat_wx = torch.exp(-flat_edge_scale * tdx.abs())
    flat_wy = torch.exp(-flat_edge_scale * tdy.abs())
    flat_per_sample = (flat_wx * pdx.abs()).mean(dim=(1, 2, 3)) + (
        flat_wy * pdy.abs()
    ).mean(dim=(1, 2, 3))
    if flat_gradient_gate > 0.0:
        target_grad_energy = tdx.abs().mean(dim=(1, 2, 3)) + tdy.abs().mean(dim=(1, 2, 3))
        flat_mask = (target_grad_energy < flat_gradient_gate).float()
        denom = flat_mask.sum().clamp_min(1.0)
        flat_loss = (flat_per_sample * flat_mask).sum() / denom
    else:
        flat_loss = flat_per_sample.mean()

    total = (
        pixel
        + mean_weight * mean_loss
        + gradient_weight * grad_loss
        + flat_smooth_weight * flat_loss
    )
    return total, {
        "pixel": pixel.item(),
        "mean": mean_loss.item(),
        "grad": grad_loss.item(),
        "flat": flat_loss.item(),
    }


def worker_init_fn(worker_id: int) -> None:
    import numpy as np

    info = torch.utils.data.get_worker_info()
    dataset = info.dataset
    dataset.rng = np.random.default_rng(1000 + worker_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("restormer/cache/train_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("restormer/checkpoints"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--iters-per-epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--mean-weight", type=float, default=0.5)
    parser.add_argument("--gradient-weight", type=float, default=0.2)
    parser.add_argument("--flat-smooth-weight", type=float, default=0.03)
    parser.add_argument("--flat-edge-scale", type=float, default=80.0)
    parser.add_argument(
        "--flat-gradient-gate",
        type=float,
        default=0.0,
        help="Apply flat smooth loss only when mean target gradient is below this (VST).",
    )
    parser.add_argument(
        "--flat-roi-fraction",
        type=float,
        default=0.0,
        help="Fraction of patches sampled from the central flat ROI (0-1).",
    )
    parser.add_argument(
        "--brightness-condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add auxiliary brightness condition channel.",
    )
    parser.add_argument(
        "--brightness-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exposure-relative brightness offset (0.5=typical) instead of absolute DN.",
    )
    parser.add_argument(
        "--exposure-priors",
        type=Path,
        default=None,
        help="JSON map of exposure_ms -> expected ROI mean DN.",
    )
    parser.add_argument(
        "--no-align-target-brightness",
        action="store_true",
        help="Disable per-frame brightness alignment of the clean target.",
    )
    parser.add_argument("--exposure-boost-ms", type=float, default=50.0)
    parser.add_argument("--exposure-boost-sigma", type=float, default=15.0)
    parser.add_argument("--fps20-boost", type=float, default=3.0)
    parser.add_argument("--exposure-boost", type=float, default=3.0)
    parser.add_argument("--flicker-boost", type=float, default=1.5)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing.")
    parser.add_argument("--max-steps", type=int, default=0, help="If >0, stop after this many optimizer steps (smoke test).")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--load-weights",
        type=Path,
        default=None,
        help="Load model weights only (fresh optimizer) for fine-tuning.",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VstDenoiseDataset(
        args.manifest,
        patch_size=args.patch_size,
        samples_per_epoch=args.iters_per_epoch * args.micro_batch,
        augment=True,
        flat_roi_fraction=args.flat_roi_fraction,
        brightness_condition=args.brightness_condition,
        brightness_offset=args.brightness_offset,
        exposure_priors_path=args.exposure_priors,
        align_target_brightness=not args.no_align_target_brightness,
        exposure_boost_ms=args.exposure_boost_ms,
        exposure_boost_sigma=args.exposure_boost_sigma,
        fps20_boost=args.fps20_boost,
        exposure_boost=args.exposure_boost,
        flicker_boost=args.flicker_boost,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    model = Restormer(
        inp_channels=dataset.inp_channels,
        out_channels=1,
        dim=args.dim,
        use_checkpoint=not args.no_checkpoint,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Restormer parameters: {param_count/1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    total_steps = args.epochs * args.iters_per_epoch // args.accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    start_epoch = 0
    if args.load_weights and args.load_weights.exists():
        state = torch.load(args.load_weights, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"Loaded weights from {args.load_weights} (epoch {state.get('epoch', '?')}) for fine-tuning")
    elif args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    global_step = 0
    best_pixel = float("inf")
    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {"pixel": 0.0, "mean": 0.0, "grad": 0.0, "flat": 0.0}
        count = 0
        epoch_start = time.time()

        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(inputs)
                loss, parts = combined_loss(
                    pred.float(),
                    targets.float(),
                    args.mean_weight,
                    args.gradient_weight,
                    args.flat_smooth_weight,
                    args.flat_edge_scale,
                    args.flat_gradient_gate,
                )
            (loss / args.accum_steps).backward()

            for key in running:
                running[key] += parts[key]
            count += 1

            if (i + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        elapsed = time.time() - epoch_start
        avg = {k: v / max(count, 1) for k, v in running.items()}
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"epoch {epoch} | steps {global_step} | pixel {avg['pixel']:.5f} "
            f"mean {avg['mean']:.5f} grad {avg['grad']:.5f} flat {avg['flat']:.5f} "
            f"| {elapsed:.1f}s | peak {peak:.2f} GB | lr {scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        checkpoint_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(checkpoint_state, args.output_dir / "last.pt")
        if avg["pixel"] < best_pixel:
            best_pixel = avg["pixel"]
            torch.save(checkpoint_state, args.output_dir / "best.pt")

        if stop:
            print("Reached max_steps; stopping (smoke test).", flush=True)
            break

    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
