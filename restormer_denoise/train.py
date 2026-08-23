"""Train the conditional Restormer denoiser on an 8 GB GPU via RAM offload.

Memory strategy for the RTX 5060 (8 GB):
* bf16 autocast for activations and matmuls;
* gradient checkpointing on every transformer block (set in the model);
* gradient accumulation to reach a useful effective batch size;
* optimizer state offloaded to CPU RAM through a paged 8-bit optimizer
  (bitsandbytes ``PagedAdamW8bit``) when available, else a CPU-side AdamW.

Usage (inside WSL, torch venv):
    python -m restormer_denoise.train \
        --manifest restormer_denoise/cache/manifest.json \
        --out-dir restormer_denoise/checkpoints \
        --patch-size 128 --micro-batch 2 --accum-steps 8 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import VstPatchDataset
from .restormer import build_restormer


def build_optimizer(model: torch.nn.Module, lr: float):
    """Prefer a paged 8-bit optimizer that keeps its state in CPU RAM."""
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer, "bitsandbytes.PagedAdamW8bit (CPU-paged state)"
    except Exception as error:  # noqa: BLE001 - fall back to plain AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
        return optimizer, f"torch.AdamW (fallback: {error})"


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def denoise_loss(pred, target, grad_weight: float, mean_weight: float):
    pixel = F.l1_loss(pred, target)
    grad = gradient_loss(pred, target)
    mean = (pred.mean(dim=(1, 2, 3)) - target.mean(dim=(1, 2, 3))).abs().mean()
    total = pixel + grad_weight * grad + mean_weight * mean
    return total, pixel, grad, mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("restormer_denoise/checkpoints"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patches-per-epoch", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-weight", type=float, default=0.20)
    parser.add_argument("--mean-weight", type=float, default=0.50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps (smoke test).")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available inside this environment.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = VstPatchDataset(
        args.manifest,
        patch_size=args.patch_size,
        patches_per_epoch=args.patches_per_epoch,
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

    model = build_restormer(use_checkpoint=not args.no_checkpoint).to(device)
    optimizer, optimizer_name = build_optimizer(model, args.lr)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.2f} M")
    print(f"Optimizer: {optimizer_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    start_epoch = 0
    global_step = 0
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0)
        global_step = state.get("step", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_path = args.out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,total,pixel,grad,mean,sec,max_mem_gb\n", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        running = 0.0
        micro_index = 0
        last = {"total": 0.0, "pixel": 0.0, "grad": 0.0, "mean": 0.0}

        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(inputs)
                total, pixel, grad, mean = denoise_loss(
                    pred, targets, args.grad_weight, args.mean_weight
                )
            (total / args.accum_steps).backward()
            micro_index += 1
            running += float(total.detach())
            last = {
                "total": float(total.detach()),
                "pixel": float(pixel.detach()),
                "grad": float(grad.detach()),
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
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        steps = max(micro_index, 1)
        print(
            f"epoch {epoch}: total={running / steps:.5f} pixel={last['pixel']:.5f} "
            f"grad={last['grad']:.5f} mean={last['mean']:.5f} "
            f"{elapsed:.1f}s peak_mem={max_mem:.2f} GB"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{global_step},{running / steps:.6f},{last['pixel']:.6f},"
                f"{last['grad']:.6f},{last['mean']:.6f},{elapsed:.2f},{max_mem:.3f}\n"
            )

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch + 1, "step": global_step, "args": vars(args) | {"out_dir": str(args.out_dir)}},
            args.out_dir / "last.pt",
        )
        if args.max_steps and global_step >= args.max_steps:
            print("Reached max-steps; stopping (smoke test).")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
