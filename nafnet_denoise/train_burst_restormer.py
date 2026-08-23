"""Train 16f BurstRestormer with temporal targets and flat highpass-σ loss."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .burst_restormer import (
    build_burst_restormer,
    load_burst_restormer_weights_partial,
    load_restormer_weights_partial,
)
from .common import (
    RAW_MAX,
    decode_mono10,
    memmap_frames,
    model_output_to_raw,
    parse_exposure_ms,
    parse_geometry,
)
from .infer_burst import denoise_burst, prepare_aligned_burst
from .train_burst import (
    build_dataset,
    burst_loss,
    flow_smoothness,
    masked_charbonnier,
    masked_mean,
    save_checkpoint,
    ssim_map,
)
from .train_distill import (
    denoise_edge_score,
    edge_metrics_dn,
    highpass,
    soft_edge_mask,
)
from .validate import aligned_temporal_trimmed_mean, central_roi, highpass_noise_sigma


def masked_mad(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-batch median absolute deviation of masked values, then mean over batch."""
    batch = values.shape[0]
    mads = []
    flat_values = values.reshape(batch, -1)
    flat_mask = mask.reshape(batch, -1) > 0.5
    for index in range(batch):
        sample = flat_values[index][flat_mask[index]]
        if sample.numel() < 16:
            mads.append(values.new_zeros(()))
            continue
        median = sample.median()
        mads.append((sample - median).abs().median())
    return torch.stack(mads).mean()


def burst_restormer_loss(
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
    flat_hp_weight: float,
    flat_sigma_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    total, metrics = burst_loss(
        prediction,
        target,
        teacher,
        mask,
        teacher_valid,
        aux,
        teacher_weight,
        grad_weight,
        ssim_weight,
        mean_weight,
        flow_weight,
    )
    reliability = mask.clamp(0.0, 1.0)
    edge_map = soft_edge_mask(target)
    flat_map = 1.0 - edge_map
    flat_region = reliability * flat_map
    if float(flat_region.mean()) > 1e-4:
        flat_hp = masked_charbonnier(
            highpass(prediction),
            highpass(target),
            flat_region,
        )
        sigma_pred = masked_mad(highpass(prediction), flat_region)
        sigma_tgt = masked_mad(highpass(target), flat_region)
        flat_sigma = (sigma_pred - sigma_tgt).abs()
    else:
        flat_hp = prediction.new_zeros(())
        flat_sigma = prediction.new_zeros(())
    total = total + flat_hp_weight * flat_hp + flat_sigma_weight * flat_sigma
    metrics = {
        **metrics,
        "flat_hp": float(flat_hp.detach()),
        "flat_sigma": float(flat_sigma.detach()),
        "flow": float(flow_smoothness(aux["flows"]).detach()),
    }
    return total, metrics


@torch.no_grad()
def validate_full_frames_des(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """MAE / SSIM / edge retention / DES on validation RAW clips."""
    files = sorted(Path(args.validation_dir).rglob("*_pMono10_f*.raw"))
    if not files:
        raise ValueError(f"No validation RAW videos in {args.validation_dir}")

    model.eval()
    maes: list[float] = []
    ssims: list[float] = []
    retentions: list[float] = []
    edge_maes: list[float] = []
    des_scores: list[float] = []
    for path in files:
        width, height, _fps = parse_geometry(path.name)
        frames = memmap_frames(path, width, height)
        frame_index = (
            frames.shape[0] // 2
            if args.validation_frame_index < 0
            else min(args.validation_frame_index, frames.shape[0] - 1)
        )
        ys, xs = central_roi(height, width)
        input_dn = decode_mono10(frames[frame_index])
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
            parse_exposure_ms(path.name),
            device,
            tile_size=args.validation_tile_size,
        )
        maes.append(float(np.mean(np.abs(prediction - temporal))))
        prediction_tensor = torch.from_numpy(prediction[None, None]).to(device) / RAW_MAX
        temporal_tensor = torch.from_numpy(temporal[None, None]).to(device) / RAW_MAX
        ssims.append(float(ssim_map(prediction_tensor, temporal_tensor).mean()))
        retention, edge_mae = edge_metrics_dn(prediction, temporal)
        retentions.append(retention)
        edge_maes.append(edge_mae)
        des, _ng, _ef = denoise_edge_score(
            highpass_noise_sigma(prediction, ys, xs),
            highpass_noise_sigma(input_dn, ys, xs),
            retention,
        )
        des_scores.append(des)
    model.train()
    return (
        float(np.mean(maes)),
        float(np.mean(ssims)),
        float(np.nanmean(retentions)),
        float(np.nanmean(edge_maes)),
        float(np.mean(des_scores)),
    )


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-mode",
        choices=("synthetic", "real", "video", "mixed"),
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, default=None)
    parser.add_argument("--video-fraction", type=float, default=0.5)
    parser.add_argument("--motion-blur-strength", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-frames", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--num-blocks", type=str, default="2,3,3,4")
    parser.add_argument("--num-refinement-blocks", type=int, default=2)
    parser.add_argument("--heads", type=str, default="1,2,4,8")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument(
        "--center-mode",
        choices=("anchor", "random"),
        default="random",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(r"D:\denoise\素材\降噪素材"),
    )
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--validation-frame-index", type=int, default=-1)
    parser.add_argument("--validation-temporal-window", type=int, default=16)
    parser.add_argument("--validation-tile-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-weight", type=float, default=0.20)
    parser.add_argument("--ssim-weight", type=float, default=0.10)
    parser.add_argument("--mean-weight", type=float, default=0.50)
    parser.add_argument("--flow-weight", type=float, default=1e-3)
    parser.add_argument(
        "--teacher-weight-start",
        type=float,
        default=0.0,
        help="BM3D teacher disabled by default for pure-network training.",
    )
    parser.add_argument("--teacher-weight-end", type=float, default=0.0)
    parser.add_argument("--flat-hp-weight", type=float, default=0.45)
    parser.add_argument("--flat-sigma-weight", type=float, default=0.35)
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
        "--init-restormer",
        type=Path,
        default=None,
        help="Partial warm-start from a single-frame Restormer checkpoint.",
    )
    parser.add_argument(
        "--init-burst-checkpoint",
        type=Path,
        default=None,
        help="Partial warm-start from another BurstRestormer ckpt (e.g. 16f→4f).",
    )
    parser.add_argument("--fresh-resume", action="store_true")
    parser.add_argument("--reset-best", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()
    args.num_blocks_tuple = parse_int_tuple(args.num_blocks)
    args.heads_tuple = parse_int_tuple(args.heads)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for BurstRestormer training.")
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
    model = build_burst_restormer(
        width=args.width,
        input_frames=args.input_frames,
        dim=args.dim,
        num_blocks=args.num_blocks_tuple,
        num_refinement_blocks=args.num_refinement_blocks,
        heads=args.heads_tuple,
        use_checkpoint=not args.no_checkpoint,
    ).to(device)

    if args.init_burst_checkpoint is not None and args.resume is None:
        loaded, skipped = load_burst_restormer_weights_partial(
            model,
            args.init_burst_checkpoint,
            device,
        )
        print(
            f"Partial BurstRestormer init from {args.init_burst_checkpoint}: "
            f"loaded={loaded} skipped={skipped}",
            flush=True,
        )
    elif args.init_restormer is not None and args.resume is None:
        loaded, skipped = load_restormer_weights_partial(
            model,
            args.init_restormer,
            device,
        )
        print(
            f"Partial Restormer init from {args.init_restormer}: "
            f"loaded={loaded} skipped={skipped}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    global_step = 0
    best = {
        "mae": float("inf"),
        "ssim": float("-inf"),
        "composite": float("inf"),
        "des": float("-inf"),
    }
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        if not args.fresh_resume and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if args.fresh_resume:
            print(f"Loaded weights from {args.resume}; restarting metrics/epoch", flush=True)
        else:
            start_epoch = int(state.get("epoch", 0))
            global_step = int(state.get("step", 0))
            if not args.reset_best:
                best.update(state.get("best", {}))
            print(f"Resumed {args.resume} at epoch {start_epoch}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"BurstRestormer: {parameter_count / 1e6:.2f} M parameters, "
        f"{args.input_frames} frames, width={args.width}, dim={args.dim}, "
        f"blocks={args.num_blocks_tuple}, patch={args.patch_size}",
        flush=True,
    )
    train_log = args.out_dir / "train_log.csv"
    validation_log = args.out_dir / "validation_log.csv"
    if not train_log.exists():
        train_log.write_text(
            "epoch,step,total,pixel,gradient,ssim_loss,mean,teacher,flow,"
            "flat_hp,flat_sigma,attention_entropy,teacher_weight,sec,max_mem_gb\n",
            encoding="utf-8",
        )
    if not validation_log.exists():
        validation_log.write_text(
            "epoch,step,mae_dn,ssim,composite,edge_retention,edge_sobel_mae,des\n",
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
                total, latest = burst_restormer_loss(
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
                    args.flat_hp_weight,
                    args.flat_sigma_weight,
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
            f"epoch {epoch}: total={mean_total:.5f} pixel={latest.get('pixel', 0):.5f} "
            f"flat_hp={latest.get('flat_hp', 0):.5f} flat_σ={latest.get('flat_sigma', 0):.5f} "
            f"{elapsed:.1f}s mem={max_mem:.2f}GB",
            flush=True,
        )
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{global_step},{mean_total:.7f},{latest.get('pixel', 0):.7f},"
                f"{latest.get('gradient', 0):.7f},{latest.get('ssim_loss', 0):.7f},"
                f"{latest.get('mean', 0):.7f},{latest.get('teacher', 0):.7f},"
                f"{latest.get('flow', 0):.7f},{latest.get('flat_hp', 0):.7f},"
                f"{latest.get('flat_sigma', 0):.7f},{latest.get('attention_entropy', 0):.7f},"
                f"{teacher_weight:.6f},{elapsed:.2f},{max_mem:.3f}\n"
            )

        metrics = None
        should_validate = (epoch + 1) % args.validation_every == 0 or stop or (
            epoch + 1 == args.epochs
        )
        if should_validate:
            mae, ssim, edge_ret, edge_mae, des = validate_full_frames_des(
                model,
                args,
                device,
            )
            composite = mae / 1023.0 + (1.0 - ssim)
            metrics = (mae, ssim)
            print(
                f"validation: MAE={mae:.4f} DN SSIM={ssim:.6f} "
                f"edge_ret={edge_ret:.4f} DES={des:.4f} composite={composite:.6f}",
                flush=True,
            )
            with validation_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch},{global_step},{mae:.6f},{ssim:.8f},{composite:.8f},"
                    f"{edge_ret:.6f},{edge_mae:.6f},{des:.6f}\n"
                )
            if mae < best["mae"]:
                best["mae"] = mae
                save_checkpoint(
                    args.out_dir / "best_mae.pt",
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    args,
                    best,
                    metrics,
                )
            if ssim > best["ssim"]:
                best["ssim"] = ssim
                save_checkpoint(
                    args.out_dir / "best_ssim.pt",
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    args,
                    best,
                    metrics,
                )
            if composite < best["composite"]:
                best["composite"] = composite
                save_checkpoint(
                    args.out_dir / "best_composite.pt",
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    args,
                    best,
                    metrics,
                )
            if des > best["des"]:
                best["des"] = des
                save_checkpoint(
                    args.out_dir / "best.pt",
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    args,
                    best,
                    metrics,
                )
                save_checkpoint(
                    args.out_dir / "best_des.pt",
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
        if stop:
            print("Reached max steps.", flush=True)
            break

    print(f"done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
