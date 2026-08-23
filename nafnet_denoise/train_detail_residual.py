"""Train a frozen-base detail residual head on flat-region BM3D residuals."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .burst_dataset import RealBurstDataset
from .detail_residual import DetailResidualNet, residual_conditioning_input
from .infer import load_model
from .nafnet import build_nafnet
from .train import build_optimizer, save_model_checkpoint
from .common import expand_intro_state_for_sigma
from .train_distill import (
    RealNAFNetDistillDataset,
    collate_distill,
    denoise_edge_score,
    edge_metrics_dn,
    highpass,
    masked_charbonnier,
    soft_edge_mask,
)


def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.clamp(0.0, 1.0)
    denom = weight.mean().clamp_min(1e-6)
    return ((pred - target).abs() * weight).mean() / denom


def residual_loss(
    residual: torch.Tensor,
    base_pred: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    teacher_valid: torch.Tensor,
    flat_weight: float,
    edge_suppress: float,
    highpass_weight: float,
    flat_l1_weight: float = 0.0,
):
    use_teacher = teacher_valid.view(-1, 1, 1, 1)
    edge_map = soft_edge_mask(target)
    flat_map = 1.0 - edge_map
    reliability = mask.clamp(0.0, 1.0)
    pred = base_pred + residual
    desired = torch.where(use_teacher > 0.5, teacher - base_pred, target - base_pred)
    flat_region = reliability * (flat_map * use_teacher + (1.0 - use_teacher))
    teacher_flat = reliability * flat_map * use_teacher
    if float(flat_region.mean()) > 1e-4:
        flat = masked_charbonnier(residual, desired, flat_region)
        hp = masked_charbonnier(
            highpass(pred),
            highpass(torch.where(use_teacher > 0.5, teacher, target)),
            flat_region,
        )
    else:
        flat = ((residual - desired).abs()).mean()
        hp = flat * 0.0
    if flat_l1_weight > 0.0 and float(teacher_flat.mean()) > 1e-4:
        flat_l1 = masked_l1(pred, teacher, teacher_flat)
    else:
        flat_l1 = pred.new_zeros(())
    edge_region = reliability * edge_map
    if float(edge_region.mean()) > 1e-4:
        edge = masked_charbonnier(residual, torch.zeros_like(residual), edge_region)
    else:
        edge = (residual.abs()).mean() * 0.0
    total = (
        flat_weight * flat
        + flat_l1_weight * flat_l1
        + edge_suppress * edge
        + highpass_weight * hp
    )
    return {
        "total": total,
        "flat": flat,
        "flat_l1": flat_l1,
        "edge": edge,
        "highpass": hp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="Frozen 4f NAFNet checkpoint (preferably sigma-conditioned).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_detail_residual"),
    )
    parser.add_argument("--input-frames", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--residual-width", type=int, default=16)
    parser.add_argument(
        "--residual-depth",
        type=int,
        default=0,
        help="U-Net downsample levels; 0 keeps the shallow conv stack.",
    )
    parser.add_argument("--use-sigma", action="store_true")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--flat-weight", type=float, default=1.0)
    parser.add_argument(
        "--flat-l1-weight",
        type=float,
        default=0.0,
        help="Direct flat-region L1 of (base+residual) vs BM3D teacher.",
    )
    parser.add_argument("--edge-suppress", type=float, default=0.75)
    parser.add_argument("--highpass-weight", type=float, default=0.45)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(r"D:\denoise\素材\降噪素材"),
    )
    parser.add_argument("--validation-every", type=int, default=3)
    parser.add_argument("--validation-frame-index", type=int, default=-1)
    parser.add_argument("--validation-tile", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    real = RealNAFNetDistillDataset(
        RealBurstDataset(
            args.real_manifest,
            patch_size=args.patch_size,
            input_frames=args.input_frames,
            samples_per_epoch=args.patches_per_epoch,
            center_mode="random",
            seed=3,
        ),
        use_sigma=args.use_sigma,
    )
    loader = DataLoader(
        real,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_distill,
    )

    base, loaded_frames, loaded_width = load_model(
        args.base_checkpoint,
        args.width,
        device,
        input_frames=args.input_frames,
    )
    if loaded_frames != args.input_frames:
        raise ValueError(f"Base frames mismatch: {loaded_frames} vs {args.input_frames}")
    args.width = loaded_width
    use_sigma = bool(getattr(base, "use_sigma", False)) or args.use_sigma
    if args.use_sigma and not getattr(base, "use_sigma", False):
        # Expand a legacy checkpoint into sigma-conditioned base for residual training.
        state = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
        weights = state["model"] if isinstance(state, dict) and "model" in state else state
        weights = expand_intro_state_for_sigma(weights, args.input_frames)
        base = build_nafnet(
            width=args.width,
            input_frames=args.input_frames,
            use_sigma=True,
        ).to(device)
        base.load_state_dict(weights)
        base.use_sigma = True
        use_sigma = True
    args.use_sigma = use_sigma
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)

    head = DetailResidualNet(
        width=args.residual_width,
        depth=args.residual_depth,
    ).to(device)
    optimizer, optimizer_name = build_optimizer(head, args.lr)
    print(
        f"device={device} optimizer={optimizer_name} use_sigma={use_sigma} "
        f"residual_width={args.residual_width} residual_depth={args.residual_depth} "
        f"flat_l1_weight={args.flat_l1_weight}",
        flush=True,
    )

    best_des = float("-inf")
    global_step = 0
    log_path = args.out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,step,total,flat,flat_l1,edge,highpass,sec,max_mem_gb\n",
            encoding="utf-8",
        )
    validation_log = args.out_dir / "validation_log.csv"
    if not validation_log.exists():
        validation_log.write_text(
            "epoch,step,mae_dn,ssim,composite,edge_retention,edge_sobel_mae,des\n",
            encoding="utf-8",
        )

    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        running = 0.0
        micro_index = 0
        last = {
            "total": 0.0,
            "flat": 0.0,
            "flat_l1": 0.0,
            "edge": 0.0,
            "highpass": 0.0,
        }

        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            teachers = batch["teacher"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            teacher_valid = batch["teacher_valid"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=device.type == "cuda",
            ):
                with torch.no_grad():
                    base_pred = base(inputs).float()
                cond = residual_conditioning_input(inputs, base_pred, args.input_frames)
                residual = head(cond)
                losses = residual_loss(
                    residual,
                    base_pred,
                    targets,
                    teachers,
                    masks,
                    teacher_valid,
                    args.flat_weight,
                    args.edge_suppress,
                    args.highpass_weight,
                    flat_l1_weight=args.flat_l1_weight,
                )
                total = losses["total"]
            (total / args.accum_steps).backward()
            micro_index += 1
            running += float(total.detach())
            last = {key: float(value.detach()) for key, value in losses.items()}
            if micro_index % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    break

        elapsed = time.time() - epoch_start
        max_mem = (
            torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        steps = max(micro_index, 1)
        print(
            f"epoch {epoch}: total={running / steps:.5f} flat={last['flat']:.5f} "
            f"flat_l1={last['flat_l1']:.5f} edge={last['edge']:.5f} "
            f"hp={last['highpass']:.5f} {elapsed:.1f}s mem={max_mem:.2f}GB",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{global_step},{running / steps:.6f},{last['flat']:.6f},"
                f"{last['flat_l1']:.6f},{last['edge']:.6f},{last['highpass']:.6f},"
                f"{elapsed:.2f},{max_mem:.3f}\n"
            )

        should_validate = (epoch + 1) % args.validation_every == 0 or epoch + 1 == args.epochs
        if should_validate:
            # Temporarily wrap base+head for validation via denoise_frame residual arg.
            mae, ssim, edge_retention, edge_sobel_mae, des = _validate_with_residual(
                base,
                head,
                args.validation_dir,
                args.validation_frame_index,
                args.input_frames,
                device,
                args.validation_tile,
            )
            composite = mae / 1023.0 + (1.0 - ssim)
            print(
                f"validation: MAE={mae:.4f} DN SSIM={ssim:.6f} "
                f"edge_ret={edge_retention:.4f} DES={des:.4f}",
                flush=True,
            )
            with validation_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch},{global_step},{mae:.6f},{ssim:.8f},{composite:.8f},"
                    f"{edge_retention:.6f},{edge_sobel_mae:.6f},{des:.6f}\n"
                )
            save_model_checkpoint(
                args.out_dir / "last.pt",
                head,
                epoch + 1,
                global_step,
                args,
                mae,
                ssim,
            )
            if des > best_des:
                best_des = des
                save_model_checkpoint(
                    args.out_dir / "best.pt",
                    head,
                    epoch + 1,
                    global_step,
                    args,
                    mae,
                    ssim,
                )
                # Also stash base path for inference wiring.
                (args.out_dir / "base_checkpoint.txt").write_text(
                    str(args.base_checkpoint.resolve()),
                    encoding="utf-8",
                )

        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"done -> {args.out_dir}", flush=True)


@torch.no_grad()
def _validate_with_residual(
    base: torch.nn.Module,
    head: torch.nn.Module,
    validation_dir: Path,
    frame_index: int,
    input_frames: int,
    device: torch.device,
    tile: int,
) -> tuple[float, float, float, float, float]:
    from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
    from .infer import denoise_frame
    from .validate import (
        aligned_temporal_trimmed_mean,
        central_roi,
        highpass_noise_sigma,
    )
    from .train import ssim_index

    files = sorted(validation_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise ValueError(f"No validation RAW videos found in {validation_dir}")

    head.eval()
    maes: list[float] = []
    ssims: list[float] = []
    retentions: list[float] = []
    edge_maes: list[float] = []
    des_scores: list[float] = []
    for path in files:
        frame_width, frame_height, _fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, frame_width, frame_height)
        target_index = frames.shape[0] // 2 if frame_index < 0 else frame_index
        ys, xs = central_roi(frame_height, frame_width)
        input_dn = decode_mono10(frames[target_index])
        temporal = aligned_temporal_trimmed_mean(
            frames,
            target_index,
            ys,
            xs,
            frames.shape[0],
            window=16,
            trim_fraction=0.10,
            exclude_frame_index=target_index,
        )
        output = denoise_frame(
            base,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=input_frames,
            tile=tile,
            residual_model=head,
        )
        maes.append(float(np.mean(np.abs(output - temporal))))
        output_tensor = torch.from_numpy(output[None, None]).to(device) / 1023.0
        target_tensor = torch.from_numpy(temporal[None, None]).to(device) / 1023.0
        ssims.append(float(ssim_index(output_tensor, target_tensor)))
        retention, edge_mae = edge_metrics_dn(output, temporal)
        retentions.append(retention)
        edge_maes.append(edge_mae)
        des, _ng, _ef = denoise_edge_score(
            highpass_noise_sigma(output, ys, xs),
            highpass_noise_sigma(input_dn, ys, xs),
            retention,
        )
        des_scores.append(des)
    return (
        float(np.mean(maes)),
        float(np.mean(ssims)),
        float(np.nanmean(retentions)),
        float(np.nanmean(edge_maes)),
        float(np.mean(des_scores)),
    )


if __name__ == "__main__":
    main()
