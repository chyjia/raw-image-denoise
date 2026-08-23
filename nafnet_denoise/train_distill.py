"""Fine-tune NAFNet (1f/4f) with BM3D teacher distillation on real bursts.

Mixes PixelShift synthetic patches (clean GT) with real Mono10 bursts that have
offline PTC+VST+BM3D teachers. Highpass distillation cleans flats; Sobel edge
loss preserves sharpness against the non-BM3D target so the student does not
inherit teacher blur.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .burst_dataset import RealBurstDataset
from .common import (
    DEFAULT_DARK_VARIANCE_PER_S,
    EXPOSURE_NORM_MS,
    PTC_INTERCEPT,
    PTC_SLOPE,
    RAW_MAX,
    VST_MIN,
    VST_MAX,
    VST_SCALE,
    center_frame_index,
    expand_intro_state_for_input_frames,
    expand_intro_state_for_sigma,
)
from .dataset import PixelShiftPatchDataset
from .dual_head_nafnet import build_dual_head_nafnet, load_dual_head_from_nafnet_state
from .nafnet import build_nafnet, expand_state_dict_width
from .stacked_restormer import build_stacked_restormer, load_stacked_restormer_partial
from .train import (
    build_optimizer,
    charbonnier_loss,
    denoise_loss,
    save_model_checkpoint,
    ssim_index,
)


def intercept_from_exposure_map(
    exposure_map: torch.Tensor,
    dark_variance_per_s: float = DEFAULT_DARK_VARIANCE_PER_S,
) -> torch.Tensor:
    """Build per-sample VST intercept ``[B,1,1,1]`` from the exposure input channel."""
    exposure_ms = exposure_map[:, :, :1, :1] * EXPOSURE_NORM_MS
    return PTC_INTERCEPT + (exposure_ms / 1000.0) * max(dark_variance_per_s, 0.0)


def normalized_vst_to_dn(
    normalized: torch.Tensor,
    intercept: torch.Tensor | float = PTC_INTERCEPT,
) -> torch.Tensor:
    """Invert normalized VST → DN, then scale to ``[0, 1]`` for stable loss magnitudes."""
    transformed = normalized * VST_SCALE + VST_MIN
    if not torch.is_tensor(intercept):
        intercept = normalized.new_full((), float(intercept))
    image = (
        (PTC_SLOPE / 4.0) * transformed.square()
        - 0.375 * PTC_SLOPE
        - intercept / PTC_SLOPE
    )
    return image.clamp(0.0, RAW_MAX) / RAW_MAX


def maybe_to_loss_domain(
    *tensors: torch.Tensor,
    loss_domain: str,
    intercept: torch.Tensor | float,
) -> tuple[torch.Tensor, ...]:
    if loss_domain == "vst":
        return tensors
    return tuple(normalized_vst_to_dn(tensor, intercept) for tensor in tensors)


def highpass(image: torch.Tensor) -> torch.Tensor:
    blur = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
    return image - blur


def gaussian_blur(image: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """Separable Gaussian blur matching OpenCV ``GaussianBlur(..., sigmaX=sigma)``."""
    import math

    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * float(sigma) ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    channels = image.shape[1]
    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    blurred = F.conv2d(image, kernel_x, padding=(0, radius), groups=channels)
    return F.conv2d(blurred, kernel_y, padding=(radius, 0), groups=channels)


def des_highpass(image: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """Highpass residual used by ``validate.highpass_noise_sigma`` (σ=1.2)."""
    return image - gaussian_blur(image, sigma=sigma)


def sobel_xy(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Sobel ``(gx, gy)`` for BCHW tensors."""
    kernel_x = image.new_tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]]
    )
    kernel_y = image.new_tensor(
        [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]]
    )
    channels = image.shape[1]
    grad_x = F.conv2d(image, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    grad_y = F.conv2d(image, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return grad_x, grad_y


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude for BCHW tensors."""
    grad_x, grad_y = sobel_xy(image)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


def mixed_grad_xy_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """SBFBurst-style mixed gradient: match gx/gy on masked pixels."""
    gx_p, gy_p = sobel_xy(pred)
    gx_r, gy_r = sobel_xy(reference)
    w = mask.clamp(0.0, 1.0)
    denom = w.mean().clamp_min(1e-6)
    loss_x = ((gx_p - gx_r).abs() * w).mean() / denom
    loss_y = ((gy_p - gy_r).abs() * w).mean() / denom
    return 0.5 * (loss_x + loss_y)


def soft_edge_mask(image: torch.Tensor, temperature: float = 8.0) -> torch.Tensor:
    """Soft [0,1] edge map from per-sample Sobel percentiles."""
    magnitude = sobel_magnitude(image)
    flat = magnitude.flatten(2)
    lo = torch.quantile(flat, 0.50, dim=-1, keepdim=True).unsqueeze(-1)
    hi = torch.quantile(flat, 0.85, dim=-1, keepdim=True).unsqueeze(-1)
    scaled = (magnitude - lo) / (hi - lo).clamp_min(1e-6)
    return torch.sigmoid(temperature * (scaled - 0.5))


def masked_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    residual = torch.sqrt((pred - target).square() + epsilon**2)
    weight = mask.clamp(0.0, 1.0)
    denom = weight.mean().clamp_min(1e-6)
    return (residual * weight).mean() / denom


def numpy_sobel_magnitude(image_dn: np.ndarray) -> np.ndarray:
    image = image_dn.astype(np.float32)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def edge_metrics_dn(
    prediction_dn: np.ndarray,
    reference_dn: np.ndarray,
    percentile: float = 80.0,
) -> tuple[float, float]:
    """Return (edge_retention, edge_sobel_mae) on reference top-percentile edges."""
    reference_edge = numpy_sobel_magnitude(reference_dn)
    prediction_edge = numpy_sobel_magnitude(prediction_dn)
    threshold = float(np.percentile(reference_edge, percentile))
    mask = reference_edge >= threshold
    if not np.any(mask):
        return float("nan"), float("nan")
    reference_mean = float(reference_edge[mask].mean())
    retention = float(prediction_edge[mask].mean() / max(reference_mean, 1e-6))
    sobel_mae = float(np.mean(np.abs(prediction_edge - reference_edge)[mask]))
    return retention, sobel_mae


def denoise_edge_score(
    sigma: float,
    sigma_input: float,
    edge_retention: float,
    noise_weight: float = 0.6,
    edge_weight: float = 0.4,
) -> tuple[float, float, float]:
    """Composite Denoise-Edge Score (DES) in ``[0, 1]`` (higher is better).

    Combines:
    - ``noise_gain = clip(1 - sigma / sigma_input, 0, 1)``
    - ``edge_fidelity = clip(1 - |edge_retention - 1|, 0, 1)``

    Default weights favor denoising slightly while still punishing blur/oversharpen.
    """
    noise_gain = float(np.clip(1.0 - sigma / max(sigma_input, 1e-6), 0.0, 1.0))
    edge_fidelity = float(np.clip(1.0 - abs(edge_retention - 1.0), 0.0, 1.0))
    score = noise_weight * noise_gain + edge_weight * edge_fidelity
    return score, noise_gain, edge_fidelity


def masked_mad(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over batch of per-sample MAD on masked pixels."""
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


def des_noise_sigma(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """DES flat-noise σ: MAD(highpass) / 0.6745 on masked pixels."""
    return masked_mad(des_highpass(image), mask) / 0.6745


def harden_edge_map(edge_map: torch.Tensor, harden: float) -> torch.Tensor:
    """Sharpen soft edge map so BM3D flat teacher leaks less into structure."""
    if harden <= 0.0:
        return edge_map
    return torch.sigmoid(float(harden) * (edge_map - 0.5))


def des_edge_fidelity_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    reliability: torch.Tensor,
    percentile: float = 0.80,
    soft_k: float = 20.0,
) -> torch.Tensor:
    """Differentiable proxy of eval edge_fidelity vs temporal reference.

    Eval uses top-percentile Sobel retention; here we soft-gate the same
    percentile and penalize ``|retention - 1|`` (i.e. 1 - edge_fidelity when
    retention ∈ [0, 2]).
    """
    mag_ref = sobel_magnitude(reference)
    mag_pred = sobel_magnitude(pred)
    flat = mag_ref.flatten(2)
    thr = torch.quantile(flat, float(percentile), dim=-1, keepdim=True).unsqueeze(-1)
    edge_w = torch.sigmoid(soft_k * (mag_ref - thr) / thr.clamp_min(1e-6))
    edge_w = edge_w * reliability.clamp(0.0, 1.0)
    if float(edge_w.mean()) < 1e-4:
        return pred.new_zeros(())
    num = (mag_pred * edge_w).flatten(1).sum(dim=-1)
    den = (mag_ref * edge_w).flatten(1).sum(dim=-1).clamp_min(1e-6)
    retention = num / den
    return (retention - 1.0).abs().mean()


def build_distill_target(
    target: torch.Tensor,
    teacher: torch.Tensor,
    teacher_valid: torch.Tensor,
    edge_map: torch.Tensor,
    teacher_mode: str = "partitioned",
    bm3d_flat_alpha: float = 0.25,
    edge_teacher: torch.Tensor | None = None,
    edge_kd_mix: float = 1.0,
) -> torch.Tensor:
    """Build the secondary distill target from temporal ``target`` and BM3D ``teacher``.

    Modes:
    - ``partitioned``: flats <- BM3D, edges <- temporal (legacy)
    - ``temporal``: always temporal (ignore BM3D)
    - ``blend``: flats <- (1-α)*temporal + α*BM3D, edges <- temporal

    When ``edge_teacher`` is set (e.g. frozen low-σ specialist), edges blend
    toward that prediction: ``mix * edge_teacher + (1-mix) * temporal``.
    """
    use_teacher = teacher_valid.view(-1, 1, 1, 1)
    flat_map = 1.0 - edge_map
    mix = float(np.clip(edge_kd_mix, 0.0, 1.0))
    if edge_teacher is not None and mix > 0.0:
        edge_ref = (1.0 - mix) * target + mix * edge_teacher
    else:
        edge_ref = target
    if teacher_mode == "temporal":
        return edge_ref if edge_teacher is not None else target
    if teacher_mode in ("partitioned", "merge_flat"):
        # merge_flat: caller replaces ``teacher`` with burst-mean; same mask mix.
        blended = edge_map * edge_ref + flat_map * teacher
        return torch.where(use_teacher > 0.5, blended, edge_ref)
    if teacher_mode == "blend":
        alpha = float(np.clip(bm3d_flat_alpha, 0.0, 1.0))
        flat_blend = (1.0 - alpha) * target + alpha * teacher
        blended = edge_map * edge_ref + flat_map * flat_blend
        return torch.where(use_teacher > 0.5, blended, edge_ref)
    # partitioned (fallback)
    blended = edge_map * edge_ref + flat_map * teacher
    return torch.where(use_teacher > 0.5, blended, edge_ref)


def distill_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    teacher_valid: torch.Tensor,
    grad_weight: float,
    ssim_weight: float,
    mean_weight: float,
    teacher_weight: float,
    highpass_weight: float,
    edge_weight: float,
    flat_boost: float = 1.0,
    flat_sigma_weight: float = 0.0,
    teacher_mode: str = "partitioned",
    bm3d_flat_alpha: float = 0.25,
    loss_domain: str = "vst",
    vst_intercept: torch.Tensor | float = PTC_INTERCEPT,
    noisy: torch.Tensor | None = None,
    des_flat_sigma_weight: float = 0.0,
    des_noise_gain_weight: float = 0.0,
    is_hard: torch.Tensor | None = None,
    hard_des_sigma_mult: float = 1.0,
    hard_des_hp_weight: float = 0.0,
    hard_des_use_pure_teacher: bool = True,
    is_edge_focus: torch.Tensor | None = None,
    edge_focus_edge_mult: float = 1.0,
    edge_focus_flat_des_mult: float = 1.0,
    edge_focus_bm3d_alpha: float | None = None,
    distill_edge_harden: float = 0.0,
    ms_edge_weight: float = 0.0,
    ms_edge_scales: int = 3,
    des_edge_fid_weight: float = 0.0,
    edge_focus_des_fid_mult: float = 1.0,
    edge_teacher: torch.Tensor | None = None,
    edge_kd_mix: float = 1.0,
    mixed_grad_xy_weight: float = 0.0,
):
    """Distill loss in normalized-VST or inv-VST DN (scaled to ``[0,1]``)."""
    if noisy is None:
        if edge_teacher is not None:
            pred_d, target_d, teacher_d, edge_teacher_d = maybe_to_loss_domain(
                pred,
                target,
                teacher,
                edge_teacher,
                loss_domain=loss_domain,
                intercept=vst_intercept,
            )
        else:
            pred_d, target_d, teacher_d = maybe_to_loss_domain(
                pred,
                target,
                teacher,
                loss_domain=loss_domain,
                intercept=vst_intercept,
            )
            edge_teacher_d = None
        noisy_d = None
    else:
        if edge_teacher is not None:
            pred_d, target_d, teacher_d, noisy_d, edge_teacher_d = maybe_to_loss_domain(
                pred,
                target,
                teacher,
                noisy,
                edge_teacher,
                loss_domain=loss_domain,
                intercept=vst_intercept,
            )
        else:
            pred_d, target_d, teacher_d, noisy_d = maybe_to_loss_domain(
                pred,
                target,
                teacher,
                noisy,
                loss_domain=loss_domain,
                intercept=vst_intercept,
            )
            edge_teacher_d = None
    _total, pixel, grad, structural, mean = denoise_loss(
        pred_d,
        target_d,
        grad_weight,
        ssim_weight,
        mean_weight,
    )
    use_teacher = teacher_valid.view(-1, 1, 1, 1)
    edge_map = soft_edge_mask(target_d)
    edge_for_distill = harden_edge_map(edge_map, distill_edge_harden)
    flat_map = 1.0 - edge_map
    reliability = mask.clamp(0.0, 1.0)
    edge_frac = (
        is_edge_focus.float().mean().clamp(0.0, 1.0)
        if is_edge_focus is not None
        else pred_d.new_zeros(())
    )
    blend_alpha = float(bm3d_flat_alpha)
    if (
        edge_focus_bm3d_alpha is not None
        and is_edge_focus is not None
        and float(edge_frac) > 0.5
    ):
        # Edge-focus clips: lean distill target toward temporal on structure.
        blend_alpha = float(edge_focus_bm3d_alpha)
    distill_target = build_distill_target(
        target_d,
        teacher_d,
        teacher_valid,
        edge_for_distill,
        teacher_mode=teacher_mode,
        bm3d_flat_alpha=blend_alpha,
        edge_teacher=edge_teacher_d,
        edge_kd_mix=edge_kd_mix,
    )
    mix = float(np.clip(edge_kd_mix, 0.0, 1.0))
    if edge_teacher_d is not None and mix > 0.0:
        edge_structure_ref = (1.0 - mix) * target_d + mix * edge_teacher_d
    else:
        edge_structure_ref = target_d
    # Emphasize flat pixels for highpass / teacher paths.
    if teacher_mode == "temporal":
        flat_emphasis = 1.0 + (max(flat_boost, 1.0) - 1.0) * flat_map
        flat_hp_mask = reliability * flat_emphasis
    else:
        flat_emphasis = 1.0 + (max(flat_boost, 1.0) - 1.0) * flat_map * use_teacher
        flat_hp_mask = (
            reliability * (flat_map * use_teacher + (1.0 - use_teacher)) * flat_emphasis
        )
    teacher_mask = reliability * flat_emphasis
    if float(reliability.mean()) > 1e-4:
        teacher_pixel = masked_charbonnier(pred_d, distill_target, teacher_mask)
        hp = masked_charbonnier(
            highpass(pred_d),
            highpass(distill_target),
            flat_hp_mask,
        )
    else:
        teacher_pixel = charbonnier_loss(pred_d, distill_target)
        hp = charbonnier_loss(highpass(pred_d), highpass(distill_target))
    edge_region = reliability * edge_map
    if float(edge_region.mean()) > 1e-4:
        edge = masked_charbonnier(
            sobel_magnitude(pred_d),
            sobel_magnitude(edge_structure_ref),
            edge_region,
        )
    else:
        edge = charbonnier_loss(
            sobel_magnitude(pred_d),
            sobel_magnitude(edge_structure_ref),
        )
    flat_region = reliability * flat_map
    if flat_sigma_weight > 0.0 and float(flat_region.mean()) > 1e-4:
        # Align flat grain energy to the distill target (avg-pool highpass).
        sigma_pred = masked_mad(highpass(pred_d), flat_region)
        sigma_tgt = masked_mad(highpass(distill_target), flat_region)
        flat_sigma = (sigma_pred - sigma_tgt).abs()
    else:
        flat_sigma = pred_d.new_zeros(())
    hard_scale = pred_d.new_ones(())
    if is_hard is not None and float(hard_des_sigma_mult) > 1.0:
        hard_frac = is_hard.float().mean().clamp(0.0, 1.0)
        hard_scale = 1.0 + (float(hard_des_sigma_mult) - 1.0) * hard_frac
    edge_scale = pred_d.new_ones(())
    flat_des_scale = pred_d.new_ones(())
    if is_edge_focus is not None and float(edge_frac) > 0.0:
        if float(edge_focus_edge_mult) != 1.0:
            edge_scale = 1.0 + (float(edge_focus_edge_mult) - 1.0) * edge_frac
        if float(edge_focus_flat_des_mult) != 1.0:
            flat_des_scale = 1.0 + (float(edge_focus_flat_des_mult) - 1.0) * edge_frac
    des_ref = distill_target
    if (
        hard_des_use_pure_teacher
        and is_hard is not None
        and float(is_hard.float().mean()) > 0.5
        and float(teacher_valid.float().mean()) > 0.5
    ):
        # Hard clips: regress DES flat stats to pure Wiener-BM3D teacher.
        des_ref = teacher_d
    des_flat_sigma = pred_d.new_zeros(())
    des_noise_gain = pred_d.new_zeros(())
    des_hp = pred_d.new_zeros(())
    if (
        (des_flat_sigma_weight > 0.0 or des_noise_gain_weight > 0.0 or hard_des_hp_weight > 0.0)
        and float(flat_region.mean()) > 1e-4
    ):
        # Match eval DES noise σ (Gaussian σ=1.2 highpass + MAD/0.6745).
        des_sigma_pred = des_noise_sigma(pred_d, flat_region)
        des_sigma_tgt = des_noise_sigma(des_ref, flat_region)
        if des_flat_sigma_weight > 0.0:
            des_flat_sigma = (des_sigma_pred - des_sigma_tgt).abs()
        if des_noise_gain_weight > 0.0 and noisy_d is not None:
            des_sigma_in = des_noise_sigma(noisy_d, flat_region).clamp_min(1e-6)
            gain_pred = (des_sigma_pred / des_sigma_in).clamp(0.0, 2.0)
            gain_tgt = (des_sigma_tgt / des_sigma_in).clamp(0.0, 2.0)
            des_noise_gain = (gain_pred - gain_tgt).abs()
        if hard_des_hp_weight > 0.0 and is_hard is not None and float(is_hard.mean()) > 0.0:
            hard_flat = flat_region * is_hard.view(-1, 1, 1, 1)
            if float(hard_flat.mean()) > 1e-4:
                des_hp = masked_charbonnier(
                    des_highpass(pred_d),
                    des_highpass(teacher_d),
                    hard_flat * use_teacher,
                )
    des_edge_fid = pred_d.new_zeros(())
    des_fid_scale = pred_d.new_ones(())
    if des_edge_fid_weight > 0.0:
        if is_edge_focus is not None and float(edge_focus_des_fid_mult) != 1.0:
            des_fid_scale = 1.0 + (float(edge_focus_des_fid_mult) - 1.0) * edge_frac
        # Always vs temporal target (eval reference), never BM3D.
        des_edge_fid = des_edge_fidelity_loss(pred_d, target_d, reliability)
    mixed_xy = pred_d.new_zeros(())
    if mixed_grad_xy_weight > 0.0:
        # Match gx/gy to temporal structure on edges (SBFBurst-style).
        mixed_xy = mixed_grad_xy_loss(pred_d, target_d, reliability * edge_map)
    # Young/FitNet-inspired multi-scale edge feature match (Sobel pyramid).
    ms_edge = pred_d.new_zeros(())
    if ms_edge_weight > 0.0:
        p_ms, t_ms, m_ms = pred_d, edge_structure_ref, reliability
        scales = max(1, int(ms_edge_scales))
        acc = pred_d.new_zeros(())
        for si in range(scales):
            acc = acc + masked_charbonnier(
                sobel_magnitude(p_ms),
                sobel_magnitude(t_ms),
                m_ms,
            )
            if si + 1 < scales and min(p_ms.shape[-2:]) >= 4:
                p_ms = F.avg_pool2d(p_ms, 2)
                t_ms = F.avg_pool2d(t_ms, 2)
                m_ms = F.avg_pool2d(m_ms, 2)
        ms_edge = acc / float(scales)
    total = (
        (1.0 - teacher_weight) * pixel
        + teacher_weight * teacher_pixel
        + grad_weight * grad
        + ssim_weight * structural
        + mean_weight * mean
        + highpass_weight * hp
        + edge_weight * edge_scale * edge
        + flat_sigma_weight * flat_sigma
        + des_flat_sigma_weight * hard_scale * flat_des_scale * des_flat_sigma
        + des_noise_gain_weight * hard_scale * flat_des_scale * des_noise_gain
        + hard_des_hp_weight * flat_des_scale * des_hp
        + des_edge_fid_weight * des_fid_scale * des_edge_fid
        + mixed_grad_xy_weight * edge_scale * mixed_xy
        + ms_edge_weight * edge_scale * ms_edge
    )
    return {
        "total": total,
        "pixel": pixel,
        "teacher": teacher_pixel,
        "grad": grad,
        "ssim_loss": structural,
        "mean": mean,
        "highpass": hp,
        "edge": edge,
        "flat_sigma": flat_sigma,
        "des_flat_sigma": des_flat_sigma,
        "des_noise_gain": des_noise_gain,
        "des_hp": des_hp,
        "des_edge_fid": des_edge_fid,
        "mixed_grad_xy": mixed_xy,
        "ms_edge": ms_edge,
    }


@torch.no_grad()
def validate_real_sequences_with_edges(
    model: torch.nn.Module,
    validation_dir: Path,
    frame_index: int,
    input_frames: int,
    device: torch.device,
    tile: int,
) -> tuple[float, float, float, float, float]:
    """MAE/SSIM/edge metrics plus mean DES vs temporal reference."""
    from .common import (
        decode_mono10,
        memmap_frames,
        parse_exposure_ms,
        parse_geometry,
    )
    from .infer import denoise_frame
    from .validate import (
        aligned_temporal_trimmed_mean,
        central_roi,
        highpass_noise_sigma,
    )

    files = sorted(validation_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise ValueError(f"No validation RAW videos found in {validation_dir}")

    was_training = model.training
    model.eval()
    maes: list[float] = []
    ssims: list[float] = []
    retentions: list[float] = []
    edge_maes: list[float] = []
    des_scores: list[float] = []
    for path in files:
        frame_width, frame_height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, frame_width, frame_height)
        target_index = frames.shape[0] // 2 if frame_index < 0 else frame_index
        if target_index >= frames.shape[0]:
            raise ValueError(
                f"{path.name}: validation frame {target_index} exceeds "
                f"available range 0-{frames.shape[0] - 1}"
            )
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
            model,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=input_frames,
            tile=tile,
            fps=float(fps),
        )
        maes.append(float(np.mean(np.abs(output - temporal))))
        output_tensor = torch.from_numpy(output[None, None]).to(device) / 1023.0
        target_tensor = torch.from_numpy(temporal[None, None]).to(device) / 1023.0
        ssims.append(float(ssim_index(output_tensor, target_tensor)))
        retention, edge_mae = edge_metrics_dn(output, temporal)
        retentions.append(retention)
        edge_maes.append(edge_mae)
        des, _noise_gain, _edge_fid = denoise_edge_score(
            highpass_noise_sigma(output, ys, xs),
            highpass_noise_sigma(input_dn, ys, xs),
            retention,
        )
        des_scores.append(des)

    if was_training:
        model.train()
    return (
        float(np.mean(maes)),
        float(np.mean(ssims)),
        float(np.nanmean(retentions)),
        float(np.nanmean(edge_maes)),
        float(np.mean(des_scores)),
    )


class RealNAFNetDistillDataset(Dataset):
    """Adapt RealBurstDataset bursts into stacked-VST NAFNet inputs."""

    def __init__(self, real: RealBurstDataset, use_sigma: bool = False):
        self.real = real
        self.use_sigma = use_sigma

    def __len__(self) -> int:
        return len(self.real)

    def __getitem__(self, index: int):
        from .common import center_frame_index

        sample = self.real[index]
        burst = sample["burst"]  # [T, 3, H, W] = vst, sigma, exposure
        vst = burst[:, 0]
        exposure = burst[:1, 2]
        if self.use_sigma:
            center = center_frame_index(int(burst.shape[0]))
            sigma = burst[center : center + 1, 1]
            inputs = torch.cat([vst, sigma, exposure], dim=0)
        else:
            inputs = torch.cat([vst, exposure], dim=0)
        return {
            "input": inputs,
            "target": sample["target"],
            "teacher": sample["teacher"],
            "mask": sample["mask"],
            "teacher_valid": sample["teacher_valid"],
            "loss_weight": sample.get(
                "loss_weight",
                torch.tensor(1.0, dtype=torch.float32),
            ),
            "is_hard": sample.get(
                "is_hard",
                torch.tensor(0.0, dtype=torch.float32),
            ),
            "is_edge_focus": sample.get(
                "is_edge_focus",
                torch.tensor(0.0, dtype=torch.float32),
            ),
        }


class MixedDistillDataset(Dataset):
    def __init__(
        self,
        synthetic: PixelShiftPatchDataset | None,
        real: RealNAFNetDistillDataset,
        samples_per_epoch: int,
        real_fraction: float,
        seed: int = 0,
    ):
        self.synthetic = synthetic
        self.real = real
        self.samples_per_epoch = samples_per_epoch
        self.real_fraction = 1.0 if synthetic is None else float(np.clip(real_fraction, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        if self.synthetic is None or self.rng.random() < self.real_fraction:
            return self.real[int(self.rng.integers(len(self.real)))]
        inputs, target = self.synthetic[int(self.rng.integers(len(self.synthetic)))]
        ones = torch.ones_like(target)
        return {
            "input": inputs,
            "target": target,
            "teacher": target,
            "mask": ones,
            "teacher_valid": torch.tensor(0.0, dtype=torch.float32),
            "loss_weight": torch.tensor(1.0, dtype=torch.float32),
            "is_hard": torch.tensor(0.0, dtype=torch.float32),
            "is_edge_focus": torch.tensor(0.0, dtype=torch.float32),
        }


def collate_distill(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input": torch.stack([item["input"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "teacher": torch.stack([item["teacher"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "teacher_valid": torch.stack([item["teacher_valid"] for item in batch], dim=0),
        "loss_weight": torch.stack(
            [
                item.get("loss_weight", torch.tensor(1.0, dtype=torch.float32))
                for item in batch
            ],
            dim=0,
        ),
        "is_hard": torch.stack(
            [
                item.get("is_hard", torch.tensor(0.0, dtype=torch.float32))
                for item in batch
            ],
            dim=0,
        ),
        "is_edge_focus": torch.stack(
            [
                item.get("is_edge_focus", torch.tensor(0.0, dtype=torch.float32))
                for item in batch
            ],
            dim=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-manifest",
        type=Path,
        default=None,
        help="PixelShift manifest; omit for real-only fine-tuning.",
    )
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_bm3d_distill"),
    )
    parser.add_argument(
        "--arch",
        choices=("nafnet", "stacked_restormer", "dual_head"),
        default="nafnet",
    )
    parser.add_argument(
        "--use-nonlocal",
        action="store_true",
        help="Insert zero-init NonLocal block(s) after the NAFNet bottleneck.",
    )
    parser.add_argument(
        "--nonlocal-count",
        type=int,
        default=1,
        help="Number of bottleneck NonLocal blocks when --use-nonlocal.",
    )
    parser.add_argument(
        "--learnable-mask",
        action="store_true",
        help="Dual-head: learn residual edge mask on top of Sobel (zero-init).",
    )
    parser.add_argument(
        "--region-towers",
        type=int,
        default=0,
        help="Dual-head: NAFBlocks per flat/edge tower after shared trunk (0=off).",
    )
    parser.add_argument(
        "--fusion-edge-harden",
        type=float,
        default=0.0,
        help="Dual-head: sharpen fusion edge_map via sigmoid(h*logit); 0=off, ~8–16 typical.",
    )
    parser.add_argument(
        "--init-edge-from",
        type=Path,
        default=None,
        help="Copy ending_edge (+tower_edge) weights from a DualHead checkpoint.",
    )
    parser.add_argument(
        "--use-deformable",
        action="store_true",
        help="Dual-head: zero-init residual deformable align of neighbor VST frames.",
    )
    parser.add_argument(
        "--deform-feat-width",
        type=int,
        default=16,
        help="Hidden width of the deformable align pair-net.",
    )
    parser.add_argument(
        "--deform-max-flow",
        type=float,
        default=2.0,
        help="Max residual flow in pixels for deformable align.",
    )
    parser.add_argument(
        "--use-burst-merge",
        action="store_true",
        help="Dual-head: learnable soft burst merge (zero-init → center), then spatial.",
    )
    parser.add_argument(
        "--merge-feat-width",
        type=int,
        default=16,
        help="Hidden width of SoftBurstMerge gate net.",
    )
    parser.add_argument(
        "--dual-aux-weight",
        type=float,
        default=0.25,
        help="Aux loss weight for flat/edge heads before blending (dual_head only).",
    )
    parser.add_argument("--input-frames", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--num-blocks", type=str, default="2,3,3,4")
    parser.add_argument("--num-refinement-blocks", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    parser.add_argument("--real-fraction", type=float, default=0.65)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-weight", type=float, default=0.20)
    parser.add_argument("--ssim-weight", type=float, default=0.15)
    parser.add_argument("--mean-weight", type=float, default=0.10)
    parser.add_argument("--teacher-weight-start", type=float, default=0.45)
    parser.add_argument("--teacher-weight-end", type=float, default=0.20)
    parser.add_argument("--highpass-weight", type=float, default=0.25)
    parser.add_argument(
        "--input-mask-ratio",
        type=float,
        default=0.0,
        help="AIM-style random 16x16 patch mask ratio on image channels (0=off).",
    )
    parser.add_argument("--input-mask-patch", type=int, default=16)
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.20,
        help="Sobel consistency vs non-BM3D target on edge regions.",
    )
    parser.add_argument(
        "--flat-boost",
        type=float,
        default=1.0,
        help="Extra weight on flat-region teacher/highpass supervision (>=1).",
    )
    parser.add_argument(
        "--des-flat-sigma-weight",
        type=float,
        default=0.0,
        help="Match DES Gaussian-MAD flat σ (pred vs distill target).",
    )
    parser.add_argument(
        "--des-noise-gain-weight",
        type=float,
        default=0.0,
        help="Match DES noise ratio σ_out/σ_in to teacher on flats.",
    )
    parser.add_argument(
        "--des-edge-fid-weight",
        type=float,
        default=0.0,
        help="DES edge_fidelity proxy: |Sobel retention - 1| vs temporal target.",
    )
    parser.add_argument(
        "--distill-edge-harden",
        type=float,
        default=0.0,
        help="Sharpen edge_map for distill partition (>0 reduces BM3D leak into edges).",
    )
    parser.add_argument(
        "--use-sigma",
        action="store_true",
        help="Append PTC sigma-map channel after VST frames (before exposure).",
    )
    parser.add_argument(
        "--measured-fe-sigma",
        action="store_true",
        help=(
            "With --use-sigma + Wiener FE: replace affine/PTC σ with measured "
            "flat highpass MAD on the merged frame (FFDNet-style)."
        ),
    )
    parser.add_argument(
        "--use-sigma-film",
        action="store_true",
        help="DualHead: FiLM residual features by mean σ channel (KPN/FFDNet).",
    )
    parser.add_argument(
        "--use-sigma-film-flat",
        action="store_true",
        help="P2-b: FiLM only the flat tower (leave edge path unmodulated).",
    )
    parser.add_argument(
        "--use-lap-edge",
        action="store_true",
        help="P2-c: add zero-init Laplacian→edge feature branch (EIID-style).",
    )
    parser.add_argument(
        "--use-lap-edge-ms",
        action="store_true",
        help="P3-a: also add half-res Laplacian→edge branch (requires --use-lap-edge).",
    )
    parser.add_argument(
        "--mixed-grad-xy-weight",
        type=float,
        default=0.0,
        help="SBFBurst-style gx/gy match on edges vs temporal.",
    )
    parser.add_argument(
        "--ms-edge-weight",
        type=float,
        default=0.0,
        help="Young/FitNet-style multi-scale Sobel feature match vs edge structure ref.",
    )
    parser.add_argument(
        "--ms-edge-scales",
        type=int,
        default=3,
        help="Number of pyramid scales for --ms-edge-weight.",
    )
    parser.add_argument(
        "--sigma-film-jitter",
        type=float,
        default=0.0,
        help="Train-time multiplicative σ jitter for FiLM/σ-channel (e.g. 0.25).",
    )
    parser.add_argument(
        "--freeze-flat-head",
        action="store_true",
        help="Freeze DualHead ending_flat / tower_flat (protect flats while FT edge).",
    )
    parser.add_argument(
        "--flat-sigma-weight",
        type=float,
        default=0.0,
        help="Match flat-region highpass MAD to the distill target.",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=("partitioned", "temporal", "blend", "merge_flat"),
        default="partitioned",
        help="How the flat distill target is built (merge_flat = mean of input burst).",
    )
    parser.add_argument(
        "--bm3d-flat-alpha",
        type=float,
        default=0.25,
        help="BM3D weight on flats when --teacher-mode blend.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=32,
        help="Leave-burst-out temporal trimmed-mean window for real targets.",
    )
    parser.add_argument(
        "--align-soft-gate",
        action="store_true",
        help="Train with residual ECC + soft neighbor→center gating (match gated infer).",
    )
    parser.add_argument("--align-threshold", type=float, default=0.03)
    parser.add_argument("--align-temperature", type=float, default=0.015)
    parser.add_argument(
        "--align-drop-prob",
        type=float,
        default=0.0,
        help="Extra probability of forcing a companion frame to the center.",
    )
    parser.add_argument(
        "--wiener-front-end",
        action="store_true",
        help="Frozen HDR+ Wiener merge before DualHead (train/infer match ablation).",
    )
    parser.add_argument(
        "--wiener-merge-frames",
        type=int,
        default=8,
        help="Frames gathered for Wiener merge when --wiener-front-end (model still uses --input-frames).",
    )
    parser.add_argument("--wiener-tile", type=int, default=32)
    parser.add_argument("--wiener-overlap", type=int, default=16)
    parser.add_argument("--wiener-c-factor", type=float, default=8.0)
    parser.add_argument(
        "--wiener-spatial",
        action="store_true",
        help="After temporal Wiener, apply HDR+ 2D spatial Wiener shrinkage.",
    )
    parser.add_argument(
        "--wiener-spatial-c-factor",
        type=float,
        default=None,
        help="Spatial Wiener c factor (default: same as --wiener-c-factor).",
    )
    parser.add_argument(
        "--wiener-spatial-adaptive",
        action="store_true",
        help="Adaptive spatial Wiener: stronger c on flats, extra boost on dark frames.",
    )
    parser.add_argument(
        "--wiener-spatial-flat-c-mult",
        type=float,
        default=2.5,
        help="Multiply spatial c on flat regions when --wiener-spatial-adaptive.",
    )
    parser.add_argument(
        "--wiener-spatial-dark-boost",
        type=float,
        default=0.5,
        help="Extra spatial-c scale for dark frames when --wiener-spatial-adaptive.",
    )
    parser.add_argument(
        "--wiener-spatial-edge-c-mult",
        type=float,
        default=1.0,
        help="Spatial-c scale on edges when adaptive (use <1 to protect structure).",
    )
    parser.add_argument(
        "--wiener-spatial-mask-harden",
        type=float,
        default=0.0,
        help="Sharpen flat/edge blend mask (0=off, ~8–16 typical).",
    )
    parser.add_argument(
        "--wiener-spatial-freq-gamma",
        type=float,
        default=0.0,
        help="HDR+/IPOL high-frequency noise shaping γ on flat spatial pass.",
    )
    parser.add_argument(
        "--wiener-fe-schedule",
        type=str,
        default=None,
        help="fps_sigma|gated: per-clip gated spatial FE (overrides fixed adaptive flags).",
    )
    parser.add_argument(
        "--train-edge-only",
        action="store_true",
        help="Freeze all params except ending_edge / tower_edge / mask_head.",
    )
    parser.add_argument(
        "--dark-boost",
        type=float,
        default=0.0,
        help="Real-clip sample weight ~ fps^(-dark_boost); >0 favors low-fps/dark DES-gap scenes.",
    )
    parser.add_argument(
        "--dark-target-mean-max",
        type=float,
        default=None,
        help="Cap synthetic target-mean DN (bias PixelShift toward darker patches).",
    )
    parser.add_argument(
        "--postmerge-noise-calib",
        type=Path,
        default=None,
        help="JSON from calibrate_postmerge_noise; residual σ channel + synth residual scale.",
    )
    parser.add_argument(
        "--no-scale-synth-residual",
        action="store_true",
        help="Keep synthetic Wiener residual as-is; only change σ channel via calib.",
    )
    parser.add_argument(
        "--wiener-bm3d-teacher",
        action="store_true",
        help="Run VST-BM3D on the Wiener-merged patch as the flat distill teacher.",
    )
    parser.add_argument(
        "--wiener-bm3d-sigma",
        type=float,
        default=0.35,
        help="BM3D sigma_psd in VST domain for post-Wiener residual (<<1 for single-frame).",
    )
    parser.add_argument(
        "--wiener-bm3d-prob",
        type=float,
        default=1.0,
        help="Probability of online Wiener-BM3D teacher vs offline RAW BM3D cache.",
    )
    parser.add_argument(
        "--scene-loss-power",
        type=float,
        default=0.0,
        help="Per-sample loss weight (ref_fps/fps)^power, clipped to [--scene-loss-max].",
    )
    parser.add_argument(
        "--scene-loss-ref-fps",
        type=float,
        default=10.0,
        help="Reference fps for scene loss weighting.",
    )
    parser.add_argument(
        "--scene-loss-max",
        type=float,
        default=4.0,
        help="Cap on per-sample scene loss weight.",
    )
    parser.add_argument(
        "--hard-scene-substr",
        type=str,
        default="",
        help="Comma-separated name/path substrings for hard-scene oversample/loss boost.",
    )
    parser.add_argument(
        "--hard-scene-sample-mult",
        type=float,
        default=1.0,
        help="Multiply sampling probability for hard-scene clips.",
    )
    parser.add_argument(
        "--hard-scene-loss-mult",
        type=float,
        default=1.0,
        help="Multiply per-sample loss weight for hard-scene clips.",
    )
    parser.add_argument(
        "--hard-wiener-bm3d-sigma",
        type=float,
        default=None,
        help="Override Wiener BM3D sigma_psd on hard scenes only.",
    )
    parser.add_argument(
        "--hard-wiener-bm3d-sigma-map",
        type=Path,
        default=None,
        help="JSON map {clip_token: sigma_psd} for per-clip Wiener BM3D teacher.",
    )
    parser.add_argument(
        "--hard-des-sigma-mult",
        type=float,
        default=1.0,
        help="Multiply DES flat-σ / noise-gain loss weights on hard samples.",
    )
    parser.add_argument(
        "--hard-des-hp-weight",
        type=float,
        default=0.0,
        help="Hard-only Charbonnier on DES highpass vs pure BM3D teacher flats.",
    )
    parser.add_argument(
        "--hard-des-use-pure-teacher",
        action="store_true",
        help="On hard samples, match DES σ/gain to pure BM3D teacher (not blend).",
    )
    parser.add_argument(
        "--edge-focus-substr",
        type=str,
        default="",
        help="Comma-separated clips that need higher edge / lower flat-DES pressure.",
    )
    parser.add_argument(
        "--edge-focus-edge-mult",
        type=float,
        default=1.0,
        help="Multiply edge loss weight on edge-focus samples.",
    )
    parser.add_argument(
        "--edge-focus-flat-des-mult",
        type=float,
        default=1.0,
        help="Multiply DES flat-σ / gain / hard highpass weights on edge-focus samples.",
    )
    parser.add_argument(
        "--edge-focus-des-fid-mult",
        type=float,
        default=1.0,
        help="Multiply DES edge_fidelity loss on edge-focus samples.",
    )
    parser.add_argument(
        "--edge-focus-bm3d-alpha",
        type=float,
        default=None,
        help="Override bm3d_flat_alpha on edge-focus batches (lower → more temporal).",
    )
    parser.add_argument(
        "--freeze-edge-head",
        action="store_true",
        help="Freeze DualHead ending_edge conv (keep edge residual fixed).",
    )
    parser.add_argument(
        "--edge-kd-checkpoint",
        type=Path,
        default=None,
        help=(
            "Frozen DualHead (e.g. low-σ specialist) whose prediction replaces "
            "the temporal target on edges in partitioned distill (P0-b)."
        ),
    )
    parser.add_argument(
        "--edge-kd-mix",
        type=float,
        default=1.0,
        help="Blend weight for edge KD teacher vs temporal on edges (1=full KD).",
    )
    parser.add_argument(
        "--blend-kd-sota",
        type=Path,
        default=None,
        help="P2-a: frozen SOTA DualHead arm for edge-soft blend teacher.",
    )
    parser.add_argument(
        "--blend-kd-edgekd",
        type=Path,
        default=None,
        help="P2-a: frozen edgeKD DualHead arm for edge-soft blend teacher.",
    )
    parser.add_argument(
        "--blend-kd-mix",
        type=float,
        default=1.0,
        help=(
            "Mix blend_edge_soft teacher into temporal/BM3D targets "
            "(1=fully distill to blend)."
        ),
    )
    parser.add_argument(
        "--blend-kd-temperature",
        type=float,
        default=8.0,
        help="Sobel soft-edge temperature for blend KD teacher (match infer).",
    )
    parser.add_argument(
        "--blend-kd-harden",
        type=float,
        default=0.0,
        help="P4-a: sharpen blend KD edge map (deploy default 16).",
    )
    parser.add_argument(
        "--blend-kd-edge-gain",
        type=float,
        default=1.0,
        help="Optional gain on blend KD edge map after harden.",
    )
    parser.add_argument(
        "--loss-domain",
        choices=("vst", "dn"),
        default="vst",
        help="Compute distill loss in normalized VST or inv-VST DN (scaled to [0,1]).",
    )
    parser.add_argument(
        "--init-restormer",
        type=Path,
        default=None,
        help="Partial warm-start for --arch stacked_restormer.",
    )
    parser.add_argument("--noise-jitter", type=float, default=0.15)
    parser.add_argument("--exposure-ms-min", type=float, default=10.0)
    parser.add_argument("--exposure-ms-max", type=float, default=2000.0)
    parser.add_argument("--black-level-dn", type=float, default=60.0)
    parser.add_argument("--dark-variance-per-s-min", type=float, default=0.0)
    parser.add_argument("--dark-variance-per-s-max", type=float, default=6.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--fresh-resume", action="store_true")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(r"D:\denoise\素材\降噪素材"),
    )
    parser.add_argument("--validation-every", type=int, default=5)
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
    postmerge_calib = None
    if args.postmerge_noise_calib is not None:
        from .postmerge_noise import load_postmerge_calib

        postmerge_calib = load_postmerge_calib(args.postmerge_noise_calib)
        # Persist a portable relative/absolute string in checkpoints.
        args.postmerge_noise_calib = str(args.postmerge_noise_calib)
        print(
            f"postmerge_calib alpha={postmerge_calib.alpha:.4f} "
            f"beta={postmerge_calib.beta_dn:.4f} n={postmerge_calib.n_samples} "
            f"median_ratio={postmerge_calib.median_ratio:.4f}",
            flush=True,
        )
    synthetic = None
    if args.synthetic_manifest is not None:
        synthetic = PixelShiftPatchDataset(
            args.synthetic_manifest,
            patch_size=args.patch_size,
            patches_per_epoch=args.patches_per_epoch,
            input_frames=args.input_frames,
            noise_jitter=args.noise_jitter,
            exposure_ms_range=(args.exposure_ms_min, args.exposure_ms_max),
            black_level_dn=args.black_level_dn,
            dark_variance_per_s_range=(
                args.dark_variance_per_s_min,
                args.dark_variance_per_s_max,
            ),
            cache_images_in_ram=True,
            use_sigma=args.use_sigma,
            wiener_front_end=args.wiener_front_end,
            wiener_merge_frames=args.wiener_merge_frames,
            wiener_tile=args.wiener_tile,
            wiener_overlap=args.wiener_overlap,
            wiener_c_factor=args.wiener_c_factor,
            wiener_spatial=args.wiener_spatial,
            wiener_spatial_c_factor=args.wiener_spatial_c_factor,
            dark_target_mean_max=args.dark_target_mean_max,
            postmerge_calib=postmerge_calib,
            scale_synth_residual=not args.no_scale_synth_residual,
        )
    real = RealNAFNetDistillDataset(
        RealBurstDataset(
            args.real_manifest,
            patch_size=args.patch_size,
            input_frames=args.input_frames,
            samples_per_epoch=args.patches_per_epoch,
            center_mode="random",
            seed=1,
            target_frames=args.target_frames,
            align_soft_gate=args.align_soft_gate,
            align_threshold=args.align_threshold,
            align_temperature=args.align_temperature,
            align_drop_prob=args.align_drop_prob,
            wiener_front_end=args.wiener_front_end,
            wiener_merge_frames=args.wiener_merge_frames,
            wiener_tile=args.wiener_tile,
            wiener_overlap=args.wiener_overlap,
            wiener_c_factor=args.wiener_c_factor,
            wiener_spatial=args.wiener_spatial,
            wiener_spatial_c_factor=args.wiener_spatial_c_factor,
            wiener_spatial_adaptive=args.wiener_spatial_adaptive,
            wiener_spatial_flat_c_mult=args.wiener_spatial_flat_c_mult,
            wiener_spatial_dark_boost=args.wiener_spatial_dark_boost,
            wiener_spatial_edge_c_mult=args.wiener_spatial_edge_c_mult,
            wiener_spatial_mask_harden=args.wiener_spatial_mask_harden,
            wiener_spatial_freq_gamma=args.wiener_spatial_freq_gamma,
            wiener_fe_schedule=args.wiener_fe_schedule,
            dark_boost=args.dark_boost,
            postmerge_calib=postmerge_calib,
            measured_fe_sigma=args.measured_fe_sigma,
            wiener_bm3d_teacher=args.wiener_bm3d_teacher,
            wiener_bm3d_sigma=args.wiener_bm3d_sigma,
            wiener_bm3d_prob=args.wiener_bm3d_prob,
            scene_loss_power=args.scene_loss_power,
            scene_loss_ref_fps=args.scene_loss_ref_fps,
            scene_loss_max=args.scene_loss_max,
            hard_scene_substr=tuple(
                part.strip()
                for part in str(args.hard_scene_substr).split(",")
                if part.strip()
            ),
            hard_scene_sample_mult=args.hard_scene_sample_mult,
            hard_scene_loss_mult=args.hard_scene_loss_mult,
            hard_wiener_bm3d_sigma=args.hard_wiener_bm3d_sigma,
            hard_wiener_bm3d_sigma_map=(
                json.loads(args.hard_wiener_bm3d_sigma_map.read_text(encoding="utf-8"))
                if args.hard_wiener_bm3d_sigma_map is not None
                else None
            ),
            edge_focus_substr=tuple(
                part.strip()
                for part in str(args.edge_focus_substr).split(",")
                if part.strip()
            ),
        ),
        use_sigma=args.use_sigma,
    )
    dataset = MixedDistillDataset(
        synthetic,
        real,
        samples_per_epoch=args.patches_per_epoch,
        real_fraction=args.real_fraction,
        seed=2,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_distill,
    )

    num_blocks = tuple(
        int(part.strip()) for part in args.num_blocks.split(",") if part.strip()
    )
    if args.arch == "stacked_restormer":
        model = build_stacked_restormer(
            input_frames=args.input_frames,
            use_sigma=args.use_sigma,
            dim=args.dim,
            num_blocks=num_blocks,
            num_refinement_blocks=args.num_refinement_blocks,
            use_checkpoint=True,
        ).to(device)
    elif args.arch == "dual_head":
        model = build_dual_head_nafnet(
            width=args.width,
            input_frames=args.input_frames,
            use_sigma=args.use_sigma,
            use_nonlocal=args.use_nonlocal,
            nonlocal_count=args.nonlocal_count,
            learnable_mask=args.learnable_mask,
            use_deformable=args.use_deformable,
            deform_feat_width=args.deform_feat_width,
            deform_max_flow=args.deform_max_flow,
            use_burst_merge=args.use_burst_merge,
            merge_feat_width=args.merge_feat_width,
            region_towers=args.region_towers,
            fusion_edge_harden=args.fusion_edge_harden,
            use_sigma_film=args.use_sigma_film,
            use_sigma_film_flat=args.use_sigma_film_flat,
            use_lap_edge=args.use_lap_edge,
            use_lap_edge_ms=args.use_lap_edge_ms,
        ).to(device)
    else:
        model = build_nafnet(
            width=args.width,
            input_frames=args.input_frames,
            use_sigma=args.use_sigma,
            use_nonlocal=args.use_nonlocal,
            nonlocal_count=args.nonlocal_count,
        ).to(device)
    model.use_sigma = args.use_sigma
    model.wiener_front_end = bool(args.wiener_front_end)
    model.wiener_merge_frames = int(args.wiener_merge_frames)
    model.wiener_tile = int(args.wiener_tile)
    model.wiener_overlap = int(args.wiener_overlap)
    model.wiener_c_factor = float(args.wiener_c_factor)
    model.wiener_spatial = bool(args.wiener_spatial)
    model.wiener_spatial_c_factor = args.wiener_spatial_c_factor
    model.wiener_spatial_adaptive = bool(args.wiener_spatial_adaptive)
    model.wiener_spatial_flat_c_mult = float(args.wiener_spatial_flat_c_mult)
    model.wiener_spatial_dark_boost = float(args.wiener_spatial_dark_boost)
    model.wiener_spatial_edge_c_mult = float(args.wiener_spatial_edge_c_mult)
    model.wiener_spatial_mask_harden = float(args.wiener_spatial_mask_harden)
    model.wiener_spatial_freq_gamma = float(args.wiener_spatial_freq_gamma)
    model.wiener_fe_schedule = args.wiener_fe_schedule
    model.postmerge_calib = postmerge_calib
    model.measured_fe_sigma = bool(args.measured_fe_sigma)
    model.use_sigma_film = bool(args.use_sigma_film)
    model.use_sigma_film_flat = bool(args.use_sigma_film_flat)
    model.use_lap_edge = bool(args.use_lap_edge)
    model.use_lap_edge_ms = bool(args.use_lap_edge_ms)
    from .infer import load_model as load_infer_model

    def _load_frozen_teacher(path: Path, label: str):
        if not path.exists():
            raise SystemExit(f"Missing {label} checkpoint: {path}")
        teacher_model, _n, _w = load_infer_model(path, None, device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        # Avoid re-running Wiener FE: training patches are already post-merge.
        teacher_model.wiener_front_end = False
        return teacher_model

    edge_kd_model = None
    if args.edge_kd_checkpoint is not None:
        edge_kd_model = _load_frozen_teacher(args.edge_kd_checkpoint, "edge_kd")
        args.edge_kd_checkpoint = str(args.edge_kd_checkpoint)
        print(
            f"edge_kd teacher={args.edge_kd_checkpoint} mix={args.edge_kd_mix:.2f}",
            flush=True,
        )

    blend_sota_model = None
    blend_edgekd_model = None
    if args.blend_kd_sota is not None or args.blend_kd_edgekd is not None:
        if args.blend_kd_sota is None or args.blend_kd_edgekd is None:
            raise SystemExit("--blend-kd-sota and --blend-kd-edgekd must be set together")
        blend_sota_model = _load_frozen_teacher(args.blend_kd_sota, "blend_kd_sota")
        blend_edgekd_model = _load_frozen_teacher(args.blend_kd_edgekd, "blend_kd_edgekd")
        args.blend_kd_sota = str(args.blend_kd_sota)
        args.blend_kd_edgekd = str(args.blend_kd_edgekd)
        print(
            f"blend_kd teachers sota={args.blend_kd_sota} "
            f"edgekd={args.blend_kd_edgekd} mix={args.blend_kd_mix:.2f} "
            f"T={args.blend_kd_temperature:.1f}",
            flush=True,
        )
    optimizer, optimizer_name = build_optimizer(model, args.lr)
    print(f"device={device} optimizer={optimizer_name} arch={args.arch}", flush=True)
    print(
        f"input_frames={args.input_frames} use_sigma={args.use_sigma} "
        f"measured_fe_sigma={args.measured_fe_sigma} "
        f"use_sigma_film={args.use_sigma_film} "
        f"use_sigma_film_flat={args.use_sigma_film_flat} "
        f"use_lap_edge={args.use_lap_edge} "
        f"use_lap_edge_ms={args.use_lap_edge_ms} "
        f"mixed_grad_xy_weight={args.mixed_grad_xy_weight} "
        f"sigma_film_jitter={args.sigma_film_jitter} "
        f"use_nonlocal={args.use_nonlocal} nonlocal_count={args.nonlocal_count} "
        f"learnable_mask={args.learnable_mask} "
        f"region_towers={args.region_towers} "
        f"fusion_edge_harden={args.fusion_edge_harden} "
        f"use_deformable={args.use_deformable} "
        f"use_burst_merge={args.use_burst_merge} "
        f"width={args.width} flat_boost={args.flat_boost} "
        f"flat_sigma_weight={args.flat_sigma_weight} "
        f"des_flat_sigma_weight={args.des_flat_sigma_weight} "
        f"des_noise_gain_weight={args.des_noise_gain_weight} "
        f"teacher_mode={args.teacher_mode} bm3d_flat_alpha={args.bm3d_flat_alpha} "
        f"target_frames={args.target_frames} "
        f"align_soft_gate={args.align_soft_gate} align_drop_prob={args.align_drop_prob} "
        f"wiener_front_end={args.wiener_front_end} "
        f"wiener_merge_frames={args.wiener_merge_frames} "
        f"wiener_spatial={args.wiener_spatial} "
        f"wiener_bm3d_teacher={args.wiener_bm3d_teacher} "
        f"wiener_bm3d_sigma={args.wiener_bm3d_sigma} "
        f"scene_loss_power={args.scene_loss_power} "
        f"hard_scene_substr={args.hard_scene_substr!r} "
        f"hard_scene_sample_mult={args.hard_scene_sample_mult} "
        f"hard_scene_loss_mult={args.hard_scene_loss_mult} "
        f"hard_wiener_bm3d_sigma={args.hard_wiener_bm3d_sigma} "
        f"hard_wiener_bm3d_sigma_map={args.hard_wiener_bm3d_sigma_map} "
        f"dark_boost={args.dark_boost} "
        f"loss_domain={args.loss_domain} "
        f"real_fraction={args.real_fraction}",
        flush=True,
    )

    start_epoch = 0
    global_step = 0
    best_mae = float("inf")
    best_ssim = float("-inf")
    best_composite = float("inf")
    best_des = float("-inf")
    if args.arch == "stacked_restormer":
        if args.resume is not None:
            state = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            if args.fresh_resume:
                print(
                    f"Loaded stacked Restormer from {args.resume}; restarting",
                    flush=True,
                )
            else:
                if "optimizer" in state:
                    optimizer.load_state_dict(state["optimizer"])
                start_epoch = int(state.get("epoch", 0))
                global_step = int(state.get("step", 0))
                print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)
        elif args.init_restormer is not None:
            loaded, skipped = load_stacked_restormer_partial(
                model,
                args.init_restormer,
                device,
            )
            print(
                f"Partial Restormer init from {args.init_restormer}: "
                f"loaded={loaded} skipped={skipped}",
                flush=True,
            )
        else:
            print("Training stacked Restormer from scratch", flush=True)
    elif args.arch == "dual_head":
        if args.resume is None:
            raise SystemExit("--resume is required for --arch dual_head")
        state = torch.load(args.resume, map_location=device, weights_only=False)
        weights = state["model"]
        saved_args = state.get("args") or {}
        old_frames = int(saved_args.get("input_frames") or 0)
        if old_frames <= 0 and "intro.weight" in weights:
            intro_c = int(weights["intro.weight"].shape[1])
            extras = 2 if args.use_sigma else 1
            old_frames = max(intro_c - extras, 1)
        if old_frames > 0 and old_frames != args.input_frames:
            weights = expand_intro_state_for_input_frames(
                weights,
                new_frames=args.input_frames,
                use_sigma=args.use_sigma,
                use_exposure=True,
            )
            print(
                f"Expanded intro {old_frames}f -> {args.input_frames}f",
                flush=True,
            )
        elif args.use_sigma:
            weights = expand_intro_state_for_sigma(weights, args.input_frames)
        if "ending_flat.weight" in weights:
            missing, unexpected = model.load_state_dict(weights, strict=False)
            print(f"Loaded dual-head weights from {args.resume}", flush=True)
        else:
            loaded, skipped = load_dual_head_from_nafnet_state(model, weights)
            missing, unexpected = [], []
            print(
                f"Warm-started dual-head from single-head {args.resume}: "
                f"loaded={loaded} skipped={skipped}",
                flush=True,
            )
        if (
            args.use_nonlocal
            or args.learnable_mask
            or args.use_deformable
            or args.use_burst_merge
            or args.region_towers > 0
            or args.use_sigma_film
            or args.use_sigma_film_flat
            or args.use_lap_edge
            or args.use_lap_edge_ms
            or (old_frames > 0 and old_frames != args.input_frames)
        ):
            print(
                f"Arch expand warm-start strict=False "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        is_dual_ckpt = "ending_flat.weight" in state["model"]
        # Only force epoch restart on true warm-starts / arch expands, not when
        # continuing the same dual-head+nonlocal recipe without --fresh-resume.
        ckpt_keys = state["model"].keys()
        needs_arch_expand = (
            (
                args.use_nonlocal
                and not any("bottleneck_nl" in key for key in ckpt_keys)
            )
            or (
                args.learnable_mask
                and not any("mask_head" in key for key in ckpt_keys)
            )
            or (
                args.use_deformable
                and not any("deform" in key for key in ckpt_keys)
            )
            or (
                args.use_burst_merge
                and not any("burst_merge" in key or "soft_merge" in key for key in ckpt_keys)
            )
            or (
                args.region_towers > 0
                and not any(key.startswith("tower_flat.") for key in ckpt_keys)
            )
            or (
                args.use_sigma_film
                and not any(
                    key.startswith("sigma_film.")
                    and not key.startswith("sigma_film_flat.")
                    for key in ckpt_keys
                )
            )
            or (
                args.use_sigma_film_flat
                and not any(key.startswith("sigma_film_flat.") for key in ckpt_keys)
            )
            or (
                args.use_lap_edge
                and not any(
                    key.startswith("lap_to_edge.")
                    and not key.startswith("lap_to_edge_ms.")
                    for key in ckpt_keys
                )
            )
            or (
                args.use_lap_edge_ms
                and not any(key.startswith("lap_to_edge_ms.") for key in ckpt_keys)
            )
            or (old_frames > 0 and old_frames != args.input_frames)
        )
        if args.fresh_resume or not is_dual_ckpt or needs_arch_expand:
            print("Restarting metrics/epoch for dual-head training", flush=True)
        else:
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state.get("epoch", 0))
            global_step = int(state.get("step", 0))
            best_mae = float(state.get("validation_mae_dn", best_mae))
            best_ssim = float(state.get("validation_ssim", best_ssim))
            print(f"Resumed dual-head from {args.resume} at epoch {start_epoch}", flush=True)
        if args.init_edge_from is not None:
            from .dual_head_nafnet import copy_edge_head_weights

            if not args.init_edge_from.exists():
                raise SystemExit(f"Missing --init-edge-from: {args.init_edge_from}")
            edge_payload = torch.load(
                args.init_edge_from, map_location=device, weights_only=False
            )
            edge_state = (
                edge_payload["model"]
                if isinstance(edge_payload, dict) and "model" in edge_payload
                else edge_payload
            )
            n_copied = copy_edge_head_weights(model, edge_state)
            args.init_edge_from = str(args.init_edge_from)
            print(
                f"Copied {n_copied} edge-head tensors from {args.init_edge_from}",
                flush=True,
            )
    else:
        if args.resume is None:
            raise SystemExit("--resume is required for --arch nafnet")
        state = torch.load(args.resume, map_location=device, weights_only=False)
        weights = state["model"]
        if args.use_sigma:
            weights = expand_intro_state_for_sigma(weights, args.input_frames)
        # Warm-start a wider network by zero-padding channel dims.
        resume_width = int((state.get("args") or {}).get("width", args.width))
        if args.width != resume_width:
            weights = expand_state_dict_width(weights, model)
            print(
                f"Expanded NAFNet width {resume_width} -> {args.width}",
                flush=True,
            )
            model.load_state_dict(weights, strict=False)
        else:
            missing, unexpected = model.load_state_dict(weights, strict=False)
            if args.use_nonlocal:
                print(
                    f"NonLocal warm-start strict=False "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
            else:
                # Re-load strict if no architecture change to catch mismatches.
                model.load_state_dict(weights, strict=True)
        if args.fresh_resume or args.width != resume_width or args.use_nonlocal:
            print(f"Loaded weights from {args.resume}; restarting metrics/epoch", flush=True)
        else:
            if "optimizer" in state and not args.use_sigma:
                optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state.get("epoch", 0))
            global_step = int(state.get("step", 0))
            best_mae = float(state.get("best_mae_dn", best_mae))
            best_ssim = float(state.get("best_ssim", best_ssim))
            best_composite = float(state.get("best_composite", best_composite))
            print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    if args.freeze_edge_head:
        if args.arch != "dual_head" or not hasattr(model, "ending_edge"):
            raise SystemExit("--freeze-edge-head requires --arch dual_head")
        frozen = 0
        for param in model.ending_edge.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen += int(param.numel())
        optimizer, optimizer_name = build_optimizer(model, args.lr)
        print(
            f"Froze ending_edge ({frozen} params); rebuilt optimizer={optimizer_name}",
            flush=True,
        )
    if args.freeze_flat_head:
        if args.arch != "dual_head" or not hasattr(model, "ending_flat"):
            raise SystemExit("--freeze-flat-head requires --arch dual_head")
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith("ending_flat.") or name.startswith("tower_flat."):
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += int(param.numel())
        optimizer, optimizer_name = build_optimizer(model, args.lr)
        print(
            f"Froze flat head ({frozen} params); rebuilt optimizer={optimizer_name}",
            flush=True,
        )
    if args.train_edge_only:
        if args.arch != "dual_head":
            raise SystemExit("--train-edge-only requires --arch dual_head")
        allow = ("ending_edge", "tower_edge", "mask_head", "lap_to_edge", "lap_to_edge_ms")
        frozen = 0
        trainable = 0
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in allow):
                param.requires_grad = True
                trainable += int(param.numel())
            else:
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += int(param.numel())
        optimizer, optimizer_name = build_optimizer(model, args.lr)
        print(
            f"train_edge_only: trainable={trainable} frozen={frozen} "
            f"optimizer={optimizer_name}",
            flush=True,
        )

    log_path = args.out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,step,total,pixel,teacher,grad,ssim_loss,mean,highpass,edge,"
            "flat_sigma,des_flat_sigma,des_noise_gain,teacher_weight,sec,max_mem_gb\n",
            encoding="utf-8",
        )
    validation_log = args.out_dir / "validation_log.csv"
    if not validation_log.exists():
        validation_log.write_text(
            "epoch,step,mae_dn,ssim,composite,edge_retention,edge_sobel_mae,des\n",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        teacher_weight = (
            args.teacher_weight_start * (1.0 - progress)
            + args.teacher_weight_end * progress
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        running = 0.0
        micro_index = 0
        last = {
            "total": 0.0,
            "pixel": 0.0,
            "teacher": 0.0,
            "grad": 0.0,
            "ssim_loss": 0.0,
            "mean": 0.0,
            "highpass": 0.0,
            "edge": 0.0,
            "flat_sigma": 0.0,
            "des_flat_sigma": 0.0,
            "des_noise_gain": 0.0,
        }

        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            teachers = batch["teacher"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            teacher_valid = batch["teacher_valid"].to(device, non_blocking=True)
            loss_weight = batch["loss_weight"].to(device, non_blocking=True)
            is_hard = batch["is_hard"].to(device, non_blocking=True)
            is_edge_focus = batch["is_edge_focus"].to(device, non_blocking=True)
            center = center_frame_index(args.input_frames)
            if (
                args.use_sigma
                and float(args.sigma_film_jitter) > 0.0
                and inputs.shape[1] > args.input_frames
            ):
                # Multiplicative σ jitter on the σ channel (KPN/FFDNet-style).
                j = float(args.sigma_film_jitter)
                scale = 1.0 + (2.0 * torch.rand(inputs.shape[0], 1, 1, 1, device=device) - 1.0) * j
                sigma_idx = args.input_frames
                inputs = inputs.clone()
                inputs[:, sigma_idx : sigma_idx + 1] = (
                    inputs[:, sigma_idx : sigma_idx + 1] * scale
                ).clamp_min(1e-4)
            # AIM / MR-CAS: random patch masking on image channels (half of batches).
            if float(getattr(args, "input_mask_ratio", 0.0) or 0.0) > 0.0:
                ratio = float(args.input_mask_ratio)
                psz = max(4, int(getattr(args, "input_mask_patch", 16) or 16))
                if float(torch.rand(())) < 0.5:
                    _b, _c, h, w = inputs.shape
                    gh, gw = max(1, h // psz), max(1, w // psz)
                    nmask = max(1, int(gh * gw * ratio))
                    flat = torch.rand(_b, gh * gw, device=inputs.device)
                    kth = torch.kthvalue(flat, min(nmask, gh * gw), dim=1).values.unsqueeze(1)
                    patch_mask = (flat <= kth).view(_b, 1, gh, gw).float()
                    patch_mask = torch.nn.functional.interpolate(
                        patch_mask, size=(h, w), mode="nearest"
                    )
                    n_img = int(args.input_frames)
                    inputs = inputs.clone()
                    inputs[:, :n_img] = inputs[:, :n_img] * (1.0 - patch_mask)
            noisy_center = inputs[:, center : center + 1]
            if args.teacher_mode == "merge_flat":
                # Multi-frame flat teacher: mean of (soft-gated) input VST frames.
                teachers = inputs[:, : args.input_frames].mean(dim=1, keepdim=True)
                teacher_valid = torch.ones(
                    inputs.shape[0], device=device, dtype=torch.float32
                )
            edge_teachers = None
            if edge_kd_model is not None:
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                    enabled=device.type == "cuda",
                ):
                    edge_teachers = edge_kd_model(inputs).float()

            if blend_sota_model is not None and blend_edgekd_model is not None:
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                    enabled=device.type == "cuda",
                ):
                    sota_pred = blend_sota_model(inputs).float()
                    edgekd_pred = blend_edgekd_model(inputs).float()
                guide = 0.5 * (sota_pred + edgekd_pred)
                edge_map_b = soft_edge_mask(
                    guide, temperature=float(args.blend_kd_temperature)
                )
                harden_b = float(getattr(args, "blend_kd_harden", 0.0) or 0.0)
                if harden_b > 0.0:
                    edge_map_b = torch.sigmoid(harden_b * (edge_map_b - 0.5))
                gain_b = float(getattr(args, "blend_kd_edge_gain", 1.0) or 1.0)
                if gain_b != 1.0:
                    edge_map_b = (edge_map_b * gain_b).clamp(0.0, 1.0)
                blend_pred = edge_map_b * edgekd_pred + (1.0 - edge_map_b) * sota_pred
                mix_b = float(np.clip(args.blend_kd_mix, 0.0, 1.0))
                if mix_b >= 1.0 - 1e-6:
                    targets = blend_pred
                    teachers = blend_pred
                    teacher_valid = torch.ones(
                        inputs.shape[0], device=device, dtype=torch.float32
                    )
                elif mix_b > 0.0:
                    targets = (1.0 - mix_b) * targets + mix_b * blend_pred
                    teachers = (1.0 - mix_b) * teachers + mix_b * blend_pred

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=device.type == "cuda",
            ):
                aux = None
                if args.arch == "dual_head":
                    pred, aux = model(inputs, return_aux=True)
                else:
                    pred = model(inputs)

                def _one_sample_loss(index: int):
                    ekd = (
                        edge_teachers[index : index + 1]
                        if edge_teachers is not None
                        else None
                    )
                    losses_i = distill_loss(
                        pred[index : index + 1],
                        targets[index : index + 1],
                        teachers[index : index + 1],
                        masks[index : index + 1],
                        teacher_valid[index : index + 1],
                        args.grad_weight,
                        args.ssim_weight,
                        args.mean_weight,
                        teacher_weight,
                        args.highpass_weight,
                        args.edge_weight,
                        flat_boost=args.flat_boost,
                        flat_sigma_weight=args.flat_sigma_weight,
                        teacher_mode=args.teacher_mode,
                        bm3d_flat_alpha=args.bm3d_flat_alpha,
                        loss_domain=args.loss_domain,
                        vst_intercept=intercept_from_exposure_map(
                            inputs[index : index + 1, -1:]
                        ),
                        noisy=noisy_center[index : index + 1],
                        des_flat_sigma_weight=args.des_flat_sigma_weight,
                        des_noise_gain_weight=args.des_noise_gain_weight,
                        is_hard=is_hard[index : index + 1],
                        hard_des_sigma_mult=args.hard_des_sigma_mult,
                        hard_des_hp_weight=args.hard_des_hp_weight,
                        hard_des_use_pure_teacher=args.hard_des_use_pure_teacher,
                        is_edge_focus=is_edge_focus[index : index + 1],
                        edge_focus_edge_mult=args.edge_focus_edge_mult,
                        edge_focus_flat_des_mult=args.edge_focus_flat_des_mult,
                        edge_focus_bm3d_alpha=args.edge_focus_bm3d_alpha,
                        distill_edge_harden=args.distill_edge_harden,
                        des_edge_fid_weight=args.des_edge_fid_weight,
                        edge_focus_des_fid_mult=args.edge_focus_des_fid_mult,
                        edge_teacher=ekd,
                        edge_kd_mix=args.edge_kd_mix,
                        mixed_grad_xy_weight=args.mixed_grad_xy_weight,
                        ms_edge_weight=args.ms_edge_weight,
                        ms_edge_scales=args.ms_edge_scales,
                    )
                    total_i = losses_i["total"]
                    if aux is not None and args.dual_aux_weight > 0:
                        vst_intercept = intercept_from_exposure_map(
                            inputs[index : index + 1, -1:]
                        )
                        pred_flat_d, pred_edge_d, target_d, teacher_d = (
                            maybe_to_loss_domain(
                                aux["pred_flat"][index : index + 1],
                                aux["pred_edge"][index : index + 1],
                                targets[index : index + 1],
                                teachers[index : index + 1],
                                loss_domain=args.loss_domain,
                                intercept=vst_intercept,
                            )
                        )
                        if args.loss_domain == "dn":
                            edge_map = soft_edge_mask(target_d)
                            flat_map = 1.0 - edge_map
                        else:
                            edge_map = aux["edge_map"][index : index + 1]
                            flat_map = aux["flat_map"][index : index + 1]
                        reliability = masks[index : index + 1].clamp(0.0, 1.0)
                        distill_target = build_distill_target(
                            target_d,
                            teacher_d,
                            teacher_valid[index : index + 1],
                            edge_map,
                            teacher_mode=args.teacher_mode,
                            bm3d_flat_alpha=args.bm3d_flat_alpha,
                            edge_teacher=(
                                None
                                if ekd is None
                                else maybe_to_loss_domain(
                                    ekd,
                                    loss_domain=args.loss_domain,
                                    intercept=vst_intercept,
                                )[0]
                            ),
                            edge_kd_mix=args.edge_kd_mix,
                        )
                        aux_flat = masked_charbonnier(
                            pred_flat_d,
                            distill_target,
                            reliability * flat_map,
                        )
                        edge_aux_ref = target_d
                        if ekd is not None and float(args.edge_kd_mix) > 0.0:
                            ekd_d = maybe_to_loss_domain(
                                ekd,
                                loss_domain=args.loss_domain,
                                intercept=vst_intercept,
                            )[0]
                            mix = float(np.clip(args.edge_kd_mix, 0.0, 1.0))
                            edge_aux_ref = (1.0 - mix) * target_d + mix * ekd_d
                        aux_edge = masked_charbonnier(
                            pred_edge_d,
                            edge_aux_ref,
                            reliability * edge_map,
                        )
                        total_i = total_i + args.dual_aux_weight * (aux_flat + aux_edge)
                        losses_i["aux_flat"] = aux_flat
                        losses_i["aux_edge"] = aux_edge
                    return total_i, losses_i

                use_scene_weight = float(args.scene_loss_power) > 0.0
                if use_scene_weight and inputs.shape[0] > 1:
                    weighted = []
                    last_parts = None
                    for index in range(inputs.shape[0]):
                        total_i, parts = _one_sample_loss(index)
                        weighted.append(total_i * loss_weight[index])
                        last_parts = parts
                    total = torch.stack(weighted).sum() / loss_weight.sum().clamp_min(
                        1e-6
                    )
                    losses = last_parts if last_parts is not None else {"total": total}
                    losses["total"] = total
                else:
                    losses = distill_loss(
                        pred,
                        targets,
                        teachers,
                        masks,
                        teacher_valid,
                        args.grad_weight,
                        args.ssim_weight,
                        args.mean_weight,
                        teacher_weight,
                        args.highpass_weight,
                        args.edge_weight,
                        flat_boost=args.flat_boost,
                        flat_sigma_weight=args.flat_sigma_weight,
                        teacher_mode=args.teacher_mode,
                        bm3d_flat_alpha=args.bm3d_flat_alpha,
                        loss_domain=args.loss_domain,
                        vst_intercept=intercept_from_exposure_map(inputs[:, -1:]),
                        noisy=noisy_center,
                        des_flat_sigma_weight=args.des_flat_sigma_weight,
                        des_noise_gain_weight=args.des_noise_gain_weight,
                        is_hard=is_hard,
                        hard_des_sigma_mult=args.hard_des_sigma_mult,
                        hard_des_hp_weight=args.hard_des_hp_weight,
                        hard_des_use_pure_teacher=args.hard_des_use_pure_teacher,
                        is_edge_focus=is_edge_focus,
                        edge_focus_edge_mult=args.edge_focus_edge_mult,
                        edge_focus_flat_des_mult=args.edge_focus_flat_des_mult,
                        edge_focus_bm3d_alpha=args.edge_focus_bm3d_alpha,
                        distill_edge_harden=args.distill_edge_harden,
                        des_edge_fid_weight=args.des_edge_fid_weight,
                        edge_focus_des_fid_mult=args.edge_focus_des_fid_mult,
                        edge_teacher=edge_teachers,
                        edge_kd_mix=args.edge_kd_mix,
                        mixed_grad_xy_weight=args.mixed_grad_xy_weight,
                        ms_edge_weight=args.ms_edge_weight,
                        ms_edge_scales=args.ms_edge_scales,
                    )
                    total = losses["total"]
                    if aux is not None and args.dual_aux_weight > 0:
                        vst_intercept = intercept_from_exposure_map(inputs[:, -1:])
                        pred_flat_d, pred_edge_d, target_d, teacher_d = (
                            maybe_to_loss_domain(
                                aux["pred_flat"],
                                aux["pred_edge"],
                                targets,
                                teachers,
                                loss_domain=args.loss_domain,
                                intercept=vst_intercept,
                            )
                        )
                        if args.loss_domain == "dn":
                            edge_map = soft_edge_mask(target_d)
                            flat_map = 1.0 - edge_map
                        else:
                            edge_map = aux["edge_map"]
                            flat_map = aux["flat_map"]
                        reliability = masks.clamp(0.0, 1.0)
                        ekd_d = None
                        if edge_teachers is not None:
                            ekd_d = maybe_to_loss_domain(
                                edge_teachers,
                                loss_domain=args.loss_domain,
                                intercept=vst_intercept,
                            )[0]
                        distill_target = build_distill_target(
                            target_d,
                            teacher_d,
                            teacher_valid,
                            edge_map,
                            teacher_mode=args.teacher_mode,
                            bm3d_flat_alpha=args.bm3d_flat_alpha,
                            edge_teacher=ekd_d,
                            edge_kd_mix=args.edge_kd_mix,
                        )
                        aux_flat = masked_charbonnier(
                            pred_flat_d,
                            distill_target,
                            reliability * flat_map,
                        )
                        edge_aux_ref = target_d
                        if ekd_d is not None and float(args.edge_kd_mix) > 0.0:
                            mix = float(np.clip(args.edge_kd_mix, 0.0, 1.0))
                            edge_aux_ref = (1.0 - mix) * target_d + mix * ekd_d
                        aux_edge = masked_charbonnier(
                            pred_edge_d,
                            edge_aux_ref,
                            reliability * edge_map,
                        )
                        aux_total = args.dual_aux_weight * (aux_flat + aux_edge)
                        total = total + aux_total
                        losses["aux_flat"] = aux_flat
                        losses["aux_edge"] = aux_edge
                    if use_scene_weight:
                        total = total * loss_weight.mean()
                        losses["total"] = total
            (total / args.accum_steps).backward()
            micro_index += 1
            running += float(total.detach())
            last = {key: float(value.detach()) for key, value in losses.items()}

            if micro_index % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            f"epoch {epoch}: total={running / steps:.5f} pixel={last['pixel']:.5f} "
            f"teacher={last['teacher']:.5f} hp={last['highpass']:.5f} "
            f"edge={last['edge']:.5f} flat_σ={last.get('flat_sigma', 0):.5f} "
            f"des_σ={last.get('des_flat_sigma', 0):.5f} "
            f"des_g={last.get('des_noise_gain', 0):.5f} "
            f"tw={teacher_weight:.3f} {elapsed:.1f}s mem={max_mem:.2f}GB",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{global_step},{running / steps:.6f},{last['pixel']:.6f},"
                f"{last['teacher']:.6f},{last['grad']:.6f},{last['ssim_loss']:.6f},"
                f"{last['mean']:.6f},{last['highpass']:.6f},{last['edge']:.6f},"
                f"{last.get('flat_sigma', 0):.6f},"
                f"{last.get('des_flat_sigma', 0):.6f},"
                f"{last.get('des_noise_gain', 0):.6f},"
                f"{teacher_weight:.6f},{elapsed:.2f},{max_mem:.3f}\n"
            )

        should_validate = (epoch + 1) % args.validation_every == 0 or epoch + 1 == args.epochs
        mae = best_mae if best_mae < float("inf") else 0.0
        ssim = best_ssim if best_ssim > float("-inf") else 0.0
        if should_validate:
            try:
                mae, ssim, edge_retention, edge_sobel_mae, des = (
                    validate_real_sequences_with_edges(
                        model,
                        args.validation_dir,
                        args.validation_frame_index,
                        args.input_frames,
                        device,
                        args.validation_tile,
                    )
                )
            except Exception as error:
                # Keep training / last.pt even if validation path is broken.
                print(f"validation failed (continuing): {error}", flush=True)
                mae = best_mae if best_mae < float("inf") else 0.0
                ssim = best_ssim if best_ssim > float("-inf") else 0.0
            else:
                composite = mae / 1023.0 + (1.0 - ssim)
                print(
                    f"validation: MAE={mae:.4f} DN SSIM={ssim:.6f} "
                    f"edge_ret={edge_retention:.4f} edge_mae={edge_sobel_mae:.2f} "
                    f"DES={des:.4f} composite={composite:.6f}",
                    flush=True,
                )
                with validation_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{epoch},{global_step},{mae:.6f},{ssim:.8f},{composite:.8f},"
                        f"{edge_retention:.6f},{edge_sobel_mae:.6f},{des:.6f}\n"
                    )
                if mae < best_mae:
                    best_mae = mae
                    save_model_checkpoint(
                        args.out_dir / "best_mae.pt",
                        model,
                        epoch + 1,
                        global_step,
                        args,
                        mae,
                        ssim,
                    )
                if ssim > best_ssim:
                    best_ssim = ssim
                    save_model_checkpoint(
                        args.out_dir / "best_ssim.pt",
                        model,
                        epoch + 1,
                        global_step,
                        args,
                        mae,
                        ssim,
                    )
                if composite < best_composite:
                    best_composite = composite
                    save_model_checkpoint(
                        args.out_dir / "best_composite.pt",
                        model,
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
                        model,
                        epoch + 1,
                        global_step,
                        args,
                        mae,
                        ssim,
                    )
                    save_model_checkpoint(
                        args.out_dir / "best_des.pt",
                        model,
                        epoch + 1,
                        global_step,
                        args,
                        mae,
                        ssim,
                    )

        # Always persist last weights (do not gate on successful validation).
        save_model_checkpoint(
            args.out_dir / "last.pt",
            model,
            epoch + 1,
            global_step,
            args,
            mae,
            ssim,
        )

        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
