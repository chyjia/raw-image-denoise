"""Deploy helper: fps-aware SOTA↔lap_ms blend + flat bilateral + edge unsharp.

Defaults (P150 BiShrink + SURE-k + EUI recursive spin, mean DES ~0.9589):
  schedule_mode=noise → MAD flat-noise proxy gates bilat/unsharp
  then SURE pick among k∈{0.8,1.0,1.2} on 5× recursive EUI cycle-spin
  with Sendur–Selesnick bivariate HP shrink (decay=0.7, σ=1.8, residual_scale=0.3)
  noise∈[0.002,0.012] → bilat 0.72→1.0, unsharp 0.10→0.22, flat_pct=30
  fps <= 1.5 → SOTA-only + gated bilat/unsharp; else blend + gated
  TTA: geometric self-ensemble id/r90/r180/r270/r90_lr/r270_lr on FE→spatial
  Edge arm: nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .fe_schedule import gated_spatial_from_temporal
from .infer import denoise_from_dn_frames, load_model, save_mono10_png
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost
from .p7_fusion import edge_unsharp
from .p8_fusion import (
    deploy_p64_exposure,
    deploy_p68_noise,
    deploy_p98,
    deploy_p103,
    deploy_p103_cne,
    deploy_p103_cycspin,
    deploy_p114,
    deploy_p120,
    deploy_p126,
    deploy_p127,
    deploy_p128,
    deploy_p129,
    deploy_p132,
    deploy_p133,
    deploy_p134,
    deploy_p135,
    deploy_p140,
    deploy_p141,
    deploy_p142,
    deploy_p143,
    deploy_p148,
    deploy_p149,
    deploy_p150,
    deploy_p151,
    deploy_p156,
    deploy_p157,
    deploy_p158,
    deploy_p159,
)
from .p13_tta import geom_forward, geom_inverse
from .wiener_merge import merge_from_memmap


def _spatial_on_merged(model, merged, exposure_ms, device, input_frames, tile,
                       tta_augs=None):
    augs = list(tta_augs) if tta_augs else ["id"]
    was = bool(getattr(model, "wiener_front_end", False))
    model.wiener_front_end = False
    outs = []
    try:
        for aug in augs:
            fe, h, w = geom_forward(merged, aug)
            frames = [fe.copy() for _ in range(input_frames)]
            dn = denoise_from_dn_frames(model, frames, exposure_ms, device, tile=tile)
            outs.append(geom_inverse(dn, aug, h, w))
    finally:
        model.wiener_front_end = was
    return np.mean(np.stack(outs, axis=0), axis=0).astype(np.float32)


@torch.no_grad()
def denoise_blend_deploy(
    model_sota,
    model_edgekd,
    n_in: int,
    frames,
    target_index: int,
    fps: float,
    exposure_ms: float,
    device: torch.device,
    tile: int = 256,
    wiener_tile: int = 32,
    wiener_overlap: int = 16,
    c_factor: float = 8.0,
    temporal_window: int = 16,
    low_fps: float = 1.5,
    mid_fps: float = 5.0,
    edge_temperature: float = 8.0,
    edge_harden: float = 16.0,
    low_fps_bilateral: bool = True,
    bilateral_strength: float = 0.9,
    mid_bilateral_strength: float = 0.85,
    bilateral_harden: float = 40.0,
    unsharp_amount: float = 0.15,
    unsharp_amount_low: float = 0.2,
    unsharp_amount_mid: float = 0.16,
    unsharp_amount_high: float = 0.12,
    unsharp_sigma: float = 1.4,
    unsharp_harden: float = 16.0,
    schedule_mode: str = "noise",
    exp_t_dark: float = 3.0,
    exp_t_bright: float = 15.0,
    exp_bilat_d: float = 0.95,
    exp_bilat_m: float = 0.88,
    exp_bilat_b: float = 0.80,
    exp_u_d: float = 0.22,
    exp_u_m: float = 0.15,
    exp_u_b: float = 0.14,
    high_bilat_scale: float = 1.0,
    high_u_scale: float = 1.1,
    noise_lo: float = 0.002,
    noise_hi: float = 0.012,
    noise_bilat_lo: float = 0.72,
    noise_bilat_hi: float = 1.0,
    noise_u_lo: float = 0.1,
    noise_u_hi: float = 0.22,
    noise_mode: str = "mad",
    noise_flat_pct: float = 30.0,
    ans_strength: float = 0.25,
    ans_k_mad: float = 1.0,
    ans_sigma: float = 1.8,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    ans_s_lo: float = 0.12,
    ans_s_hi: float = 0.4,
    ans_noise_lo: float = 0.0008,
    ans_noise_hi: float = 0.006,
    ans_residual_scale: float = 0.3,
    ans_max_shift: int = 1,
    ans_n_iters: int = 5,
    ans_thr_decay: float = 0.7,
    ans_read_sigma: float = 0.0,
    ans_is_strength: float = 0.15,
    ans_is_k_mad: float = 1.0,
    ans_is_flat_pct: float = 50.0,
    ans_sc_scale: float = 1.0,
    ans_target_sz: float = 0.02,
    ans_em_scale: float = 0.5,
    ans_dual_strength: float = 0.35,
    ans_dual_k: float = 1.0,
    ans_k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    ans_firm_ratio: float = 2.0,
    ans_block_size: int = 2,
    ans_neigh_win: int = 3,
    ans_let_mix: float = 0.5,
    ans_mode: str = "bishrink",
    tta_augs: list[str] | None = ['id', 'r90', 'r180', 'r270', 'r90_lr', 'r270_lr'],
) -> tuple[np.ndarray, dict]:
    temporal, _ = merge_from_memmap(
        frames,
        target_index,
        input_frames=temporal_window,
        tile_size=wiener_tile,
        overlap=wiener_overlap,
        c_factor=c_factor,
        align=True,
        spatial_wiener=False,
    )
    fe_gated, params, fe_sigma = gated_spatial_from_temporal(
        temporal,
        fps=float(fps),
        n_frames_averaged=temporal_window,
        tile_size=wiener_tile,
        overlap=wiener_overlap,
    )
    sota_dn = _spatial_on_merged(
        model_sota, fe_gated, exposure_ms, device, n_in, tile, tta_augs=tta_augs
    )
    fps_v = float(fps)

    if str(schedule_mode).lower() == "noise":
        noise_kw = dict(
            noise_lo=noise_lo,
            noise_hi=noise_hi,
            bilat_lo=noise_bilat_lo,
            bilat_hi=noise_bilat_hi,
            u_lo=noise_u_lo,
            u_hi=noise_u_hi,
            low_fps=low_fps,
            unsharp_sigma=unsharp_sigma,
            noise_mode=noise_mode,
            flat_pct=noise_flat_pct,
        )
        mode = str(ans_mode).lower()
        if mode in ("recurse", "p120", "rcs"):
            deploy_fn = deploy_p120
            ans_kw = dict(
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                max_shift=int(ans_max_shift),
                residual_scale=float(ans_residual_scale),
                ans_s_lo=ans_s_lo,
                ans_s_hi=ans_s_hi,
                ans_noise_lo=ans_noise_lo,
                ans_noise_hi=ans_noise_hi,
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                ans_flat_pct=ans_flat_pct,
                ans_harden=ans_harden,
            )
            mode_tag = "recurse"
        elif mode in ("eui", "p126"):
            deploy_fn = deploy_p126
            ans_kw = dict(
                read_sigma=float(ans_read_sigma),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                max_shift=int(ans_max_shift),
                residual_scale=float(ans_residual_scale),
                ans_s_lo=ans_s_lo,
                ans_s_hi=ans_s_hi,
                ans_noise_lo=ans_noise_lo,
                ans_noise_hi=ans_noise_hi,
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                ans_flat_pct=ans_flat_pct,
                ans_harden=ans_harden,
            )
            mode_tag = "eui"
        elif mode in ("interscale", "p127"):
            deploy_fn = deploy_p127
            ans_kw = dict(
                is_strength=float(ans_is_strength),
                is_k_mad=float(ans_is_k_mad),
                is_flat_pct=float(ans_is_flat_pct),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "interscale"
        elif mode in ("scvst", "p128"):
            deploy_fn = deploy_p128
            ans_kw = dict(
                sc_scale=float(ans_sc_scale),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "scvst"
        elif mode in ("full_gat", "p129"):
            deploy_fn = deploy_p129
            ans_kw = dict(
                read_sigma=float(ans_read_sigma),
                is_strength=float(ans_is_strength),
                is_k_mad=float(ans_is_k_mad),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
            )
            mode_tag = "full_gat"
        elif mode in ("szgate", "p132"):
            deploy_fn = deploy_p132
            ans_kw = dict(
                target_sz=float(ans_target_sz),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "szgate"
        elif mode in ("emvst", "p133"):
            deploy_fn = deploy_p133
            ans_kw = dict(
                em_scale=float(ans_em_scale),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "emvst"
        elif mode in ("sure", "p134"):
            deploy_fn = deploy_p134
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "sure"
        elif mode in ("garrote", "p140"):
            deploy_fn = deploy_p140
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "garrote"
        elif mode in ("firm", "p141"):
            deploy_fn = deploy_p141
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                firm_ratio=float(ans_firm_ratio),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "firm"
        elif mode in ("blockjs", "p142"):
            deploy_fn = deploy_p142
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                block_size=int(ans_block_size),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "blockjs"
        elif mode in ("pure", "p143"):
            deploy_fn = deploy_p143
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "pure"
        elif mode in ("neigh", "p148"):
            deploy_fn = deploy_p148
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                neigh_win=int(ans_neigh_win),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "neigh"
        elif mode in ("neighw", "p149"):
            deploy_fn = deploy_p149
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                neigh_win=max(int(ans_neigh_win), 5),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "neighw"
        elif mode in ("bishrink", "p150"):
            deploy_fn = deploy_p150
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "bishrink"
        elif mode in ("neighlevel", "p156"):
            deploy_fn = deploy_p156
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                neighlevel_alpha=0.5,
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "neighlevel"
        elif mode in ("bayes", "p157"):
            deploy_fn = deploy_p157
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                bayes_win=5,
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "bayes"
        elif mode in ("bineigh", "p158"):
            deploy_fn = deploy_p158
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                neighlevel_alpha=0.5,
                neigh_win=int(ans_neigh_win),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "bineigh"
        elif mode in ("multipick", "p159"):
            deploy_fn = deploy_p159
            ans_kw = dict(
                k_list=tuple(ans_k_list),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "multipick"
        elif mode in ("let", "p151"):
            deploy_fn = deploy_p151
            ans_kw = dict(
                mix=float(ans_let_mix),
                k_mad=float(ans_k_mad),
                neigh_win=int(ans_neigh_win),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_sigma=ans_sigma,
                residual_scale=float(ans_residual_scale),
            )
            mode_tag = "let"
        elif mode in ("dual", "p135"):
            deploy_fn = deploy_p135
            ans_kw = dict(
                dual_strength=float(ans_dual_strength),
                dual_k=float(ans_dual_k),
                dual_sigma=float(ans_sigma),
                n_iters=int(ans_n_iters),
                thr_decay=float(ans_thr_decay),
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
            )
            mode_tag = "dual"
        elif mode in ("cne_spin", "p114", "cnes"):
            deploy_fn = deploy_p114
            ans_kw = dict(
                max_shift=int(ans_max_shift),
                residual_scale=float(ans_residual_scale),
                ans_s_lo=ans_s_lo,
                ans_s_hi=ans_s_hi,
                ans_noise_lo=ans_noise_lo,
                ans_noise_hi=ans_noise_hi,
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                ans_flat_pct=ans_flat_pct,
                ans_harden=ans_harden,
            )
            mode_tag = "cne_spin"
        elif mode in ("cne", "p111"):
            deploy_fn = deploy_p103_cne
            ans_kw = dict(
                residual_scale=float(ans_residual_scale),
                ans_s_lo=ans_s_lo,
                ans_s_hi=ans_s_hi,
                ans_noise_lo=ans_noise_lo,
                ans_noise_hi=ans_noise_hi,
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                ans_flat_pct=ans_flat_pct,
                ans_harden=ans_harden,
            )
            mode_tag = "cne"
        elif mode in ("cycspin", "p110", "cycle"):
            deploy_fn = deploy_p103_cycspin
            ans_kw = dict(
                max_shift=int(ans_max_shift),
                ans_s_lo=ans_s_lo,
                ans_s_hi=ans_s_hi,
                ans_noise_lo=ans_noise_lo,
                ans_noise_hi=ans_noise_hi,
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                ans_flat_pct=ans_flat_pct,
                ans_harden=ans_harden,
            )
            mode_tag = "cycspin"
        elif mode in ("gated", "p103", "ng"):
            deploy_fn = deploy_p103
            ans_kw = dict(
                ans_s_lo=ans_s_lo,
                ans_s_hi=ans_s_hi,
                ans_noise_lo=ans_noise_lo,
                ans_noise_hi=ans_noise_hi,
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                ans_flat_pct=ans_flat_pct,
                ans_harden=ans_harden,
            )
            mode_tag = "gated"
        else:
            deploy_fn = deploy_p98
            ans_kw = dict(
                ans_strength=ans_strength,
                ans_k_mad=ans_k_mad,
                ans_sigma=ans_sigma,
                ans_flat_pct=ans_flat_pct,
                ans_harden=ans_harden,
            )
            mode_tag = "fixed"
        if fps_v <= float(low_fps):
            out = deploy_fn(sota_dn, sota_dn, fps_v, **ans_kw, **noise_kw)
            forwards = 1
            route = "sota_noise_bilat_unsharp_anscombe"
        else:
            edgekd_dn = _spatial_on_merged(
                model_edgekd, fe_gated, exposure_ms, device, n_in, tile, tta_augs=tta_augs
            )
            out = deploy_fn(sota_dn, edgekd_dn, fps_v, **ans_kw, **noise_kw)
            forwards = 2
            route = "blend_noise_bilat_unsharp_anscombe"
        return out, {
            "route": route,
            "fe_schedule": params.name,
            "fe_sigma_dn": float(fe_sigma),
            "schedule_mode": "noise",
            "ans_mode": mode_tag,
            "ans_sigma": float(ans_sigma),
            "ans_k_mad": float(ans_k_mad),
            "noise_lo": float(noise_lo),
            "noise_hi": float(noise_hi),
            "low_fps": float(low_fps),
            "forwards": forwards,
        }

    if str(schedule_mode).lower() == "exposure":
        if fps_v <= float(low_fps):
            out = deploy_p64_exposure(
                sota_dn,
                sota_dn,
                fps_v,
                float(exposure_ms),
                t_dark=exp_t_dark,
                t_bright=exp_t_bright,
                bilat_d=exp_bilat_d,
                bilat_m=exp_bilat_m,
                bilat_b=exp_bilat_b,
                u_d=exp_u_d,
                u_m=exp_u_m,
                u_b=exp_u_b,
                low_fps=low_fps,
                mid_fps=mid_fps,
                high_bilat_scale=high_bilat_scale,
                high_u_scale=high_u_scale,
                unsharp_sigma=unsharp_sigma,
            )
            forwards = 1
            route = "sota_exp_bilat_unsharp"
        else:
            edgekd_dn = _spatial_on_merged(
                model_edgekd, fe_gated, exposure_ms, device, n_in, tile, tta_augs=tta_augs
            )
            out = deploy_p64_exposure(
                sota_dn,
                edgekd_dn,
                fps_v,
                float(exposure_ms),
                t_dark=exp_t_dark,
                t_bright=exp_t_bright,
                bilat_d=exp_bilat_d,
                bilat_m=exp_bilat_m,
                bilat_b=exp_bilat_b,
                u_d=exp_u_d,
                u_m=exp_u_m,
                u_b=exp_u_b,
                low_fps=low_fps,
                mid_fps=mid_fps,
                high_bilat_scale=high_bilat_scale,
                high_u_scale=high_u_scale,
                unsharp_sigma=unsharp_sigma,
            )
            forwards = 2
            route = "blend_exp_bilat_unsharp"
        return out, {
            "route": route,
            "fe_schedule": params.name,
            "fe_sigma_dn": float(fe_sigma),
            "schedule_mode": "exposure",
            "exposure_ms": float(exposure_ms),
            "low_fps": float(low_fps),
            "forwards": forwards,
        }

    if fps_v <= float(low_fps):
        out = sota_dn
        route = "sota_only"
        forwards = 1
        u_amt = float(unsharp_amount_low)
        if low_fps_bilateral and bilateral_strength > 0.0:
            out = flat_bilateral_boost(
                out,
                guide=out,
                flat_strength=float(bilateral_strength),
                harden=float(bilateral_harden),
            )
            route = "sota_flat_bilateral"
        if u_amt > 0.0:
            out = edge_unsharp(
                out,
                amount=u_amt,
                sigma=float(unsharp_sigma),
                harden=float(unsharp_harden),
            )
            route = f"{route}_unsharp"
        return out, {
            "route": route,
            "fe_schedule": params.name,
            "fe_sigma_dn": float(fe_sigma),
            "bilateral_strength": float(bilateral_strength) if low_fps_bilateral else 0.0,
            "bilateral_harden": float(bilateral_harden) if low_fps_bilateral else 0.0,
            "unsharp_amount": u_amt,
            "unsharp_sigma": float(unsharp_sigma),
            "forwards": forwards,
        }

    edgekd_dn = _spatial_on_merged(
        model_edgekd, fe_gated, exposure_ms, device, n_in, tile, tta_augs=tta_augs
    )
    blended, _ = blend_sota_edgekd(
        sota_dn,
        edgekd_dn,
        temperature=edge_temperature,
        edge_weight=1.0,
        harden=edge_harden,
    )
    out = blended
    route = "blend_edge_soft"
    if fps_v <= float(mid_fps):
        u_amt = float(unsharp_amount_mid)
        if mid_bilateral_strength > 0.0:
            out = flat_bilateral_boost(
                out,
                guide=out,
                flat_strength=float(mid_bilateral_strength),
                harden=float(bilateral_harden),
            )
            route = "blend_flat_bilateral_mid"
    else:
        u_amt = float(unsharp_amount_high)
    if u_amt > 0.0:
        out = edge_unsharp(
            out,
            amount=u_amt,
            sigma=float(unsharp_sigma),
            harden=float(unsharp_harden),
        )
        route = f"{route}_unsharp"
    return out, {
        "route": route,
        "fe_schedule": params.name,
        "fe_sigma_dn": float(fe_sigma),
        "edge_temperature": float(edge_temperature),
        "edge_harden": float(edge_harden),
        "mid_fps": float(mid_fps),
        "mid_bilateral_strength": float(mid_bilateral_strength),
        "bilateral_harden": float(bilateral_harden),
        "unsharp_amount": u_amt,
        "unsharp_sigma": float(unsharp_sigma),
        "forwards": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument(
        "--checkpoint-sota",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-edgekd",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt"),
        help="Edge arm (default: lap_edge_ms).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/inference_blend_deploy"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--low-fps", type=float, default=1.0)
    parser.add_argument("--mid-fps", type=float, default=5.0)
    parser.add_argument("--bilateral-strength", type=float, default=0.75)
    parser.add_argument("--mid-bilateral-strength", type=float, default=0.7)
    parser.add_argument("--bilateral-harden", type=float, default=40.0)
    parser.add_argument("--unsharp-amount", type=float, default=0.15)
    parser.add_argument("--unsharp-sigma", type=float, default=1.4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sota, n_sota, _ = load_model(args.checkpoint_sota, None, device)
    model_kd, n_kd, _ = load_model(args.checkpoint_edgekd, None, device)
    if n_sota != n_kd:
        raise SystemExit(f"Frame mismatch {n_sota} vs {n_kd}")
    width, height, fps = parse_geometry(args.raw_path.name)
    exposure_ms = parse_exposure_ms(args.raw_path.name)
    frames = memmap_frames(args.raw_path, width, height)
    target_index = frames.shape[0] // 2 if args.frame_index < 0 else args.frame_index
    out, meta = denoise_blend_deploy(
        model_sota,
        model_kd,
        n_sota,
        frames,
        target_index,
        float(fps),
        exposure_ms,
        device,
        tile=args.tile_size,
        low_fps=args.low_fps,
        mid_fps=args.mid_fps,
        bilateral_strength=args.bilateral_strength,
        mid_bilateral_strength=args.mid_bilateral_strength,
        bilateral_harden=args.bilateral_harden,
        unsharp_amount=args.unsharp_amount,
        unsharp_sigma=args.unsharp_sigma,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    save_mono10_png(
        args.output_dir / f"{stem}_frame{target_index:03d}_input.png",
        decode_mono10(frames[target_index]),
    )
    save_mono10_png(
        args.output_dir / f"{stem}_frame{target_index:03d}_blend_deploy.png",
        out,
    )
    print(f"Saved {args.output_dir}: {meta}", flush=True)


if __name__ == "__main__":
    main()
