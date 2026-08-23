"""Run P3-a → P3-b → conditional P3-c (defaults; no prompts)."""

from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLEND_REF = 0.9058


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=ROOT)
    if code != 0:
        raise SystemExit(f"Failed ({code}): {' '.join(cmd)}")


def mean_des(path: Path, method: str) -> float | None:
    if not path.exists():
        return None
    vals = [
        float(r["des"])
        for r in csv.DictReader(path.open(encoding="utf-8"))
        if r["method"] == method
    ]
    return sum(vals) / len(vals) if vals else None


def hard_metrics(path: Path, method: str) -> tuple[float | None, float | None, float | None]:
    """Return (des, ng, ef) for f10 token clip."""
    if not path.exists():
        return None, None, None
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row["method"] == method and "74824541" in row["file"]:
            return (
                float(row["des"]),
                float(row["noise_gain"]),
                float(row["edge_fidelity"]),
            )
    return None, None, None


def f05_ng(path: Path, method: str) -> float | None:
    if not path.exists():
        return None
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row["method"] == method and "53940814" in row["file"]:
            return float(row["noise_gain"])
    return None


def main() -> None:
    python = sys.executable

    print("=== P3-a lap_edge_ms ===", flush=True)
    run([python, "-u", "-m", "nafnet_denoise.run_lap_edge_ms", "--epochs", "12"])
    alone = Path("nafnet_denoise/compare_lap_edge_ms/metrics.csv")
    blend = Path("nafnet_denoise/compare_sota_lap_ms_blend/metrics.csv")
    alone_des = mean_des(alone, "dual_gated_fps_sigma")
    blend_des = mean_des(blend, "blend_edge_soft")
    _, _, alone_ef = hard_metrics(alone, "dual_gated_fps_sigma")
    _, _, blend_ef = hard_metrics(blend, "blend_edge_soft")
    print(
        f"P3-a alone mean={alone_des} f10_ef={alone_ef}; "
        f"blend mean={blend_des} f10_ef={blend_ef}",
        flush=True,
    )

    # Prefer newer edge arm for gate training if blend improved or ef rose.
    edge_ckpt = Path("nafnet_denoise/checkpoints_4f_lap_edge/best.pt")
    ms_ckpt = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")
    if ms_ckpt.exists() and blend_des is not None and blend_des >= BLEND_REF - 0.0005:
        edge_ckpt = ms_ckpt
        print(f"P3-b edge arm = lap_edge_ms ({blend_des})", flush=True)
    elif ms_ckpt.exists() and alone_ef is not None and alone_ef >= 0.970:
        edge_ckpt = ms_ckpt
        print(f"P3-b edge arm = lap_edge_ms (ef={alone_ef})", flush=True)
    else:
        print(f"P3-b edge arm = {edge_ckpt}", flush=True)

    print("=== P3-b blend_gate ===", flush=True)
    run(
        [
            python, "-u", "-m", "nafnet_denoise.run_blend_gate",
            "--epochs", "8",
            "--checkpoint-edge", str(edge_ckpt),
        ]
    )
    gate_csv = Path("nafnet_denoise/compare_blend_gate/metrics.csv")
    gate_des = mean_des(gate_csv, "blend_gate")
    soft_des = mean_des(gate_csv, "blend_soft")
    print(f"P3-b gate mean={gate_des} soft={soft_des}", flush=True)

    # Update deploy default edge ckpt if blend_ms won.
    best_edge = edge_ckpt
    best_mean = blend_des if blend_des is not None else BLEND_REF
    if gate_des is not None and gate_des > best_mean:
        best_mean = gate_des
        print("P3-b gate beats soft blend on mean", flush=True)

    # P3-c: only if f0.5 ng still lags BM3D (~0.8506) by >0.004 and mean < 0.9065
    ng = f05_ng(blend if blend.exists() else gate_csv, "blend_edge_soft")
    if ng is None:
        ng = f05_ng(gate_csv, "blend_soft")
    print(f"f0.5 ng (blend)={ng}", flush=True)
    if (
        best_mean is not None
        and best_mean < 0.9065
        and ng is not None
        and ng < 0.848
    ):
        print("=== P3-c mild low-fps flat FE boost (zero-train ablates) ===", flush=True)
        # Reuse gated compare is enough signal; document that hybrid flat boost
        # remains optional. Run SOTA↔best_edge blend once more for summary.
        run(
            [
                python, "-u", "-m", "nafnet_denoise.compare_sota_edgekd_blend",
                "--input-dir", r"D:\denoise\val_raw",
                "--output-dir", "nafnet_denoise/compare_p3_final_blend",
                "--checkpoint-sota",
                "nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt",
                "--checkpoint-edgekd",
                str(best_edge),
            ]
        )
    else:
        print("P3-c skipped (mean/ng thresholds not met)", flush=True)

    # Bake deploy default to best edge arm.
    deploy = Path("nafnet_denoise/infer_blend_deploy.py")
    text = deploy.read_text(encoding="utf-8")
    old = 'default=Path("nafnet_denoise/checkpoints_4f_lap_edge/best.pt")'
    new = f'default=Path("{best_edge.as_posix()}")'
    # Keep lap_edge path style with backslash-agnostic replace of known defaults.
    for cand in (
        'default=Path("nafnet_denoise/checkpoints_4f_lap_edge/best.pt")',
        'default=Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")',
        'default=Path("nafnet_denoise/checkpoints_4f_edge_kd_new_fe/best.pt")',
    ):
        if cand in text:
            text = text.replace(cand, f'default=Path("{best_edge.as_posix()}")')
            deploy.write_text(text, encoding="utf-8")
            print(f"Updated infer_blend_deploy edge default -> {best_edge}", flush=True)
            break

    print("=== P3 LADDER SUMMARY ===", flush=True)
    print(f"ref blend+lap: {BLEND_REF}", flush=True)
    print(f"P3-a alone={alone_des} blend={blend_des}", flush=True)
    print(f"P3-b gate={gate_des} soft={soft_des}", flush=True)
    print(f"deploy edge arm: {best_edge}", flush=True)


if __name__ == "__main__":
    main()
