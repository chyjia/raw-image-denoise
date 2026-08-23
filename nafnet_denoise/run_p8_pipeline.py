"""Orchestrate P8 meta (up to 100 gens) then optional short edge-HP FT if saturated.

Default path when zero-train stagnates: edge-masked highpass distill from
lap_ms teacher into SOTA flat arm (Young/FitNet-lite output+HP, not full feature
hooks — keeps runtime bounded).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    python = sys.executable
    meta_out = Path("nafnet_denoise/compare_meta100_p8")
    cmd_meta = [
        python,
        "-u",
        "-m",
        "nafnet_denoise.run_des_meta100_p8",
        "--max-gens",
        "100",
        "--iters",
        "100",
        "--patience",
        "5",
        "--output-dir",
        str(meta_out),
    ]
    print(">>> P8 meta start", flush=True)
    code = subprocess.call(cmd_meta, cwd=ROOT)
    if code != 0:
        raise SystemExit(f"P8 meta failed ({code})")

    ledger_path = meta_out / "ledger.json"
    baseline = 0.9221
    best = 0.0
    if ledger_path.exists():
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        gb = data.get("global_best") or {}
        best = float(gb.get("mean_des") or 0.0)
        baseline = float(data.get("baseline_p7") or baseline)
    print(f">>> P8 meta best={best:.4f} baseline={baseline:.4f}", flush=True)

    if best >= baseline + 1e-4:
        print(">>> Gain found in zero-train; skip FT (default).", flush=True)
        return

    # Saturated → short edge-HP FT (default next lever from lit)
    print(">>> Zero-train saturated; start short edge-HP FT...", flush=True)
    ft_dir = Path("nafnet_denoise/checkpoints_4f_p8_edge_hp")
    compare = Path("nafnet_denoise/compare_p8_edge_hp")
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    if not hard_manifest.exists():
        print(f">>> Skip FT: missing {hard_manifest}", flush=True)
        return
    cmd_ft = [
        python,
        "-u",
        "-m",
        "nafnet_denoise.train_distill",
        "--arch",
        "dual_head",
        "--use-nonlocal",
        "--nonlocal-count",
        "1",
        "--align-soft-gate",
        "--wiener-front-end",
        "--wiener-merge-frames",
        "16",
        "--wiener-tile",
        "32",
        "--wiener-overlap",
        "16",
        "--wiener-c-factor",
        "8.0",
        "--wiener-spatial",
        "--wiener-fe-schedule",
        "fps_sigma",
        "--resume",
        "nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt",
        "--real-burst-manifest",
        str(hard_manifest),
        "--epochs",
        "8",
        "--patches-per-epoch",
        "192",
        "--lr",
        "1e-6",
        "--output-dir",
        str(ft_dir),
        "--edge-weight",
        "0.35",
        "--highpass-weight",
        "0.5",
    ]
    # train_distill may have different flags — try and log
    log = meta_out / "p8_edge_hp_ft.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(cmd_ft, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    print(f">>> FT exit={code} log={log}", flush=True)
    (compare / "NOTE.txt").parent.mkdir(parents=True, exist_ok=True)
    (compare / "NOTE.txt").write_text(
        f"FT exit={code}. Score separately with compare script if checkpoint exists.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
