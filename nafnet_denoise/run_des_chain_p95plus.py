"""Chain P95+ DES cycles; early-stop after consecutive no-bake (default 5)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

MODULES = [
    ("95", "nafnet_denoise.run_des_meta100_p95"),
    ("96", "nafnet_denoise.run_des_meta100_p96"),
    ("97", "nafnet_denoise.run_des_meta100_p97"),
    ("98", "nafnet_denoise.run_des_meta100_p98"),
    ("99", "nafnet_denoise.run_des_meta100_p99"),
]

EXTRA_TEMPLATES = [
    "run_des_meta100_p95.py",
    "run_des_meta100_p96.py",
    "run_des_meta100_p97.py",
    "run_des_meta100_p98.py",
]


def current_baseline(root: Path) -> float:
    for cyc in range(120, 39, -1):
        hook = root / f"deploy_p{cyc}_hook.json"
        if hook.exists():
            try:
                return float(json.loads(hook.read_text(encoding="utf-8"))["mean_des"])
            except Exception:
                pass
    dep = (root / "infer_blend_deploy.py").read_text(encoding="utf-8")
    m = re.search(r"mean DES ~([0-9.]+)", dep)
    return float(m.group(1)) if m else 0.9442


def run_one(repo: Path, mod: str, baseline: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = out_dir / "run.log"
    # P99 weight-TTT: smaller patience already in script defaults
    patience = "8" if "p99" in mod else "10"
    min_eval = "12" if "p99" in mod else "20"
    cmd = [
        sys.executable,
        "-m",
        mod,
        "--baseline",
        f"{baseline:.6f}",
        "--patience",
        patience,
        "--min-eval",
        min_eval,
    ]
    with logf.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    baked = (out_dir / "bake_recipe.json").exists()
    best = None
    for name in ("bake_recipe.json", "global_best.json"):
        p = out_dir / name
        if p.exists():
            try:
                best = json.loads(p.read_text(encoding="utf-8")).get("mean_des")
                break
            except Exception:
                pass
    return {"exit": proc.returncode, "baked": baked, "best": best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-bake-patience", type=int, default=5)
    parser.add_argument("--extra-cycles", type=int, default=15)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parent
    log_dir = root / "compare_meta_auto"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cycle_log_p95plus.jsonl"
    baseline = current_baseline(root)
    stagnant = 0
    print(f"start baseline={baseline:.4f}", flush=True)

    for cyc, mod in MODULES:
        out_dir = root / f"compare_meta100_p{cyc}"
        print(f"\n######## CYCLE P{cyc} baseline={baseline:.4f} ########", flush=True)
        rec = {"cycle": int(cyc), **run_one(repo, mod, baseline, out_dir)}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if rec["baked"] and rec["best"] is not None and float(rec["best"]) > baseline + 1e-4:
            baseline = float(rec["best"])
            stagnant = 0
        else:
            stagnant += 1

    for i in range(int(args.extra_cycles)):
        if stagnant >= int(args.no_bake_patience):
            print(f"Auto-stop: {stagnant} cycles without bake", flush=True)
            break
        cyc = 100 + i
        tmpl = root / EXTRA_TEMPLATES[i % len(EXTRA_TEMPLATES)]
        script = root / f"run_des_meta100_p{cyc}.py"
        if not script.exists():
            text = tmpl.read_text(encoding="utf-8")
            src_n = re.search(r"p(\d+)", tmpl.name).group(1)
            text = text.replace(f"P{src_n}", f"P{cyc}").replace(f"p{src_n}", f"p{cyc}")
            text = text.replace(f"compare_meta100_p{src_n}", f"compare_meta100_p{cyc}")
            text = text.replace(f"deploy_p{src_n}_hook", f"deploy_p{cyc}_hook")
            script.write_text(text, encoding="utf-8")
        out_dir = root / f"compare_meta100_p{cyc}"
        print(f"\n######## CYCLE P{cyc} baseline={baseline:.4f} ########", flush=True)
        rec = {
            "cycle": cyc,
            **run_one(repo, f"nafnet_denoise.run_des_meta100_p{cyc}", baseline, out_dir),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if rec["baked"] and rec["best"] is not None and float(rec["best"]) > baseline + 1e-4:
            baseline = float(rec["best"])
            stagnant = 0
        else:
            stagnant += 1

    print(f"done final_baseline={baseline:.4f}", flush=True)


if __name__ == "__main__":
    main()
