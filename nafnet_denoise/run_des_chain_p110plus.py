"""Chain P110+ DES cycles; early-stop after consecutive no-bake (default 5)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

MODULES = [
    ("110", "nafnet_denoise.run_des_meta100_p110"),
    ("111", "nafnet_denoise.run_des_meta100_p111"),
    ("112", "nafnet_denoise.run_des_meta100_p112"),
    ("113", "nafnet_denoise.run_des_meta100_p113"),
]

EXTRA_TEMPLATES = [
    "run_des_meta100_p110.py",
    "run_des_meta100_p111.py",
    "run_des_meta100_p112.py",
    "run_des_meta100_p113.py",
]


def current_baseline(root: Path) -> float:
    for cyc in range(140, 39, -1):
        hook = root / f"deploy_p{cyc}_hook.json"
        if hook.exists():
            try:
                return float(json.loads(hook.read_text(encoding="utf-8"))["mean_des"])
            except Exception:
                pass
    dep = (root / "infer_blend_deploy.py").read_text(encoding="utf-8")
    m = re.search(r"mean DES ~([0-9.]+)", dep)
    return float(m.group(1)) if m else 0.9506


def run_one(repo: Path, mod: str, baseline: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = out_dir / "run.log"
    cmd = [
        sys.executable,
        "-m",
        mod,
        "--baseline",
        f"{baseline:.6f}",
        "--patience",
        "10",
        "--min-eval",
        "20",
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
    parser.add_argument("--extra-cycles", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parent
    log_dir = root / "compare_meta_auto"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cycle_log_p110plus.jsonl"
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
        cyc = 114 + i
        tmpl = root / EXTRA_TEMPLATES[i % len(EXTRA_TEMPLATES)]
        script = root / f"run_des_meta100_p{cyc}.py"
        if not script.exists():
            text = tmpl.read_text(encoding="utf-8")
            src_n = re.search(r"p(\d+)", tmpl.name).group(1)
            text = text.replace(f"compare_meta100_p{src_n}", f"compare_meta100_p{cyc}")
            text = text.replace(f"deploy_p{src_n}_hook", f"deploy_p{cyc}_hook")
            text = text.replace(f'cycle="P{src_n}"', f'cycle="P{cyc}"')
            text = text.replace(f"P{src_n}:", f"P{cyc}:")
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
