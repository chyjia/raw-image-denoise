"""Chain P37–P40 then schedule pads; early-stop after no-bake streak."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


MODULES = [
    ("37", "nafnet_denoise.run_des_meta100_p37"),
    ("38", "nafnet_denoise.run_des_meta100_p38"),
    ("39", "nafnet_denoise.run_des_meta100_p39"),
    ("40", "nafnet_denoise.run_des_meta100_p40"),
]


def current_baseline(root: Path) -> float:
    for cyc in range(50, 30, -1):
        hook = root / f"deploy_p{cyc}_hook.json"
        if hook.exists():
            try:
                return float(json.loads(hook.read_text(encoding="utf-8"))["mean_des"])
            except Exception:
                pass
    dep = (root / "infer_blend_deploy.py").read_text(encoding="utf-8")
    m = re.search(r"mean DES ~([0-9.]+)", dep)
    return float(m.group(1)) if m else 0.9292


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-bake-patience", type=int, default=5)
    parser.add_argument("--extra-cycles", type=int, default=20)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    repo = root.parent
    log_dir = root / "compare_meta_auto"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cycle_log_p37plus.jsonl"

    baseline = current_baseline(root)
    stagnant = 0
    print(f"start baseline={baseline:.4f}", flush=True)

    for cyc, mod in MODULES:
        out_dir = root / f"compare_meta100_p{cyc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logf = out_dir / "run.log"
        cmd = [
            sys.executable, "-m", mod,
            "--baseline", f"{baseline:.6f}",
            "--patience", "8", "--min-eval", "16",
        ]
        print(f"\n######## CYCLE P{cyc} baseline={baseline:.4f} ########", flush=True)
        with logf.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT)
        baked = (out_dir / "bake_recipe.json").exists()
        best = None
        if (out_dir / "global_best.json").exists():
            best = json.loads((out_dir / "global_best.json").read_text(encoding="utf-8")).get("mean_des")
        elif baked:
            best = json.loads((out_dir / "bake_recipe.json").read_text(encoding="utf-8")).get("mean_des")
        rec = {"cycle": int(cyc), "exit": proc.returncode, "baked": baked, "best": best}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if baked and best is not None and float(best) > baseline + 1e-4:
            baseline = float(best)
            stagnant = 0
        else:
            stagnant += 1

    # Extra: clone p40 with nudged pads
    for i in range(int(args.extra_cycles)):
        if stagnant >= int(args.no_bake_patience):
            print(f"Auto-stop: {stagnant} cycles without bake", flush=True)
            break
        cyc = 41 + i
        script = root / f"run_des_meta100_p{cyc}.py"
        if not script.exists():
            text = (root / "run_des_meta100_p40.py").read_text(encoding="utf-8")
            text = text.replace("P40", f"P{cyc}").replace("p40", f"p{cyc}")
            text = re.sub(r"BASELINE = [0-9.]+", f"BASELINE = {baseline:.6f}", text)
            text = text.replace("compare_meta100_p40", f"compare_meta100_p{cyc}")
            text = text.replace("deploy_p40_hook", f"deploy_p{cyc}_hook")
            text = text.replace(
                "low_fps=1.25 + (i % 5) * 0.25",
                f"low_fps=1.25 + ((i + {cyc}) % 5) * 0.25",
            )
            script.write_text(text, encoding="utf-8")
        out_dir = root / f"compare_meta100_p{cyc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logf = out_dir / "run.log"
        cmd = [
            sys.executable, "-m", f"nafnet_denoise.run_des_meta100_p{cyc}",
            "--baseline", f"{baseline:.6f}",
            "--patience", "8", "--min-eval", "16",
        ]
        print(f"\n######## CYCLE P{cyc} baseline={baseline:.4f} ########", flush=True)
        with logf.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT)
        baked = (out_dir / "bake_recipe.json").exists()
        best = None
        if (out_dir / "global_best.json").exists():
            best = json.loads((out_dir / "global_best.json").read_text(encoding="utf-8")).get("mean_des")
        rec = {"cycle": cyc, "exit": proc.returncode, "baked": baked, "best": best}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if baked and best is not None and float(best) > baseline + 1e-4:
            baseline = float(best)
            stagnant = 0
        else:
            stagnant += 1

    print(f"done final_baseline={baseline:.4f}", flush=True)


if __name__ == "__main__":
    main()
