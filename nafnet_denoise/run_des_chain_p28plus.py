"""Chain DES meta cycles P28+ with diverse levers; early-stop on no-bake streak.

Updates baseline from bake hooks / deploy docstring. Does not ask for confirmation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


MODULES = [
    ("28", "nafnet_denoise.run_des_meta100_p28"),
    ("29", "nafnet_denoise.run_des_meta100_p29"),
    ("30", "nafnet_denoise.run_des_meta100_p30"),
    ("31", "nafnet_denoise.run_des_meta100_p31"),
]


def current_baseline(root: Path) -> float:
    for cyc in range(50, 21, -1):
        hook = root / f"deploy_p{cyc}_hook.json"
        if hook.exists():
            try:
                return float(json.loads(hook.read_text(encoding="utf-8"))["mean_des"])
            except Exception:
                pass
    dep = (root / "infer_blend_deploy.py").read_text(encoding="utf-8")
    m = re.search(r"mean DES ~([0-9.]+)", dep)
    return float(m.group(1)) if m else 0.9290


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-bake-patience", type=int, default=5)
    parser.add_argument("--extra-schedule-cycles", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    repo = root.parent
    log_dir = root / "compare_meta_auto"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cycle_log_p28plus.jsonl"

    stagnant = 0
    baseline = current_baseline(root)
    print(f"start baseline={baseline:.4f}", flush=True)

    # Fixed diverse cycles first
    for cyc, mod in MODULES:
        out_dir = root / f"compare_meta100_p{cyc}"
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
        print(f"\n######## CYCLE P{cyc} baseline={baseline:.4f} ########", flush=True)
        with logf.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        baked = (out_dir / "bake_recipe.json").exists()
        best = None
        gbest = out_dir / "global_best.json"
        if gbest.exists():
            best = json.loads(gbest.read_text(encoding="utf-8")).get("mean_des")
        elif (out_dir / "bake_recipe.json").exists():
            best = json.loads(
                (out_dir / "bake_recipe.json").read_text(encoding="utf-8")
            ).get("mean_des")
        rec = {"cycle": int(cyc), "exit": proc.returncode, "baked": baked, "best": best}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if baked and best is not None:
            baseline = float(best)
            stagnant = 0
        else:
            stagnant += 1

    # Extra schedule-neighborhood cycles via p22 template
    for i in range(int(args.extra_schedule_cycles)):
        if stagnant >= int(args.no_bake_patience):
            break
        cyc = 31 + i
        # generate thin wrapper
        script = root / f"run_des_meta100_p{cyc}.py"
        if not script.exists():
            text = (root / "run_des_meta100_p22.py").read_text(encoding="utf-8")
            text = text.replace("P22", f"P{cyc}").replace("p22", f"p{cyc}")
            text = re.sub(r"BASELINE = [0-9.]+", f"BASELINE = {baseline:.6f}", text)
            text = text.replace("compare_meta100_p22", f"compare_meta100_p{cyc}")
            text = text.replace("deploy_p22_hook", f"deploy_p{cyc}_hook")
            # nudge pad grid by cycle index
            text = text.replace(
                "u_low=0.18 + (i % 6) * 0.01",
                f"u_low=0.18 + ((i + {cyc}) % 6) * 0.01",
            )
            text = text.replace(
                "low_bilat=0.8 + (i % 3) * 0.05",
                f"low_bilat=0.8 + ((i + {cyc}) % 4) * 0.04",
            )
            script.write_text(text, encoding="utf-8")
        out_dir = root / f"compare_meta100_p{cyc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logf = out_dir / "run.log"
        cmd = [
            sys.executable,
            "-m",
            f"nafnet_denoise.run_des_meta100_p{cyc}",
            "--patience",
            "8",
            "--min-eval",
            "20",
        ]
        print(f"\n######## CYCLE P{cyc} baseline={baseline:.4f} ########", flush=True)
        with logf.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        baked = (out_dir / "bake_recipe.json").exists()
        best = None
        if (out_dir / "global_best.json").exists():
            best = json.loads(
                (out_dir / "global_best.json").read_text(encoding="utf-8")
            ).get("mean_des")
        rec = {"cycle": cyc, "exit": proc.returncode, "baked": baked, "best": best}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if baked and best is not None and float(best) > baseline + 1e-4:
            baseline = float(best)
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= int(args.no_bake_patience):
            print(f"Auto-stop: {stagnant} cycles without bake", flush=True)
            break

    print(f"done final_baseline={baseline:.4f}", flush=True)


if __name__ == "__main__":
    main()
