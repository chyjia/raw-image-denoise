"""Chain light DES schedule meta cycles (default: continue from current deploy).

Runs ``run_des_meta100_pNN`` style schedule grids for ``--cycles`` iterations,
bumping BASELINE after each bake. Stops early after ``--no-bake-patience``
consecutive cycles without a bake. Intended for zero-train postprocess search.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-cycle", type=int, default=22)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--no-bake-patience", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    repo = root.parent
    stagnant = 0
    log_path = root / "compare_meta_auto" / "cycle_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(int(args.cycles)):
        cyc = int(args.start_cycle) + i
        mod = f"nafnet_denoise.run_des_meta100_p{cyc}"
        script = root / f"run_des_meta100_p{cyc}.py"
        if not script.exists():
            # clone from last existing p22-style runner
            src = root / "run_des_meta100_p22.py"
            if not src.exists():
                raise SystemExit("missing p22 template")
            text = src.read_text(encoding="utf-8")
            # read current baseline from deploy docstring or hook
            hook = root / f"deploy_p{cyc-1}_hook.json"
            base = 0.9281
            if hook.exists():
                base = float(json.loads(hook.read_text(encoding="utf-8"))["mean_des"])
            else:
                # fall back to infer docstring
                dep = (root / "infer_blend_deploy.py").read_text(encoding="utf-8")
                m = re.search(r"mean DES ~([0-9.]+)", dep)
                if m:
                    base = float(m.group(1))
            text = text.replace("P22", f"P{cyc}").replace("p22", f"p{cyc}")
            text = re.sub(r"BASELINE = [0-9.]+", f"BASELINE = {base:.6f}", text, count=1)
            text = text.replace("compare_meta100_p22", f"compare_meta100_p{cyc}")
            text = text.replace("deploy_p22_hook", f"deploy_p{cyc}_hook")
            script.write_text(text, encoding="utf-8")
            print(f"generated {script.name} baseline={base:.4f}", flush=True)

        out_dir = root / f"compare_meta100_p{cyc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logf = out_dir / "run.log"
        cmd = [
            sys.executable,
            "-m",
            mod,
            "--patience",
            str(args.patience),
            "--min-eval",
            str(args.min_eval),
        ]
        print(f"\n######## AUTO CYCLE P{cyc} ########", flush=True)
        with logf.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        baked = (out_dir / "bake_recipe.json").exists()
        best = None
        gbest = out_dir / "global_best.json"
        if gbest.exists():
            best = json.loads(gbest.read_text(encoding="utf-8")).get("mean_des")
        rec = {
            "cycle": cyc,
            "exit": proc.returncode,
            "baked": baked,
            "best": best,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if baked:
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= int(args.no_bake_patience):
                print(
                    f"Auto-stop: {stagnant} cycles without bake",
                    flush=True,
                )
                break


if __name__ == "__main__":
    main()
