"""Resume P148+ chain after reboot interrupt (from P155, stagnant=4)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

EXTRA_TEMPLATES = [
    "run_des_meta100_p148.py",
    "run_des_meta100_p149.py",
    "run_des_meta100_p150.py",
    "run_des_meta100_p151.py",
]


def run_one(repo: Path, mod: str, baseline: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = out_dir / "run.log"
    # clear stale bake from interrupted/incomplete cycles
    for name in ("bake_recipe.json",):
        p = out_dir / name
        if p.exists():
            p.unlink()
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
    parser.add_argument("--start-cycle", type=int, default=155)
    parser.add_argument("--stagnant", type=int, default=4)
    parser.add_argument("--no-bake-patience", type=int, default=5)
    parser.add_argument("--extra-cycles", type=int, default=100)
    parser.add_argument("--baseline", type=float, default=0.9589)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parent
    log_dir = root / "compare_meta_auto"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cycle_log_p148plus.jsonl"
    baseline = float(args.baseline)
    stagnant = int(args.stagnant)
    print(f"resume start_cycle={args.start_cycle} baseline={baseline:.4f} stagnant={stagnant}", flush=True)

    for i in range(int(args.extra_cycles)):
        if stagnant >= int(args.no_bake_patience):
            print(f"Auto-stop: {stagnant} cycles without bake", flush=True)
            break
        cyc = int(args.start_cycle) + i
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
