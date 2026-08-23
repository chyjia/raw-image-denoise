"""Chain P51+ new levers; early-stop after no-bake streak."""



from __future__ import annotations



import argparse

import json

import re

import subprocess

import sys

from pathlib import Path



# Import shared runner pieces from p46plus

from nafnet_denoise.run_des_chain_p46plus import current_baseline, run_one



MODULES = [

    ("51", "nafnet_denoise.run_des_meta100_p51"),

    ("52", "nafnet_denoise.run_des_meta100_p52"),

]



EXTRA_TEMPLATES = [

    "run_des_meta100_p52.py",

    "run_des_meta100_p51.py",

    "run_des_meta100_p50.py",

    "run_des_meta100_p49.py",

]





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--no-bake-patience", type=int, default=5)

    parser.add_argument("--extra-cycles", type=int, default=20)

    args = parser.parse_args()



    root = Path(__file__).resolve().parent

    repo = root.parent

    log_dir = root / "compare_meta_auto"

    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "cycle_log_p51plus.jsonl"



    baseline = current_baseline(root)

    stagnant = 0

    print(f"start baseline={baseline:.4f}", flush=True)



    for cyc, mod in MODULES:

        print(f"\n######## CYCLE P{cyc} baseline={baseline:.4f} ########", flush=True)

        out_dir = root / f"compare_meta100_p{cyc}"

        rec = {"cycle": int(cyc), **run_one(repo, mod, baseline, out_dir)}

        with log_path.open("a", encoding="utf-8") as handle:

            handle.write(json.dumps(rec) + "\n")

        print(rec, flush=True)

        if rec["baked"] and rec["best"] is not None and float(rec["best"]) > baseline + 1e-4:

            baseline = float(rec["best"])

            stagnant = 0

        else:

            stagnant += 1

        if stagnant >= int(args.no_bake_patience):

            print(f"Auto-stop: {stagnant} cycles without bake", flush=True)

            print(f"done final_baseline={baseline:.4f}", flush=True)

            return



    for i in range(int(args.extra_cycles)):

        if stagnant >= int(args.no_bake_patience):

            break

        cyc = 53 + i

        src = root / EXTRA_TEMPLATES[i % len(EXTRA_TEMPLATES)]

        text = src.read_text(encoding="utf-8")

        old = re.search(r"run_des_meta100_p(\d+)", src.name)

        old_n = old.group(1) if old else "52"

        text = text.replace(f"P{old_n}", f"P{cyc}").replace(f"p{old_n}", f"p{cyc}")

        text = text.replace(f"compare_meta100_p{old_n}", f"compare_meta100_p{cyc}")

        text = text.replace(f"deploy_p{old_n}_hook", f"deploy_p{cyc}_hook")

        # perturb pads slightly so extras aren't identical

        text = text.replace("pad_", f"pad{cyc}_")

        (root / f"run_des_meta100_p{cyc}.py").write_text(text, encoding="utf-8")

        print(f"\n######## CYCLE P{cyc} baseline={baseline:.4f} ########", flush=True)

        out_dir = root / f"compare_meta100_p{cyc}"

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

        if stagnant >= int(args.no_bake_patience):

            print(f"Auto-stop: {stagnant} cycles without bake", flush=True)

            break



    print(f"done final_baseline={baseline:.4f}", flush=True)





if __name__ == "__main__":

    main()


