"""Chain P46–P52 then varied extras; early-stop after no-bake streak.



Default: patience=5 no-bake → stop (user: loop ~100, early-stop OK).

"""



from __future__ import annotations



import argparse

import json

import re

import subprocess

import sys

from pathlib import Path





MODULES = [

    ("46", "nafnet_denoise.run_des_meta100_p46"),

    ("47", "nafnet_denoise.run_des_meta100_p47"),

    ("48", "nafnet_denoise.run_des_meta100_p48"),

    ("49", "nafnet_denoise.run_des_meta100_p49"),

    ("50", "nafnet_denoise.run_des_meta100_p50"),

    ("51", "nafnet_denoise.run_des_meta100_p51"),

    ("52", "nafnet_denoise.run_des_meta100_p52"),

]



# Extra cycle templates (rotate) — avoid pure P48 schedule clones.

EXTRA_TEMPLATES = [

    "nafnet_denoise/run_des_meta100_p50.py",

    "nafnet_denoise/run_des_meta100_p52.py",

    "nafnet_denoise/run_des_meta100_p49.py",

    "nafnet_denoise/run_des_meta100_p47.py",

]





def current_baseline(root: Path) -> float:

    for cyc in range(80, 39, -1):

        hook = root / f"deploy_p{cyc}_hook.json"

        if hook.exists():

            try:

                return float(json.loads(hook.read_text(encoding="utf-8"))["mean_des"])

            except Exception:

                pass

    dep = (root / "infer_blend_deploy.py").read_text(encoding="utf-8")

    m = re.search(r"mean DES ~([0-9.]+)", dep)

    return float(m.group(1)) if m else 0.9294





def run_one(repo: Path, mod: str, baseline: float, out_dir: Path, patience: int = 8) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    logf = out_dir / "run.log"

    # clear prior bake marker so re-runs are honest

    bake = out_dir / "bake_recipe.json"

    if bake.exists():

        bake.unlink()

    cmd = [

        sys.executable,

        "-m",

        mod,

        "--baseline",

        f"{baseline:.6f}",

        "--patience",

        str(patience),

        "--min-eval",

        "16",

        "--output-dir",

        str(out_dir),

    ]

    with logf.open("w", encoding="utf-8") as handle:

        proc = subprocess.run(cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT)

    baked = bake.exists()

    best = None

    if (out_dir / "global_best.json").exists():

        best = json.loads((out_dir / "global_best.json").read_text(encoding="utf-8")).get(

            "mean_des"

        )

    elif baked:

        best = json.loads(bake.read_text(encoding="utf-8")).get("mean_des")

    return {"exit": proc.returncode, "baked": baked, "best": best}





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--no-bake-patience", type=int, default=5)

    parser.add_argument("--extra-cycles", type=int, default=30)

    parser.add_argument("--start-from", type=int, default=46)

    args = parser.parse_args()



    root = Path(__file__).resolve().parent

    repo = root.parent

    log_dir = root / "compare_meta_auto"

    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "cycle_log_p46plus.jsonl"



    baseline = current_baseline(root)

    stagnant = 0

    print(f"start baseline={baseline:.4f}", flush=True)



    for cyc, mod in MODULES:

        if int(cyc) < int(args.start_from):

            continue

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

        tmpl = Path(EXTRA_TEMPLATES[i % len(EXTRA_TEMPLATES)])

        src = root / tmpl.name if not tmpl.is_absolute() else tmpl

        if not src.exists():

            src = root / "run_des_meta100_p48.py"

        text = src.read_text(encoding="utf-8")

        # rewrite cycle id in docstring/strings

        old = re.search(r"P(\d+)", text)

        old_n = old.group(1) if old else "48"

        text = text.replace(f"P{old_n}", f"P{cyc}").replace(f"p{old_n}", f"p{cyc}")

        text = text.replace(f"compare_meta100_p{old_n}", f"compare_meta100_p{cyc}")

        text = text.replace(f"deploy_p{old_n}_hook", f"deploy_p{cyc}_hook")

        script = root / f"run_des_meta100_p{cyc}.py"

        script.write_text(text, encoding="utf-8")

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


