"""Wait for a P12 runner PID (optional) then launch P13. Used to chain meta loops."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--p12-done-marker", type=Path, default=None)
    args = parser.parse_args()

    if args.wait_pid:
        print(f"Waiting for PID {args.wait_pid} ...", flush=True)
        while True:
            try:
                import os

                os.kill(args.wait_pid, 0)
            except OSError:
                break
            time.sleep(30)
        print(f"PID {args.wait_pid} exited", flush=True)

    # Prefer marker written at end of P12
    marker = args.p12_done_marker or Path("nafnet_denoise/compare_meta100_p12/DONE")
    # Always run P13 (idempotent via --start / reuse ckpts)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nafnet_denoise.run_des_meta100_p13",
        "--max-recipes",
        "100",
        "--patience",
        "8",
        "--min-eval",
        "15",
    ]
    print("Starting P13:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    marker.write_text(f"p13_rc={rc}\n", encoding="utf-8")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
