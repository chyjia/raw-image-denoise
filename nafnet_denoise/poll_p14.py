"""Poll P14 history until the runner exits."""
from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None


def p14_alive() -> bool:
    if psutil is None:
        import subprocess

        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline,processid"],
            capture_output=True,
            text=True,
            errors="ignore",
        )
        return "meta100_p14" in (r.stdout or "")
    for proc in psutil.process_iter(["cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if "meta100_p14" in cmd:
            return True
    return False


def main() -> None:
    hist = Path("nafnet_denoise/compare_meta100_p14/history.json")
    while True:
        if hist.exists():
            h = json.loads(hist.read_text(encoding="utf-8"))
            b = h.get("best_safe") or {}
            last = h["history"][-1]
            print(
                time.strftime("%H:%M:%S"),
                "n",
                len(h["history"]),
                "last",
                last.get("name"),
                round(float(last.get("mean_des") or 0), 6),
                "best",
                b.get("name"),
                round(float(b.get("mean_des") or 0), 6),
                flush=True,
            )
        else:
            print(time.strftime("%H:%M:%S"), "waiting history", flush=True)
        if not p14_alive():
            print("P14 gone", flush=True)
            break
        time.sleep(90)


if __name__ == "__main__":
    main()
