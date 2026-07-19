"""Wait for NAFNet training to finish, then run validation automatically."""

from __future__ import annotations

from nafnet_denoise.validate import main

if __name__ == "__main__":
    import sys

    if "--wait" not in sys.argv:
        sys.argv.insert(1, "--wait")
    main()
