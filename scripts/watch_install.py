"""Watch the uv download cache so long installs can be verified as progressing."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "artifacts" / "logs" / "install_progress.log"
CACHE = Path.home() / "AppData" / "Local" / "uv" / "cache"


def cache_gb() -> float:
    total = 0
    for path in CACHE.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total / 2**30


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
    start = time.time()
    previous = 0.0
    with LOG.open("a", encoding="utf-8") as fh:
        while time.time() - start < seconds:
            current = cache_gb()
            line = (
                f"{datetime.now():%H:%M:%S} cache={current:.2f}GB "
                f"delta={(current - previous) * 1024:+.0f}MB"
            )
            print(line, flush=True)
            fh.write(line + "\n")
            fh.flush()
            previous = current
            time.sleep(60)


if __name__ == "__main__":
    main()
