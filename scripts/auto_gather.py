#!/usr/bin/env python3
"""
auto_gather.py — command-line entry point for the scheduled song-gathering
loop (pipeline steps 1-4: sample -> audio -> popularity -> mood/MERT/chorus).

    python scripts/auto_gather.py status
    python scripts/auto_gather.py enable
    python scripts/auto_gather.py disable
    python scripts/auto_gather.py set-hours 1 6      # local time, 24h clock
    python scripts/auto_gather.py run                # leave running; Ctrl-C to stop
    python scripts/auto_gather.py run --once          # one batch now, for testing

The scheduler starts disabled and does nothing until `enable` is called --
see README.md "Auto-gathering on a schedule" for the full walkthrough,
including how to leave `run` going in the background across terminal
sessions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmc.scheduler import main  # noqa: E402

if __name__ == "__main__":
    main()
