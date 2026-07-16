"""
scheduler.py — an on/off, hours-of-day gated loop around gather.run_batch(),
so the data-gathering stages (1-4 of notebooks/pipeline.ipynb) can be left
running unattended and only do work during an allowed window each day.

State (enabled flag + allowed hours) is persisted to a small JSON file under
DATA_DIR so the toggle survives across `run` invocations -- flip it with the
`enable` / `disable` / `set-hours` commands, then start (or leave running) a
`run` loop; it wakes up periodically, checks the toggle and the clock, and
only calls run_batch() when both say go.

Command-line entry point: see scripts/auto_gather.py at the repo root.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import config
from .gather import DEFAULT_BATCH_SIZE, run_batch

logger = logging.getLogger(__name__)

STATE_PATH = config.DATA_DIR / "scheduler_state.json"

DEFAULT_START_HOUR = 1   # 1am
DEFAULT_END_HOUR = 6     # 6am -- an overnight window is just a sane default;
                          # change it any time with `set-hours`.
DEFAULT_POLL_SECONDS = 60


@dataclass
class SchedulerState:
    enabled: bool = False           # off by default -- must be explicitly turned on
    start_hour: int = DEFAULT_START_HOUR
    end_hour: int = DEFAULT_END_HOUR

    @staticmethod
    def load(path: Path = STATE_PATH) -> "SchedulerState":
        if not path.exists():
            return SchedulerState()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Scheduler state at %s is unreadable; using defaults.", path)
            return SchedulerState()
        return SchedulerState(
            enabled=bool(data.get("enabled", False)),
            start_hour=int(data.get("start_hour", DEFAULT_START_HOUR)),
            end_hour=int(data.get("end_hour", DEFAULT_END_HOUR)),
        )

    def save(self, path: Path = STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def in_window(hour: int, start_hour: int, end_hour: int) -> bool:
    """
    Whether `hour` (0-23) falls in the [start_hour, end_hour) window. Handles
    windows that wrap past midnight (e.g. start=22, end=6 covers 22,23,0..5).
    start_hour == end_hour means "always on" (a 24-hour window).
    """
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def status(path: Path = STATE_PATH) -> dict:
    s = SchedulerState.load(path)
    now = datetime.now()
    return {
        "enabled": s.enabled,
        "start_hour": s.start_hour,
        "end_hour": s.end_hour,
        "current_hour": now.hour,
        "would_run_now": s.enabled and in_window(now.hour, s.start_hour, s.end_hour),
    }


def enable(path: Path = STATE_PATH) -> SchedulerState:
    s = SchedulerState.load(path)
    s.enabled = True
    s.save(path)
    return s


def disable(path: Path = STATE_PATH) -> SchedulerState:
    s = SchedulerState.load(path)
    s.enabled = False
    s.save(path)
    return s


def set_hours(start_hour: int, end_hour: int, path: Path = STATE_PATH) -> SchedulerState:
    if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
        raise ValueError("hours must be in 0..23")
    s = SchedulerState.load(path)
    s.start_hour = start_hour
    s.end_hour = end_hour
    s.save(path)
    return s


def run(
    batch_size: int = DEFAULT_BATCH_SIZE,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    once: bool = False,
    max_iterations: int | None = None,
    path: Path = STATE_PATH,
) -> None:
    """
    Foreground loop: while the toggle is on and the current hour is within
    the configured window, run one batch, then immediately check again (so a
    whole open window is spent working through batches back-to-back). While
    off, or outside the window, sleep `poll_seconds` and recheck. Intended to
    be left running (e.g. under `nohup` / `screen` / a login item) for as
    long as the device is on -- it does no gathering at all until `enable`
    has been called.

    `once=True` runs a single batch immediately regardless of the toggle or
    the clock, for manual/interactive testing.
    """
    if once:
        logger.info("Running a single on-demand batch (ignoring toggle/hours).")
        run_batch(batch_size=batch_size)
        return

    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        s = SchedulerState.load(path)
        now = datetime.now()
        if s.enabled and in_window(now.hour, s.start_hour, s.end_hour):
            logger.info("Within window (hour=%d, enabled) — running a batch.", now.hour)
            run_batch(batch_size=batch_size)
        else:
            logger.debug(
                "Idle (enabled=%s, hour=%d, window=%d-%d) — sleeping %ds.",
                s.enabled, now.hour, s.start_hour, s.end_hour, poll_seconds,
            )
            time.sleep(poll_seconds)
        iterations += 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auto_gather",
        description="Toggle and run the scheduled song-gathering batches (pipeline steps 1-4).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show whether the scheduler is on and its hours window.")
    sub.add_parser("enable", help="Turn the scheduler on.")
    sub.add_parser("disable", help="Turn the scheduler off.")

    hours = sub.add_parser("set-hours", help="Set the daily hours window (local time, 24h clock).")
    hours.add_argument("start_hour", type=int, help="Window start hour, 0-23 (inclusive).")
    hours.add_argument("end_hour", type=int, help="Window end hour, 0-23 (exclusive).")

    run_p = sub.add_parser("run", help="Start the scheduling loop (leave this running).")
    run_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run_p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    run_p.add_argument("--once", action="store_true",
                        help="Run a single batch immediately and exit, ignoring the toggle/hours.")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "status":
        print(json.dumps(status(), indent=2))
    elif args.command == "enable":
        enable()
        print("Scheduler enabled.")
    elif args.command == "disable":
        disable()
        print("Scheduler disabled.")
    elif args.command == "set-hours":
        s = set_hours(args.start_hour, args.end_hour)
        print(f"Window set to {s.start_hour}:00-{s.end_hour}:00.")
    elif args.command == "run":
        run(batch_size=args.batch_size, poll_seconds=args.poll_seconds, once=args.once)


if __name__ == "__main__":
    main()
