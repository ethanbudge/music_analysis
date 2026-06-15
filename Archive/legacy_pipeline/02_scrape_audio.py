"""
02_scrape_audio.py — Download MP3s for every song in the catalog via yt-dlp.

Prerequisites
-------------
  pip install yt-dlp
  brew install ffmpeg        # or: pip install imageio-ffmpeg

  yt-dlp uses YouTube's search API (ytsearch1:) to find the top result for
  each song's yt_query string and downloads it as 192kbps MP3.

Outputs
-------
  audio/{ARTIST_FOLDER}/{SONG_ID}.mp3

  e.g.  audio/KL/KL_01.mp3

Resume-safe
-----------
  Songs with an existing MP3 are skipped. Delete the file to re-download.

Rate-limiting
-------------
  A random 2–5 s sleep between downloads avoids triggering YouTube's
  rate-limiter. For large runs, consider spreading the download over
  multiple sessions using --artists to process one artist at a time.

IMPORTANT — Legal note
----------------------
  Downloading YouTube audio for academic research falls under fair-use
  arguments in many jurisdictions but is technically against YouTube ToS.
  This script is intended for non-commercial research purposes only.
  Do not redistribute the downloaded audio files.
"""

from __future__ import annotations
import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from config import CATALOG, AUDIO_BASE_DIR, YTDLP_CONFIG
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def build_ydl_opts(output_template: str) -> dict:
    """Build yt-dlp options dict for a given output path template."""
    opts = dict(YTDLP_CONFIG)   # shallow copy
    opts["outtmpl"] = output_template
    return opts


def song_already_downloaded(song_id: str, folder: str) -> str | None:
    """
    Return the path to the existing file if already downloaded, else None.
    Checks .mp3, .wav, .flac extensions.
    """
    base = Path(AUDIO_BASE_DIR) / folder / song_id
    for ext in [".mp3", ".wav", ".flac"]:
        p = base.with_suffix(ext)
        if p.exists() and p.stat().st_size > 100_000:  # > 100 KB — not a stub
            return str(p)
    return None


def download_song(song_id: str, yt_query: str, folder: str,
                  dry_run: bool = False) -> bool:
    """
    Download a single song as MP3.
    Returns True on success, False on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp not installed.  Run: pip install yt-dlp")
        sys.exit(1)

    out_dir = Path(AUDIO_BASE_DIR) / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp outtmpl — %(ext)s will be replaced by 'mp3' after postprocessing
    outtmpl = str(out_dir / f"{song_id}.%(ext)s")

    search_url = f"ytsearch1:{yt_query}"
    logger.info(f"  [{song_id}] Searching: {yt_query}")

    if dry_run:
        logger.info(f"    DRY RUN — would download to {outtmpl}")
        return True

    opts = build_ydl_opts(outtmpl)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_url, download=True)
            if info is None:
                raise RuntimeError("extract_info returned None")
            # For search results, info is a playlist with one entry
            if "entries" in info:
                info = info["entries"][0]
            video_title = info.get("title", "unknown")
            duration    = info.get("duration", 0)
            logger.info(f"    ✓ Downloaded: '{video_title}' ({duration}s)")
            return True
    except Exception as e:
        logger.warning(f"    ✗ FAILED: {e}")
        return False


def download_all(dry_run: bool = False, artist_filter: list[str] | None = None,
                 max_retries: int = 2):
    """
    Download all songs in the catalog that don't already have an MP3.

    Parameters
    ----------
    dry_run       : Print what would be downloaded, don't actually download.
    artist_filter : Restrict to these artist codes.
    max_retries   : Retry failed downloads this many times before giving up.
    """
    manifest = {
        "downloaded": [],
        "skipped":    [],
        "failed":     [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    for artist_code, artist_data in CATALOG.items():
        if artist_filter and artist_code not in artist_filter:
            continue

        folder = artist_data["folder"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"Artist: {artist_data['name']}  ({artist_code})")

        for song_id, song_meta in artist_data["songs"].items():

            # ── Resume: skip if already downloaded ────────────────────────
            existing = song_already_downloaded(song_id, folder)
            if existing:
                logger.info(f"  [{song_id}] SKIP — {existing}")
                manifest["skipped"].append(song_id)
                continue

            # ── Download with retries ──────────────────────────────────────
            success = False
            for attempt in range(1, max_retries + 2):
                success = download_song(
                    song_id   = song_id,
                    yt_query  = song_meta["yt_query"],
                    folder    = folder,
                    dry_run   = dry_run,
                )
                if success:
                    break
                if attempt <= max_retries:
                    wait = random.uniform(5, 10)
                    logger.info(f"    Retry {attempt}/{max_retries} in {wait:.1f}s…")
                    time.sleep(wait)

            if success:
                manifest["downloaded"].append(song_id)
            else:
                manifest["failed"].append(song_id)
                # Create a tiny stub so we know it failed
                stub = Path(AUDIO_BASE_DIR) / folder / f"{song_id}.FAILED"
                stub.touch()

            if not dry_run:
                sleep_s = random.uniform(
                    YTDLP_CONFIG["min_sleep_interval"],
                    YTDLP_CONFIG["max_sleep_interval"],
                )
                logger.debug(f"  Sleeping {sleep_s:.1f}s…")
                time.sleep(sleep_s)

    # ── Write manifest ─────────────────────────────────────────────────────
    manifest["n_downloaded"] = len(manifest["downloaded"])
    manifest["n_skipped"]    = len(manifest["skipped"])
    manifest["n_failed"]     = len(manifest["failed"])

    manifest_path = Path(AUDIO_BASE_DIR) / "_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n{'═'*60}")
    logger.info(f"Done. Downloaded: {manifest['n_downloaded']}  |  "
                f"Skipped: {manifest['n_skipped']}  |  Failed: {manifest['n_failed']}")
    if manifest["failed"]:
        logger.warning(f"Failed songs (check .FAILED stubs): {manifest['failed']}")
    logger.info(f"Manifest → {manifest_path}")

    return manifest


def get_audio_path(song_id: str, folder: str) -> str | None:
    """
    Return path to audio file for a song, or None if not found.
    Checks .mp3, .wav, .flac in order.
    """
    base = Path(AUDIO_BASE_DIR) / folder / song_id
    for ext in [".mp3", ".wav", ".flac"]:
        p = base.with_suffix(ext)
        if p.exists() and p.stat().st_size > 100_000:
            return str(p)
    return None


def get_coverage_report() -> dict:
    """Return a dict summarising which songs have audio vs. not."""
    report = {"have_audio": [], "missing": [], "failed_stubs": []}
    for artist_code, artist_data in CATALOG.items():
        folder = artist_data["folder"]
        for song_id in artist_data["songs"]:
            if get_audio_path(song_id, folder):
                report["have_audio"].append(song_id)
            elif (Path(AUDIO_BASE_DIR) / folder / f"{song_id}.FAILED").exists():
                report["failed_stubs"].append(song_id)
            else:
                report["missing"].append(song_id)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio via yt-dlp")
    parser.add_argument("--dry-run",  action="store_true", help="Show what would be fetched")
    parser.add_argument("--artists",  nargs="*", help="Artist codes (default: all)")
    parser.add_argument("--report",   action="store_true", help="Print coverage report and exit")
    args = parser.parse_args()

    if args.report:
        rep = get_coverage_report()
        print(f"Have audio : {len(rep['have_audio'])}")
        print(f"Missing    : {len(rep['missing'])}")
        print(f"Failed     : {len(rep['failed_stubs'])}")
        if rep["missing"]:
            print("Missing:", rep["missing"])
    else:
        download_all(dry_run=args.dry_run, artist_filter=args.artists or None)
