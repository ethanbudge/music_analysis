"""
01_scrape_lyrics.py — Scrape lyrics for every song in the catalog via Genius API.

Prerequisites
-------------
  pip install lyricsgenius

  export GENIUS_API_TOKEN="your_token"   # https://genius.com/api-clients

Outputs
-------
  lyrics/{SONG_ID}.json   for each song, containing:
    {
      "song_id":      "KL_01",
      "title":        "HUMBLE.",
      "artist":       "Kendrick Lamar",
      "lyrics":       "...(full lyrics with [Section] headers)...",
      "lyrics_clean": "...(section headers stripped)...",
      "genius_url":   "https://genius.com/...",
      "fetched_at":   "2025-..."
    }

  lyrics/_manifest.json   summary of all fetched / missing songs

Notes on section headers
------------------------
  lyricsgenius by default strips [Verse], [Chorus] etc. We configure it NOT
  to strip them (remove_section_headers=False) so they are available for the
  segment analysis in step 07. The 'lyrics_clean' field stores a version with
  headers stripped, used for track-level embedding.

Resume-safe
-----------
  Already-fetched songs are skipped. Delete the JSON to re-fetch.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import CATALOG, GENIUS_API_TOKEN, GENIUS_CONFIG, LYRICS_DIR
from utils import setup_logging, strip_section_headers

setup_logging()
logger = logging.getLogger(__name__)


def build_genius_client():
    try:
        import lyricsgenius
    except ImportError:
        logger.error("lyricsgenius not installed.  Run: pip install lyricsgenius")
        sys.exit(1)

    if GENIUS_API_TOKEN in ("", "YOUR_GENIUS_TOKEN_HERE"):
        logger.error("GENIUS_API_TOKEN not set.  export GENIUS_API_TOKEN=your_token")
        sys.exit(1)

    genius = lyricsgenius.Genius(GENIUS_API_TOKEN)
    genius.sleep_time             = GENIUS_CONFIG["sleep_time"]
    genius.timeout                = GENIUS_CONFIG["timeout"]
    genius.retries                = GENIUS_CONFIG["retries"]
    genius.remove_section_headers = GENIUS_CONFIG["remove_section_headers"]
    genius.skip_non_songs         = True
    genius.verbose                = False
    return genius


def fetch_lyrics_for_song(genius, song_id: str, title: str, artist_name: str,
                           genius_query: str) -> dict | None:
    """
    Try to fetch lyrics from Genius. Uses genius_query as primary search,
    falls back to '{title} {artist_name}' if that fails.
    Returns a result dict or None on failure.
    """
    queries = [genius_query, f"{title} {artist_name}"]
    for q in queries:
        try:
            song = genius.search_song(q, get_full_info=False)
            if song and song.lyrics:
                raw_lyrics   = song.lyrics
                clean_lyrics = strip_section_headers(raw_lyrics)
                return {
                    "song_id":      song_id,
                    "title":        title,
                    "artist":       artist_name,
                    "lyrics":       raw_lyrics,
                    "lyrics_clean": clean_lyrics,
                    "genius_url":   getattr(song, "url", ""),
                    "fetched_at":   datetime.now(timezone.utc).isoformat(),
                    "query_used":   q,
                }
        except Exception as e:
            logger.warning(f"  Query '{q}' failed: {e}")
            time.sleep(1)
    return None


def scrape_all(dry_run: bool = False, artist_filter: list[str] | None = None):
    """
    Iterate over the catalog and fetch lyrics for every song not already saved.

    Parameters
    ----------
    dry_run       : Print what would be fetched, don't actually fetch.
    artist_filter : List of artist codes to restrict scraping (e.g. ['KL', 'TS']).
    """
    out_dir = Path(LYRICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    genius = None if dry_run else build_genius_client()

    manifest  = {"fetched": [], "skipped": [], "failed": []}
    total     = 0
    processed = 0

    for artist_code, artist_data in CATALOG.items():
        if artist_filter and artist_code not in artist_filter:
            continue

        artist_name = artist_data["name"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"Artist: {artist_name}  ({artist_code})")

        for song_id, song_meta in artist_data["songs"].items():
            total += 1
            out_path = out_dir / f"{song_id}.json"

            # ── Resume: skip already-fetched ──────────────────────────────
            if out_path.exists():
                logger.info(f"  [{song_id}] SKIP (already fetched)")
                manifest["skipped"].append(song_id)
                continue

            title = song_meta["title"]
            logger.info(f"  [{song_id}] Fetching: {title}")

            if dry_run:
                logger.info(f"    DRY RUN — would search: {song_meta['genius_query']}")
                continue

            result = fetch_lyrics_for_song(
                genius     = genius,
                song_id    = song_id,
                title      = title,
                artist_name = artist_name,
                genius_query = song_meta["genius_query"],
            )

            if result:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                word_count = len(result["lyrics_clean"].split())
                logger.info(f"    ✓ Saved ({word_count} words)")
                manifest["fetched"].append(song_id)
            else:
                logger.warning(f"    ✗ FAILED — no lyrics found for {title}")
                manifest["failed"].append(song_id)

            processed += 1
            time.sleep(GENIUS_CONFIG["sleep_time"])

    # ── Write manifest ─────────────────────────────────────────────────────
    manifest_path = out_dir / "_manifest.json"
    manifest["total"]       = total
    manifest["n_fetched"]   = len(manifest["fetched"])
    manifest["n_skipped"]   = len(manifest["skipped"])
    manifest["n_failed"]    = len(manifest["failed"])
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n{'═'*60}")
    logger.info(f"Done. Fetched: {manifest['n_fetched']}  |  "
                f"Skipped: {manifest['n_skipped']}  |  Failed: {manifest['n_failed']}")
    if manifest["failed"]:
        logger.warning(f"Failed songs: {manifest['failed']}")
    logger.info(f"Manifest saved → {manifest_path}")

    return manifest


def load_lyrics(song_id: str) -> dict | None:
    """Load the lyrics JSON for a single song_id. Returns None if not found."""
    p = Path(LYRICS_DIR) / f"{song_id}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_all_lyrics() -> dict[str, dict]:
    """Load all available lyrics JSON files. Returns {song_id: lyrics_dict}."""
    out = {}
    for path in sorted(Path(LYRICS_DIR).glob("*.json")):
        if path.name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out[data["song_id"]] = data
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape lyrics from Genius")
    parser.add_argument("--dry-run",  action="store_true", help="Show what would be fetched")
    parser.add_argument("--artists",  nargs="*", help="Artist codes to scrape (default: all)")
    args = parser.parse_args()

    scrape_all(dry_run=args.dry_run, artist_filter=args.artists or None)
