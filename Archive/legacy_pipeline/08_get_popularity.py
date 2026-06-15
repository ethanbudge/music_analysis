"""
08_get_popularity.py — Fetch Spotify popularity scores for all songs.

What Spotify provides (free, public API)
-----------------------------------------
  popularity         : 0–100 recency-weighted streaming score (primary outcome)
  followers          : artist follower count (control variable)
  Audio features     : tempo, danceability, energy, valence, acousticness,
                       speechiness, instrumentalness, liveness, loudness
                       (these are Spotify/Echo Nest features — useful as controls)

What we DON'T get
-----------------
  Raw stream counts are not exposed via the public API.
  Use the popularity score as the outcome variable (it's arguably better anyway;
  see lit review Section 2.6 for the recency-weighting argument).
  For raw streams you can manually look up values on Spotify's artist dashboard
  or third-party sites (Chartmetric, Spotify Charts) and add them to the CSV.

Matching strategy
-----------------
  1. Search Spotify for '{title} {artist}' → take the top result.
  2. Filter by exact artist name match (or fuzzy match if exact fails).
  3. Store the track's popularity, ID, and audio features.

Outputs
-------
  results/spotify_data.json   — full API response per song
  results/spotify_summary.csv — flat CSV for joining into master data
"""

from __future__ import annotations
import os
import sys
import csv
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CATALOG, RESULTS_DIR,
    SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET,
)
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OUT_DIR     = Path(RESULTS_DIR)
DATA_PATH   = OUT_DIR / "spotify_data.json"
SUMMARY_CSV = OUT_DIR / "spotify_summary.csv"


def build_spotify_client():
    """Initialise and return a spotipy client using Client Credentials flow."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except ImportError:
        logger.error("spotipy not installed.  Run: pip install spotipy")
        sys.exit(1)

    if SPOTIFY_CLIENT_ID in ("", "YOUR_SPOTIFY_CLIENT_ID"):
        logger.error("Spotify credentials not set.  "
                     "export SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=...")
        sys.exit(1)

    auth = SpotifyClientCredentials(
        client_id     = SPOTIFY_CLIENT_ID,
        client_secret = SPOTIFY_CLIENT_SECRET,
    )
    return spotipy.Spotify(auth_manager=auth)


def normalise(s: str) -> str:
    """Lowercase, remove punctuation for fuzzy matching."""
    import re
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()


def search_track(sp, title: str, artist_name: str, max_results: int = 10) -> dict | None:
    LIVE_KEYWORDS = ["live", "concert", "tour", "session", "acoustic",
                     "sxsw", "unplugged", "at the", "in concert", "recorded at"]

    def is_live(track: dict) -> bool:
        name  = track.get("name", "").lower()
        album = track.get("album", {}).get("name", "").lower()
        return any(kw in name or kw in album for kw in LIVE_KEYWORDS)

    query = f"track:{title} artist:{artist_name}"
    try:
        results = sp.search(q=query, type="track", limit=max_results)
        tracks  = results.get("tracks", {}).get("items", [])
    except Exception as e:
        logger.warning(f"  Spotify search error: {e}")
        return None

    if not tracks:
        try:
            results = sp.search(q=f"{title} {artist_name}", type="track", limit=max_results)
            tracks  = results.get("tracks", {}).get("items", [])
        except Exception:
            return None

    if not tracks:
        return None

    norm_artist = normalise(artist_name)

    # Prefer: artist match + not live
    for track in tracks:
        track_artists = [normalise(a["name"]) for a in track["artists"]]
        artist_match  = any(norm_artist in ta or ta in norm_artist for ta in track_artists)
        if artist_match and not is_live(track):
            return track

    # Fallback: artist match even if live
    for track in tracks:
        track_artists = [normalise(a["name"]) for a in track["artists"]]
        if any(norm_artist in ta or ta in norm_artist for ta in track_artists):
            return track

    # Last resort: first result
    return tracks[0] if not is_live(tracks[0]) else None


def get_audio_features(sp, track_id: str) -> dict | None:
    """Fetch Spotify audio features for a track ID."""
    try:
        feats = sp.audio_features([track_id])
        if feats and feats[0]:
            return feats[0]
    except Exception as e:
        logger.debug(f"  Audio features error: {e}")
    return None


def fetch_all_popularity(force: bool = False) -> dict:
    """
    Fetch popularity + audio features for every song in the catalog.
    Returns {song_id: {...}} dict (also written to disk).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume: load existing data
    existing: dict = {}
    if not force and DATA_PATH.exists():
        with open(DATA_PATH) as f:
            existing = json.load(f)
        logger.info(f"Loaded existing Spotify data for {len(existing)} songs.")

    sp = build_spotify_client()

    for artist_code, artist_data in CATALOG.items():
        artist_name = artist_data["name"]
        logger.info(f"\n{'─'*55}  {artist_name}")

        for song_id, song_meta in artist_data["songs"].items():
            if not force and song_id in existing:
                logger.debug(f"  [{song_id}] SKIP")
                continue

            title = song_meta["title"]
            logger.info(f"  [{song_id}] Searching: {title}")

            track = search_track(sp, title, artist_name)
            if not track:
                logger.warning(f"    ✗ Not found on Spotify")
                existing[song_id] = {
                    "song_id":     song_id,
                    "title":       title,
                    "artist":      artist_name,
                    "found":       False,
                    "fetched_at":  datetime.now(timezone.utc).isoformat(),
                }
                continue

            track_id   = track["id"]
            popularity = track["popularity"]
            logger.info(f"    ✓ Found: '{track['name']}' | popularity={popularity}")

            entry = {
                "song_id":        song_id,
                "title":          title,
                "artist":         artist_name,
                "found":          True,
                "spotify_id":     track_id,
                "spotify_name":   track["name"],
                "popularity":     popularity,
                "duration_ms":    track.get("duration_ms"),
                "explicit":       track.get("explicit", False),
                "release_date":   track.get("album", {}).get("release_date", ""),
                # Audio features (Spotify/Echo Nest)
                "tempo":              None,
                "danceability":       None,
                "energy":             None,
                "valence":            None,
                "acousticness":       None,
                "speechiness":        None,
                "instrumentalness":   None,
                "liveness":           None,
                "loudness":           None,
                "key":                None,
                "mode":               None,
                "time_signature":     None,
                "fetched_at":     datetime.now(timezone.utc).isoformat(),
            }
            existing[song_id] = entry

            # Save after each song
            with open(DATA_PATH, "w") as f:
                json.dump(existing, f, indent=2)

            time.sleep(0.3)   # respect rate limits

    # ── Write summary CSV ──────────────────────────────────────────────────
    _write_summary_csv(existing)

    n_found   = sum(1 for v in existing.values() if v.get("found"))
    n_missing = len(existing) - n_found
    logger.info(f"\n{'═'*60}")
    logger.info(f"Spotify: found {n_found}, missing {n_missing}")
    logger.info(f"Data → {DATA_PATH}\nCSV  → {SUMMARY_CSV}")

    return existing


def _write_summary_csv(data: dict) -> None:
    """Write a flat CSV from the Spotify data dict."""
    fields = [
        "song_id", "title", "artist", "found", "spotify_id",
        "popularity", "duration_ms", "explicit", "release_date",
        "tempo", "danceability", "energy", "valence",
        "acousticness", "speechiness", "instrumentalness",
        "liveness", "loudness", "key", "mode", "time_signature",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for song_id, entry in sorted(data.items()):
            writer.writerow({k: entry.get(k, "") for k in fields})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Spotify popularity data")
    parser.add_argument("--force", action="store_true", help="Re-fetch everything")
    args = parser.parse_args()

    fetch_all_popularity(force=args.force)
