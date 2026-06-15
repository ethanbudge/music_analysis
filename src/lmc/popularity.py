"""
popularity.py — Enrich corpus songs with public popularity metrics and recover
the genre / orientation design variables that the original curated catalog
provided by hand.

Sources (all keyed off the LRCLIB song we already have):
  • Spotify  — 0–100 popularity score (primary outcome) + artist genre tags.
  • YouTube  — view / like / comment counts (mirrored from the audio stage).
  • Deezer   — public `rank` (no key required) as a secondary popularity signal.
  • Last.fm  — listeners / playcount (optional, needs LASTFM_API_KEY).

Genre / orientation recovery
----------------------------
The original study crossed a coarse genre cluster with a narrative-vs-production
"orientation". We reconstruct both from Spotify's artist genre tags via a keyword
map (see GENRE_MAP / ORIENTATION_MAP). This is heuristic and documented as a
limitation in the README; songs with no usable tags get genre='unknown'.

Resumable: songs already attempted are skipped unless force=True.
"""

from __future__ import annotations
import re
import json
import time
import logging
from datetime import datetime, timezone

import requests

from .config import (SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET,
                     LASTFM_API_KEY)
from . import db as projdb

logger = logging.getLogger(__name__)

# Spotify genre tag → coarse cluster. First matching keyword wins.
GENRE_MAP = [
    ("hip-hop",    ["hip hop", "rap", "trap", "drill", "grime"]),
    ("country",    ["country", "americana", "bluegrass"]),
    ("folk",       ["folk", "singer-songwriter", "songwriter"]),
    ("electronic", ["edm", "house", "techno", "electro", "dubstep", "trance",
                    "drum and bass", "dance", "electronic"]),
    ("rock",       ["rock", "punk", "grunge", "metal", "emo"]),
    ("r&b",        ["r&b", "rnb", "soul", "funk", "neo soul"]),
    ("latin",      ["latin", "reggaeton", "salsa", "bachata", "cumbia"]),
    ("pop",        ["pop", "boy band", "girl group"]),
]
# Coarse cluster → narrative (lyrics-forward) vs production (sound-forward).
ORIENTATION_MAP = {
    "hip-hop": "narrative", "country": "narrative", "folk": "narrative",
    "electronic": "production", "pop": "production", "r&b": "production",
    "rock": "narrative", "latin": "production", "unknown": "unknown",
}


def recover_genre_orientation(genre_tags: list[str]) -> tuple[str, str]:
    """Map a list of Spotify artist genre tags to (cluster, orientation)."""
    blob = " ".join(genre_tags).lower()
    for cluster, kws in GENRE_MAP:
        if any(kw in blob for kw in kws):
            return cluster, ORIENTATION_MAP[cluster]
    return "unknown", "unknown"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


# ─── Spotify ─────────────────────────────────────────────────────────────────────
def build_spotify_client():
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise RuntimeError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in the environment.")
    auth = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                    client_secret=SPOTIFY_CLIENT_SECRET)
    return spotipy.Spotify(auth_manager=auth, requests_timeout=15, retries=3)


def _spotify_match(sp, song: dict) -> dict:
    """Find the best Spotify track for a corpus song; return popularity + genres."""
    title, artist, dur = song["title"], song["artist"], song["duration"]
    try:
        res = sp.search(q=f'track:{title} artist:{artist}', type="track", limit=10)
        items = res.get("tracks", {}).get("items", [])
        if not items:
            res = sp.search(q=f"{title} {artist}", type="track", limit=10)
            items = res.get("tracks", {}).get("items", [])
    except Exception as e:                                     # noqa: BLE001
        return {"found": 0, "error": f"search: {e}"}

    if not items:
        return {"found": 0}

    norm_artist = _norm(artist)
    def good(t):
        names = [_norm(a["name"]) for a in t["artists"]]
        artist_ok = any(norm_artist in n or n in norm_artist for n in names)
        dur_ok = (dur is None) or abs(t["duration_ms"] / 1000.0 - dur) <= 12
        return artist_ok, dur_ok

    ranked = sorted(items, key=lambda t: (good(t)[0], good(t)[1], t["popularity"]),
                    reverse=True)
    track = ranked[0]

    genres: list[str] = []
    try:
        artist_id = track["artists"][0]["id"]
        genres = sp.artist(artist_id).get("genres", [])
    except Exception:                                          # noqa: BLE001
        pass

    cluster, orientation = recover_genre_orientation(genres)
    return {
        "found": 1,
        "spotify_id": track["id"],
        "spotify_popularity": track["popularity"],
        "spotify_artist_genres": json.dumps(genres),
        "release_date": track.get("album", {}).get("release_date", ""),
        "genre": cluster,
        "orientation": orientation,
    }


# ─── Deezer (public, no key) ─────────────────────────────────────────────────────
def _deezer_rank(title: str, artist: str) -> int | None:
    try:
        r = requests.get("https://api.deezer.com/search",
                         params={"q": f'artist:"{artist}" track:"{title}"'}, timeout=10)
        data = r.json().get("data", [])
        return int(data[0]["rank"]) if data else None
    except Exception:                                          # noqa: BLE001
        return None


# ─── Last.fm (optional) ──────────────────────────────────────────────────────────
def _lastfm_info(title: str, artist: str) -> dict:
    if not LASTFM_API_KEY:
        return {}
    try:
        r = requests.get("https://ws.audioscrobbler.com/2.0/", params={
            "method": "track.getInfo", "api_key": LASTFM_API_KEY,
            "artist": artist, "track": title, "format": "json", "autocorrect": 1,
        }, timeout=10)
        t = r.json().get("track", {})
        return {"lastfm_listeners": int(t.get("listeners", 0) or 0),
                "lastfm_playcount": int(t.get("playcount", 0) or 0)}
    except Exception:                                          # noqa: BLE001
        return {}


def fetch_pending(limit: int | None = None, force: bool = False,
                  use_deezer: bool = True) -> dict:
    """Fetch popularity metrics for corpus songs not yet attempted."""
    with projdb.connect() as conn:
        done = set() if force else {
            r["track_id"] for r in conn.execute("SELECT track_id FROM popularity")
        }
        songs = [dict(r) for r in conn.execute("SELECT * FROM songs ORDER BY track_id")
                 if force or r["track_id"] not in done]
        yt = {r["track_id"]: r for r in conn.execute(
            "SELECT track_id, view_count, like_count, comment_count FROM audio")}
    if limit:
        songs = songs[:limit]
    if not songs:
        logger.info("Popularity: nothing pending.")
        return {"attempted": 0, "found": 0}

    sp = build_spotify_client()
    logger.info("Popularity: %d songs to enrich.", len(songs))
    found = 0
    for i, song in enumerate(songs, 1):
        tid = song["track_id"]
        logger.info("[%d/%d] %s — %s", i, len(songs), song["artist"], song["title"])
        row = {"track_id": tid, "fetched_at": datetime.now(timezone.utc).isoformat()}
        row.update(_spotify_match(sp, song))
        if use_deezer:
            row["deezer_rank"] = _deezer_rank(song["title"], song["artist"])
        row.update(_lastfm_info(song["title"], song["artist"]))
        if tid in yt:
            row["yt_view_count"]    = yt[tid]["view_count"]
            row["yt_comment_count"] = yt[tid]["comment_count"]
            row["yt_like_count"]    = yt[tid]["like_count"]
        with projdb.connect() as conn:
            projdb.upsert(conn, "popularity", row)
        if row.get("found"):
            found += 1
            logger.info("    ✓ pop=%s genre=%s/%s", row.get("spotify_popularity"),
                        row.get("genre"), row.get("orientation"))
        time.sleep(0.2)   # be polite to the APIs

    logger.info("Popularity done: %d/%d matched on Spotify.", found, len(songs))
    return {"attempted": len(songs), "found": found}
