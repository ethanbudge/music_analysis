"""
acquire.py — build the validation corpus from a Spotify playlist.

Flow (reuses the main pipeline wherever possible):
  1. read the playlist track list from Spotify           (spotipy + your creds)
  2. fetch each track's OWN time-synced lyrics from LRCLIB (REST API — the main
     pipeline reads the bulk dump; for a handful of known tracks the public API is
     simpler and gets that exact recording's timing)
  3. download the official audio                          (yt-dlp, reusing
     lmc.audio._search_best / _score_candidate for candidate ranking)
  4. parse the LRC into timed lines                       (lmc.utils.parse_lrc)
  5. label chorus vs non-chorus lines                     (lmc.chorus.detect_chorus)

Each cover uses its OWN synced lyrics, so the (slightly different) wording and the
timestamps line up with that specific recording — exactly as intended.

Everything is cached (audio files, .lrc files, a manifest) so re-runs are cheap.
"""

from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from . import config
from lmc import audio as lmc_audio
from lmc import chorus as chorus_mod
from lmc.utils import parse_lrc, lrc_to_plaintext

logger = logging.getLogger(__name__)

LRCLIB_BASE = "https://lrclib.net/api"
LRCLIB_UA = "lmc-validation/0.1 (research; ethanbudge2000@gmail.com)"


@dataclass
class Track:
    slug: str
    artist: str
    title: str
    album: str
    duration_s: float | None
    spotify_id: str = ""
    audio_path: str | None = None
    synced_lyrics: str = ""
    lines: list = field(default_factory=list)       # parse_lrc() output
    chorus_flags: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.audio_path and self.synced_lyrics and self.lines)


def _slugify(*parts: str) -> str:
    s = "_".join(p for p in parts if p)
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[\s]+", "-", s).strip("-")[:80]


def _clean(text: str) -> str:
    """Drop parenthetical qualifiers ('(feat. …)', '- Remastered') for lookup."""
    text = re.sub(r"\s*[\(\[].*?[\)\]]", "", text)
    text = re.sub(r"\s*-\s*(feat|remaster|remastered|live|mono|stereo).*$", "", text, flags=re.I)
    return text.strip()


# ── Spotify ───────────────────────────────────────────────────────────────────────
def get_spotify_client():
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from lmc import config as lmc_config
    if not lmc_config.SPOTIFY_CLIENT_ID:
        raise RuntimeError("Spotify credentials missing — set SPOTIFY_CLIENT_ID / "
                           "SPOTIFY_CLIENT_SECRET (see main README).")
    auth = SpotifyClientCredentials(client_id=lmc_config.SPOTIFY_CLIENT_ID,
                                    client_secret=lmc_config.SPOTIFY_CLIENT_SECRET)
    return spotipy.Spotify(auth_manager=auth)


def playlist_tracks(playlist_id: str | None = None) -> list[dict]:
    """Return [{artist, title, album, duration_s, spotify_id}] for the playlist."""
    sp = get_spotify_client()
    pid = playlist_id or config.PLAYLIST_ID
    out, results = [], sp.playlist_items(pid, additional_types=("track",))
    while results:
        for it in results["items"]:
            t = it.get("track") or {}
            if not t or t.get("type") != "track":
                continue
            out.append({
                "artist": ", ".join(a["name"] for a in t.get("artists", [])) or "Unknown",
                "title":  t.get("name", ""),
                "album":  (t.get("album") or {}).get("name", ""),
                "duration_s": (t.get("duration_ms") or 0) / 1000.0,
                "spotify_id": t.get("id", ""),
            })
        results = sp.next(results) if results.get("next") else None
    logger.info("Playlist %s: %d tracks.", pid, len(out))
    return out


# ── LRCLIB (REST) ─────────────────────────────────────────────────────────────────
def lrclib_get(artist: str, title: str, album: str, duration_s: float | None) -> str | None:
    """Fetch synced lyrics for one exact track signature; fall back to search."""
    params = {"artist_name": _clean(artist), "track_name": _clean(title)}
    if album:
        params["album_name"] = _clean(album)
    if duration_s:
        params["duration"] = int(round(duration_s))
    try:
        r = requests.get(f"{LRCLIB_BASE}/get", params=params,
                         headers={"User-Agent": LRCLIB_UA}, timeout=30)
        if r.status_code == 200:
            j = r.json()
            if j.get("syncedLyrics"):
                return j["syncedLyrics"]
    except Exception as e:                                     # noqa: BLE001
        logger.debug("  lrclib get failed: %s", e)

    # Fallback: search by artist+title, take the best synced hit near the duration.
    try:
        r = requests.get(f"{LRCLIB_BASE}/search",
                         params={"artist_name": _clean(artist), "track_name": _clean(title)},
                         headers={"User-Agent": LRCLIB_UA}, timeout=30)
        hits = [h for h in (r.json() if r.status_code == 200 else []) if h.get("syncedLyrics")]
        if hits:
            if duration_s:
                hits.sort(key=lambda h: abs((h.get("duration") or 0) - duration_s))
            return hits[0]["syncedLyrics"]
    except Exception as e:                                     # noqa: BLE001
        logger.debug("  lrclib search failed: %s", e)
    return None


# ── audio download (reuse the main pipeline's candidate scoring) ─────────────────
def download_audio(artist: str, title: str, duration_s: float | None, slug: str) -> str | None:
    """Download the best official-audio candidate to validation/data/audio/<slug>.mp3."""
    import yt_dlp
    out = config.AUDIO_DIR / f"{slug}.mp3"
    if out.exists():
        return str(out)
    query = f"{artist} {title} audio"
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True,
                           "default_search": "ytsearch", "skip_download": True}) as ydl:
        best = lmc_audio._search_best(ydl, query, duration_s)     # reused scorer
    if best is None:
        logger.warning("  no audio candidate for %s — %s", artist, title)
        return None
    dl_opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": str(config.AUDIO_DIR / f"{slug}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                            "preferredquality": "192"}],
    }
    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.extract_info(f"https://www.youtube.com/watch?v={best.get('id')}", download=True)
    except Exception as e:                                     # noqa: BLE001
        logger.warning("  download failed for %s: %s", slug, e)
        return None
    return str(out) if out.exists() else None


# ── build the corpus ──────────────────────────────────────────────────────────────
def build_tracks(playlist_id: str | None = None, limit: int | None = None,
                 download: bool = True) -> list[Track]:
    """Assemble Track objects (audio + synced lyrics + chorus flags), resumable."""
    config.ensure_dirs()
    raw = playlist_tracks(playlist_id)
    if limit:
        raw = raw[:limit]

    tracks: list[Track] = []
    for meta in raw:
        slug = _slugify(meta["artist"], meta["title"])
        tk = Track(slug=slug, artist=meta["artist"], title=meta["title"],
                   album=meta["album"], duration_s=meta["duration_s"],
                   spotify_id=meta["spotify_id"])

        # synced lyrics (cached .lrc)
        lrc_path = config.LYRICS_DIR / f"{slug}.lrc"
        if lrc_path.exists():
            tk.synced_lyrics = lrc_path.read_text(encoding="utf-8")
        else:
            synced = lrclib_get(meta["artist"], meta["title"], meta["album"], meta["duration_s"])
            if synced:
                lrc_path.write_text(synced, encoding="utf-8")
                tk.synced_lyrics = synced
                time.sleep(0.3)                                # be polite to LRCLIB
            else:
                logger.warning("  no synced lyrics for %s — %s", meta["artist"], meta["title"])

        # audio
        if download:
            tk.audio_path = download_audio(meta["artist"], meta["title"],
                                           meta["duration_s"], slug)
        elif (config.AUDIO_DIR / f"{slug}.mp3").exists():
            tk.audio_path = str(config.AUDIO_DIR / f"{slug}.mp3")

        # parse + chorus
        if tk.synced_lyrics:
            tk.lines = parse_lrc(tk.synced_lyrics)
            tk.chorus_flags = chorus_mod.detect_chorus(tk.lines)

        tracks.append(tk)
        logger.info("  %-28s audio=%s lyrics=%s lines=%d",
                    meta["artist"][:28], "ok" if tk.audio_path else "—",
                    "ok" if tk.synced_lyrics else "—", len(tk.lines))

    # manifest for reference
    manifest = [{"slug": t.slug, "artist": t.artist, "title": t.title,
                 "duration_s": t.duration_s, "has_audio": bool(t.audio_path),
                 "has_lyrics": bool(t.synced_lyrics), "n_lines": len(t.lines)}
                for t in tracks]
    (config.CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    usable = sum(t.ok for t in tracks)
    logger.info("Built %d tracks (%d usable with audio+lyrics).", len(tracks), usable)
    return tracks
