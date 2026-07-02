"""
genre.py — Ensemble genre recovery (cascade), replacing the Spotify-only heuristic.

For each song we resolve a coarse cluster (config.GENRE_VOCAB) by trying, in order:

  1. spotify     — Spotify artist genre tags → keyword map (the existing method).
  2. musicbrainz — MusicBrainz recording/artist genre+tag list (backfilled from
                   Discogs / Last.fm) → keyword map. Per-song, objective provenance.
  3. zeroshot    — cosine of the song's cached MuLan audio embedding against
                   "This is a {genre} song" text embeddings; argmax cluster, with a
                   softmax-margin confidence.

Each song gets genre, genre_source, and genre_confidence written to the `genre`
table; combine.build_master() prefers this over the Spotify-only popularity.genre.
This shrinks the large 'unknown' bin that was diluting the genre contrasts.

Network (MusicBrainz) and the MuLan model (zero-shot) are only touched for songs
that still need them, and everything is cached/resumable.
"""

from __future__ import annotations
import json
import logging
import time

import numpy as np

from .config import (GENRE_VOCAB, GENRE_ZEROSHOT_MODEL, GENRE_ZEROSHOT_PROMPT,
                     MUSICBRAINZ_APP, EMBEDDINGS_DIR)
from .popularity import recover_genre_orientation, ORIENTATION_MAP
from .utils import cosine_sim, load_song_embeddings, embedding_path, get_device
from . import db as projdb

logger = logging.getLogger(__name__)


# ─── 1. Spotify tags (reuse existing keyword map) ────────────────────────────
def _from_spotify(spotify_genres_json: str | None):
    if not spotify_genres_json:
        return None
    try:
        tags = json.loads(spotify_genres_json)
    except Exception:                                          # noqa: BLE001
        return None
    cluster, _ = recover_genre_orientation(tags)
    return cluster if cluster != "unknown" else None


# ─── 2. MusicBrainz genres/tags ──────────────────────────────────────────────
def _from_musicbrainz(mb, title: str, artist: str):
    try:
        res = mb.search_recordings(recording=title, artist=artist, limit=3)
        tags = []
        for rec in res.get("recording-list", []):
            tags += [g["name"] for g in rec.get("genre-list", [])]
            tags += [t["name"] for t in rec.get("tag-list", [])]
            for ac in rec.get("artist-credit", []):
                art = ac.get("artist", {}) if isinstance(ac, dict) else {}
                tags += [g["name"] for g in art.get("genre-list", [])]
        if not tags:
            return None
        cluster, _ = recover_genre_orientation(tags)
        return cluster if cluster != "unknown" else None
    except Exception as e:                                     # noqa: BLE001
        logger.debug("  musicbrainz lookup failed: %s", e)
        return None


# ─── 3. Zero-shot from the joint embedding ───────────────────────────────────
class _ZeroShot:
    def __init__(self, model_key=GENRE_ZEROSHOT_MODEL):
        from .embeddings import _load_embedder
        self.model_key = model_key
        self.embedder = _load_embedder(model_key, get_device())
        prompts = [GENRE_ZEROSHOT_PROMPT.format(genre=g) for g in GENRE_VOCAB]
        self.proto = np.stack([self.embedder.embed_text(p) for p in prompts])  # [G, D]

    def classify(self, track_id: int):
        bundle = load_song_embeddings(embedding_path(EMBEDDINGS_DIR, self.model_key, track_id))
        if bundle is None or not np.any(bundle.get("audio_full", 0)):
            return None, 0.0
        a = bundle["audio_full"]
        sims = np.array([cosine_sim(a, p) for p in self.proto])
        # softmax-margin confidence (temperature 20 — CLAP/MuLan cosines are small).
        ex = np.exp(20.0 * (sims - sims.max()))
        probs = ex / ex.sum()
        k = int(probs.argmax())
        return GENRE_VOCAB[k], float(probs[k])


def recover_pending(force: bool = False, use_musicbrainz: bool = True,
                    use_zeroshot: bool = True, limit: int | None = None) -> dict:
    """Run the genre cascade for every song missing an ensemble genre."""
    with projdb.connect() as conn:
        done = set() if force else {r["track_id"] for r in conn.execute("SELECT track_id FROM genre")}
        songs = [dict(r) for r in conn.execute(
            """SELECT s.track_id, s.title, s.artist, p.spotify_artist_genres
               FROM songs s LEFT JOIN popularity p ON p.track_id = s.track_id""")]
    todo = [s for s in songs if force or s["track_id"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        logger.info("Genre: nothing pending.")
        return {"attempted": 0, "done": 0}

    mb = None
    if use_musicbrainz:
        try:
            import musicbrainzngs as mb
            mb.set_useragent(*MUSICBRAINZ_APP)
        except Exception as e:                                 # noqa: BLE001
            logger.warning("musicbrainzngs unavailable (%s) — skipping MB step.", e)
            mb = None
    zs = _ZeroShot() if use_zeroshot else None

    counts = {"spotify": 0, "musicbrainz": 0, "zeroshot": 0, "unknown": 0}
    for i, s in enumerate(todo, 1):
        tid = s["track_id"]
        genre, source, conf = None, "unknown", 0.0

        g = _from_spotify(s.get("spotify_artist_genres"))
        if g:
            genre, source, conf = g, "spotify", 1.0
        elif mb is not None:
            g = _from_musicbrainz(mb, s["title"], s["artist"]); time.sleep(1.05)  # MB rate limit
            if g:
                genre, source, conf = g, "musicbrainz", 0.9
        if genre is None and zs is not None:
            g, c = zs.classify(tid)
            if g:
                genre, source, conf = g, "zeroshot", c
        if genre is None:
            genre = "unknown"

        counts[source] += 1
        with projdb.connect() as conn:
            projdb.upsert(conn, "genre", {
                "track_id": tid, "genre": genre, "genre_source": source,
                "genre_confidence": round(float(conf), 4),
                "orientation": ORIENTATION_MAP.get(genre, "unknown")})
        if i % 25 == 0:
            logger.info("  genre %d/%d  %s", i, len(todo), counts)

    logger.info("Genre done: %s", counts)
    return {"attempted": len(todo), **counts}
