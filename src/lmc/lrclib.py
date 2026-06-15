"""
lrclib.py — Read the LRCLIB dump, report corpus status, and randomly sample
new songs that have time-synced lyrics into the project database.

LRCLIB is the source of truth. The dump has ~19.5M lyrics rows with synced
lyrics, so we never use `ORDER BY RANDOM()` (a full 100+ GB scan). Instead we
draw random primary-key probe points and walk forward to the next qualifying
row, which uses the integer-PK / synced-lyrics indexes and stays fast.

Each sampled song is keyed by its LRCLIB `tracks.id`, which becomes the
canonical song ID used to name audio files and join every other data source.
"""

from __future__ import annotations
import sqlite3
import random
import logging
from datetime import datetime, timezone

from .config import LRCLIB_DUMP, PROJECT_DB, SAMPLE_FILTERS
from .utils import parse_lrc, lrc_to_plaintext, ascii_ratio
from . import db as projdb

logger = logging.getLogger(__name__)


def _dump_connect() -> sqlite3.Connection:
    if LRCLIB_DUMP is None or not LRCLIB_DUMP.exists():
        raise FileNotFoundError(
            "LRCLIB dump not found. Place lrclib-db-dump-*.sqlite3 under data/ "
            "or set the LRCLIB_DUMP environment variable."
        )
    # Read-only; immutable=1 tells SQLite the file will not change (faster, no locks).
    uri = f"file:{LRCLIB_DUMP}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, timeout=120)
    conn.row_factory = sqlite3.Row
    return conn


# ─── Cached universe size (the full COUNT is expensive on the dump) ──────────────
def _meta_get(conn: sqlite3.Connection, key: str):
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))


def universe_size(force: bool = False) -> int:
    """
    Total LRCLIB lyrics rows with synced lyrics (cached in the project DB after
    the first, expensive, full count). This is the upper bound on sampleable songs.
    """
    with projdb.connect() as conn:
        cached = _meta_get(conn, "synced_universe")
        if cached is not None and not force:
            return int(cached)
    logger.info("Counting synced-lyric rows in the dump (one-time, may take a minute)…")
    with _dump_connect() as dump:
        n = dump.execute(
            "SELECT COUNT(*) AS n FROM lyrics WHERE has_synced_lyrics = 1"
        ).fetchone()["n"]
    with projdb.connect() as conn:
        _meta_set(conn, "synced_universe", n)
    return n


def setup(force_count: bool = False) -> dict:
    """
    Initialise the project DB and report corpus status:
      • how many synced-lyric songs exist in LRCLIB (the universe),
      • how many we have already sampled,
      • roughly how many remain.

    Call this at the top of a session before sampling.
    """
    projdb.init_db()
    universe = universe_size(force=force_count)
    with projdb.connect() as conn:
        sampled = projdb.count(conn, "songs")
    report = {
        "universe":  universe,
        "sampled":   sampled,
        "remaining": max(universe - sampled, 0),
    }
    logger.info(
        "LRCLIB corpus — universe %s | sampled %s | remaining ≈ %s",
        f"{universe:,}", f"{sampled:,}", f"{report['remaining']:,}",
    )
    return report


# ─── Sampling ────────────────────────────────────────────────────────────────────
def _id_bounds(dump: sqlite3.Connection) -> tuple[int, int]:
    row = dump.execute(
        "SELECT MIN(id) AS lo, MAX(id) AS hi FROM lyrics WHERE has_synced_lyrics = 1"
    ).fetchone()
    return int(row["lo"]), int(row["hi"])


def _passes_filters(track_row: sqlite3.Row, lines: list[dict], synced: str) -> bool:
    f = SAMPLE_FILTERS
    dur = track_row["duration"]
    if dur is not None and not (f["min_duration_s"] <= dur <= f["max_duration_s"]):
        return False
    if len(lines) < f["min_synced_lines"]:
        return False
    if ascii_ratio(synced) < f["require_ascii_ratio"]:
        return False
    title = (track_row["name"] or "").lower()
    if any(kw in title for kw in f["exclude_title_keywords"]):
        return False
    return True


def sample(n: int, seed: int | None = None, max_attempts_factor: int = 60) -> list[int]:
    """
    Randomly draw up to `n` *new* synced-lyric songs into the project DB.

    Returns the list of newly added track_ids. Already-sampled tracks are never
    re-added, so calling sample() repeatedly across sessions keeps growing the
    corpus without duplication.
    """
    projdb.init_db()
    rng = random.Random(seed)

    with projdb.connect() as conn:
        already = projdb.sampled_track_ids(conn)

    added: list[int] = []
    seen_tracks = set(already)
    now = datetime.now(timezone.utc).isoformat()

    with _dump_connect() as dump, projdb.connect() as proj:
        lo, hi = _id_bounds(dump)
        attempts, max_attempts = 0, max(n * max_attempts_factor, 200)

        while len(added) < n and attempts < max_attempts:
            attempts += 1
            probe = rng.randint(lo, hi)
            # Walk forward to the next synced, non-instrumental lyrics row.
            lyr = dump.execute(
                """SELECT id, track_id, synced_lyrics, plain_lyrics
                   FROM lyrics
                   WHERE has_synced_lyrics = 1 AND instrumental = 0 AND id >= ?
                   ORDER BY id LIMIT 1""",
                (probe,),
            ).fetchone()
            if lyr is None:
                continue

            track_id = lyr["track_id"]
            if track_id is None or track_id in seen_tracks:
                continue

            trk = dump.execute(
                "SELECT id, name, artist_name, album_name, duration FROM tracks WHERE id = ?",
                (track_id,),
            ).fetchone()
            if trk is None:
                continue

            synced = lyr["synced_lyrics"] or ""
            lines = parse_lrc(synced)
            if not _passes_filters(trk, lines, synced):
                seen_tracks.add(track_id)         # don't keep probing the same reject
                continue

            projdb.upsert(proj, "songs", {
                "track_id":       track_id,
                "lyrics_id":      lyr["id"],
                "title":          trk["name"],
                "artist":         trk["artist_name"],
                "album":          trk["album_name"],
                "duration":       trk["duration"],
                "n_synced_lines": len(lines),
                "synced_lyrics":  synced,
                "plain_lyrics":   lyr["plain_lyrics"] or lrc_to_plaintext(synced),
                "sampled_at":     now,
            })
            seen_tracks.add(track_id)
            added.append(track_id)

    logger.info("Sampled %d new songs (%d attempts). Corpus now %d.",
                len(added), attempts, len(already) + len(added))
    if len(added) < n:
        logger.warning("Requested %d but only added %d — raise max_attempts_factor "
                       "or relax SAMPLE_FILTERS if this is unexpectedly low.", n, len(added))
    return added


def get_song(track_id: int) -> sqlite3.Row | None:
    with projdb.connect() as conn:
        return conn.execute("SELECT * FROM songs WHERE track_id = ?", (track_id,)).fetchone()
