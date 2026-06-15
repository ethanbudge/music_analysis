"""
db.py — Project SQLite database: corpus + per-stage progress tracking.

This is a *separate, small* database from the LRCLIB dump. It records which
songs we have sampled and how far each one has progressed through the pipeline,
so any stage can be stopped and resumed without losing work.

Schema
------
  songs        one row per sampled LRCLIB track (the working corpus + lyrics)
  audio        YouTube audio download status + public YouTube metrics
  popularity   Spotify popularity + recovered genre/orientation + other metrics
  mood         librosa mood / acoustic proxy features
  embeddings   which (track, model) embedding bundles have been computed
  lmc          long-format LMC values: one row per (track, model, method)

All writes are idempotent (INSERT OR REPLACE / upserts) so re-running a stage
never duplicates rows.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from contextlib import contextmanager

from .config import PROJECT_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    track_id        INTEGER PRIMARY KEY,   -- LRCLIB tracks.id (the canonical song ID)
    lyrics_id       INTEGER,               -- LRCLIB lyrics.id used as the synced source
    title           TEXT,
    artist          TEXT,
    album           TEXT,
    duration        REAL,                  -- seconds, from LRCLIB
    n_synced_lines  INTEGER,
    synced_lyrics   TEXT,                  -- raw LRC
    plain_lyrics    TEXT,
    sampled_at      TEXT
);

CREATE TABLE IF NOT EXISTS audio (
    track_id      INTEGER PRIMARY KEY REFERENCES songs(track_id),
    status        TEXT,                    -- 'done' | 'failed' | 'not_found'
    source        TEXT,                    -- 'youtube'
    youtube_id    TEXT,
    youtube_title TEXT,
    channel       TEXT,
    is_topic      INTEGER,                 -- 1 if from an auto-generated "- Topic" channel
    duration_s    REAL,
    view_count    INTEGER,
    like_count    INTEGER,
    comment_count INTEGER,
    file_path     TEXT,
    error         TEXT,
    fetched_at    TEXT
);

CREATE TABLE IF NOT EXISTS popularity (
    track_id           INTEGER PRIMARY KEY REFERENCES songs(track_id),
    found              INTEGER,
    spotify_id         TEXT,
    spotify_popularity INTEGER,
    spotify_artist_genres TEXT,            -- JSON list of genre tags (for genre recovery)
    release_date       TEXT,
    genre              TEXT,               -- recovered coarse genre cluster
    orientation        TEXT,               -- recovered 'narrative' | 'production'
    lastfm_listeners   INTEGER,
    lastfm_playcount   INTEGER,
    deezer_rank        INTEGER,
    -- YouTube metrics are mirrored from the audio table for convenience
    yt_view_count      INTEGER,
    yt_comment_count   INTEGER,
    yt_like_count      INTEGER,
    error              TEXT,
    fetched_at         TEXT
);

CREATE TABLE IF NOT EXISTS mood (
    track_id           INTEGER PRIMARY KEY REFERENCES songs(track_id),
    mood_happy         REAL,
    mood_sad           REAL,
    mood_relaxed       REAL,
    mood_aggressive    REAL,
    mood_party         REAL,
    danceability       REAL,
    voice_instrumental REAL,
    fetched_at         TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    track_id   INTEGER REFERENCES songs(track_id),
    model      TEXT,                       -- 'mulan' | 'clap'
    path       TEXT,
    n_lines    INTEGER,
    fetched_at TEXT,
    PRIMARY KEY (track_id, model)
);

CREATE TABLE IF NOT EXISTS lmc (
    track_id INTEGER REFERENCES songs(track_id),
    model    TEXT,                          -- 'mulan' | 'clap'
    method   TEXT,                          -- 'song' | 'line_exact' | 'line_buf5' | 'seg_chorus' ...
    value    REAL,
    PRIMARY KEY (track_id, model, method)
);

CREATE TABLE IF NOT EXISTS chorus (
    track_id   INTEGER PRIMARY KEY REFERENCES songs(track_id),
    line_flags TEXT,                        -- JSON list[bool], aligned to parse_lrc() order
    n_chorus   INTEGER,
    method     TEXT,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS lmc_lines (
    track_id     INTEGER REFERENCES songs(track_id),
    model        TEXT,
    line_idx     INTEGER,
    window       TEXT,                       -- 'exact' | 'buf1' | 'buf5' | 'buf10'
    position_pct REAL,                       -- line midpoint as % of song duration
    is_chorus    INTEGER,
    lmc          REAL,
    PRIMARY KEY (track_id, model, line_idx, window)
);
"""


@contextmanager
def connect(db_path: Path | str = PROJECT_DB):
    """Context-managed SQLite connection (Row factory, foreign keys on)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str = PROJECT_DB) -> None:
    """Create all tables if they do not exist."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert(conn: sqlite3.Connection, table: str, row: dict) -> None:
    """INSERT OR REPLACE one row from a dict of column→value."""
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    collist = ",".join(cols)
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )


# ─── Progress queries ────────────────────────────────────────────────────────────
def count(conn: sqlite3.Connection, table: str, where: str = "") -> int:
    q = f"SELECT COUNT(*) AS n FROM {table}"
    if where:
        q += f" WHERE {where}"
    return conn.execute(q).fetchone()["n"]


def sampled_track_ids(conn: sqlite3.Connection) -> set[int]:
    return {r["track_id"] for r in conn.execute("SELECT track_id FROM songs")}


def pending_for_audio(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Songs in the corpus with no successful/attempted audio row yet."""
    return conn.execute(
        """SELECT s.* FROM songs s
           LEFT JOIN audio a ON a.track_id = s.track_id
           WHERE a.track_id IS NULL OR a.status NOT IN ('done', 'not_found')
           ORDER BY s.track_id"""
    ).fetchall()


def songs_with_audio(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Songs that have a downloaded audio file (ready for mood/embeddings)."""
    return conn.execute(
        """SELECT s.*, a.file_path FROM songs s
           JOIN audio a ON a.track_id = s.track_id
           WHERE a.status = 'done' AND a.file_path IS NOT NULL
           ORDER BY s.track_id"""
    ).fetchall()


def progress_report(db_path: Path | str = PROJECT_DB) -> dict:
    """Snapshot of how many songs have reached each pipeline stage."""
    with connect(db_path) as conn:
        return {
            "sampled":          count(conn, "songs"),
            "audio_done":       count(conn, "audio", "status = 'done'"),
            "audio_failed":     count(conn, "audio", "status IN ('failed','not_found')"),
            "popularity_found": count(conn, "popularity", "found = 1"),
            "mood_done":        count(conn, "mood"),
            "emb_mulan":        count(conn, "embeddings", "model = 'mulan'"),
            "emb_clap":         count(conn, "embeddings", "model = 'clap'"),
            "lmc_rows":         count(conn, "lmc"),
        }
