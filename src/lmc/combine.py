"""
combine.py — Merge every stage into analysis-ready CSVs for R / Stan.

Outputs (results/):
  master_results.csv  one row per song: identifiers, recovered genre/orientation,
                      all popularity metrics, mood controls, and every LMC measure
                      (model × method) as a column, e.g. `mulan_line_buf5`,
                      `clap_seg_chorus`. This is the primary modeling input.
  lmc_lines.csv       long line-level series (track × model × line × window) with
                      position_pct and chorus flag — for the timeline analysis and
                      the line-level Stan model.
  corpus_status.csv   per-song pipeline completeness (for quick auditing).
"""

from __future__ import annotations
import logging
from datetime import datetime

import pandas as pd

from .config import RESULTS_DIR, MOOD_COLUMNS
from . import db as projdb

logger = logging.getLogger(__name__)


def _read(conn, sql) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def build_master() -> dict:
    """Build master_results.csv + lmc_lines.csv + corpus_status.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with projdb.connect() as conn:
        songs = _read(conn, "SELECT * FROM songs")
        pop   = _read(conn, "SELECT * FROM popularity")
        mood  = _read(conn, "SELECT * FROM mood")
        lmc   = _read(conn, "SELECT * FROM lmc")
        lines = _read(conn, "SELECT * FROM lmc_lines")
        audio = _read(conn, "SELECT track_id, view_count, like_count, comment_count, channel, is_topic FROM audio")

    if songs.empty:
        logger.warning("No songs in corpus — nothing to combine.")
        return {}

    master = songs[["track_id", "title", "artist", "album", "duration", "n_synced_lines"]].copy()

    # Popularity + recovered design variables.
    if not pop.empty:
        keep = ["track_id", "spotify_popularity", "spotify_id", "release_date",
                "genre", "orientation", "deezer_rank", "lastfm_listeners",
                "lastfm_playcount", "yt_view_count", "yt_comment_count", "yt_like_count"]
        master = master.merge(pop[[c for c in keep if c in pop.columns]], on="track_id", how="left")
    if not audio.empty:
        master = master.merge(audio, on="track_id", how="left", suffixes=("", "_audio"))

    # Mood controls.
    if not mood.empty:
        master = master.merge(mood[["track_id", *MOOD_COLUMNS]], on="track_id", how="left")

    # LMC: pivot (model, method) → wide columns "<model>_<method>".
    if not lmc.empty:
        lmc["col"] = lmc["model"] + "_" + lmc["method"]
        wide = lmc.pivot_table(index="track_id", columns="col", values="value", aggfunc="first")
        master = master.merge(wide.reset_index(), on="track_id", how="left")

    # Derived: song age in years from release date (relative to today).
    if "release_date" in master.columns:
        yr = pd.to_datetime(master["release_date"], errors="coerce").dt.year
        master["song_age_years"] = datetime.now().year - yr

    master = master.sort_values("track_id")
    out_master = RESULTS_DIR / "master_results.csv"
    master.to_csv(out_master, index=False)

    out_lines = RESULTS_DIR / "lmc_lines.csv"
    if not lines.empty:
        lines.sort_values(["track_id", "model", "window", "line_idx"]).to_csv(out_lines, index=False)

    # Per-song completeness audit.
    status = songs[["track_id", "title", "artist"]].copy()
    status["has_audio"]      = status["track_id"].isin(audio["track_id"]) if not audio.empty else False
    status["has_popularity"] = status["track_id"].isin(pop[pop.get("found", 0) == 1]["track_id"]) if not pop.empty else False
    status["has_mood"]       = status["track_id"].isin(mood["track_id"]) if not mood.empty else False
    status["has_lmc"]        = status["track_id"].isin(lmc["track_id"]) if not lmc.empty else False
    out_status = RESULTS_DIR / "corpus_status.csv"
    status.to_csv(out_status, index=False)

    logger.info("Wrote %s (%d songs, %d cols)", out_master, len(master), master.shape[1])
    logger.info("Wrote %s (%d line rows)", out_lines, len(lines))
    logger.info("Wrote %s", out_status)
    return {"master_rows": len(master), "line_rows": len(lines),
            "master_path": str(out_master)}
