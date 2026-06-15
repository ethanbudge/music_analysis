"""
09_combine_results.py — Merge all pipeline outputs into a single master CSV for R.

Joins together:
  • Artist/song metadata from config.py
  • LMC similarity scores from all 3 models (MuLan, CLAP, MERT+SBERT)
  • Spotify popularity + audio features
  • Segment-level summary stats (if available)

Output
------
  results/master_results.csv   — one row per song, all variables

Column reference
----------------
  Identifiers
    song_id, title, artist_name, artist_code, genre, orientation

  Outcome
    popularity          Spotify 0–100 recency-weighted score

  Primary predictors — Lyric-Music Congruence (LMC)
    lmc_mulan           Diagonal cosine similarity, MuQ-MuLan joint space
    lmc_clap            Diagonal cosine similarity, LAION-CLAP-Music
    lmc_mert_sbert      Diagonal cosine similarity, MERT+SBERT late fusion

  Segment-level LMC (MuLan only; NA if <2 sections found)
    seg_n               Number of lyrical sections analysed
    seg_mean_lmc        Mean LMC across all sections
    seg_sd_lmc          SD of LMC across sections (congruence consistency)
    seg_max_lmc         Peak section LMC
    seg_min_lmc         Trough section LMC
    seg_chorus_lmc      Mean LMC for chorus sections
    seg_verse_lmc       Mean LMC for verse sections
    seg_bridge_lmc      Mean LMC for bridge sections

  Spotify audio features (control variables)
    tempo, danceability, energy, valence, acousticness,
    speechiness, instrumentalness, liveness, loudness

  Derived
    lmc_mulan_z         Z-scored lmc_mulan (for centred quadratic term)
    lmc_clap_z          Z-scored lmc_clap
    lmc_mert_sbert_z    Z-scored lmc_mert_sbert
    duration_min        Track duration in minutes
"""

from __future__ import annotations
import os
import sys
import csv
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CATALOG, RESULTS_DIR, EMBEDDINGS_DIR
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OUT_DIR     = Path(RESULTS_DIR)
MASTER_CSV  = OUT_DIR / "master_results.csv"


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_similarities(model_key: str) -> dict[str, float]:
    """Load similarity scores from results/embeddings/{model_key}/similarities.json"""
    path = Path(EMBEDDINGS_DIR) / model_key / "similarities.json"
    if not path.exists():
        logger.warning(f"  Similarities not found for model '{model_key}' — {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def load_spotify_data() -> dict[str, dict]:
    """Load results/spotify_data.json → {song_id: {...}}"""
    path = OUT_DIR / "spotify_data.json"
    if not path.exists():
        logger.warning("  spotify_data.json not found — popularity will be NA")
        return {}
    with open(path) as f:
        return json.load(f)


def load_segment_summary() -> dict[str, dict]:
    """Load results/segment_analysis/segment_summary.csv → {song_id: {...}}"""
    path = OUT_DIR / "segment_analysis" / "segment_summary.csv"
    if not path.exists():
        logger.info("  No segment summary found — segment columns will be NA")
        return {}
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["song_id"]] = row
    return rows


# ─── Combine ──────────────────────────────────────────────────────────────────

def build_master_df() -> pd.DataFrame:
    """Build the master results DataFrame."""

    # ── Load all data sources ──────────────────────────────────────────────
    sims_mulan     = load_similarities("mulan")
    sims_clap      = load_similarities("clap")
    sims_mert_sbert = load_similarities("mert_sbert")
    spotify        = load_spotify_data()
    segments       = load_segment_summary()

    logger.info(f"LMC data: MuLan={len(sims_mulan)}, CLAP={len(sims_clap)}, "
                f"MERT+SBERT={len(sims_mert_sbert)}")
    logger.info(f"Spotify: {sum(1 for v in spotify.values() if v.get('found'))} found")
    logger.info(f"Segment data: {len(segments)} songs")

    # ── Build one row per song ─────────────────────────────────────────────
    rows = []
    for artist_code, artist_data in CATALOG.items():
        for song_id, song_meta in artist_data["songs"].items():

            sp   = spotify.get(song_id, {})
            seg  = segments.get(song_id, {})

            def na(x):
                return x if x not in (None, "", "None") else np.nan

            row = {
                # Identifiers
                "song_id":       song_id,
                "title":         song_meta["title"],
                "artist_name":   artist_data["name"],
                "artist_code":   artist_code,
                "genre":         artist_data["genre"],
                "orientation":   artist_data["orientation"],   # narrative | production

                # Primary outcome
                "popularity":    na(sp.get("popularity")),

                # LMC scores (main independent variables)
                "lmc_mulan":      na(sims_mulan.get(song_id)),
                "lmc_clap":       na(sims_clap.get(song_id)),
                "lmc_mert_sbert": na(sims_mert_sbert.get(song_id)),

                # Segment-level LMC
                "seg_n":          na(seg.get("n_segments")),
                "seg_mean_lmc":   na(seg.get("mean_lmc_all")),
                "seg_sd_lmc":     na(seg.get("sd_lmc")),
                "seg_max_lmc":    na(seg.get("max_lmc")),
                "seg_min_lmc":    na(seg.get("min_lmc")),
                "seg_chorus_lmc": na(seg.get("mean_lmc_chorus")),
                "seg_verse_lmc":  na(seg.get("mean_lmc_verse")),
                "seg_bridge_lmc": na(seg.get("mean_lmc_bridge")),

                # Spotify audio features (controls)
                "tempo":              na(sp.get("tempo")),
                "danceability":       na(sp.get("danceability")),
                "energy":             na(sp.get("energy")),
                "valence":            na(sp.get("valence")),
                "acousticness":       na(sp.get("acousticness")),
                "speechiness":        na(sp.get("speechiness")),
                "instrumentalness":   na(sp.get("instrumentalness")),
                "liveness":           na(sp.get("liveness")),
                "loudness":           na(sp.get("loudness")),

                # Metadata
                "spotify_id":    sp.get("spotify_id", ""),
                "duration_ms":   na(sp.get("duration_ms")),
                "explicit":      sp.get("explicit", False),
                "release_date":  sp.get("release_date", ""),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # ── Derived variables ──────────────────────────────────────────────────
    df["duration_min"] = df["duration_ms"].astype(float) / 60_000

    # Z-score LMC variables for centred quadratic term in regressions
    for col in ["lmc_mulan", "lmc_clap", "lmc_mert_sbert"]:
        mu   = df[col].mean()
        sd   = df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sd if sd > 0 else np.nan

    # Encode orientation as binary dummy (1 = narrative, 0 = production)
    df["narrative"] = (df["orientation"] == "narrative").astype(int)

    # Genre dummies (for fixed-effects models in R)
    genre_dummies = pd.get_dummies(df["genre"], prefix="genre")
    df = pd.concat([df, genre_dummies], axis=1)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the master dataset."""
    n_songs      = len(df)
    n_with_pop   = df["popularity"].notna().sum()
    n_with_mulan = df["lmc_mulan"].notna().sum()
    n_with_clap  = df["lmc_clap"].notna().sum()
    n_with_sbert = df["lmc_mert_sbert"].notna().sum()
    n_with_seg   = df["seg_mean_lmc"].notna().sum()

    logger.info(f"\n{'═'*60}")
    logger.info(f"Master dataset: {n_songs} songs across {df['artist_code'].nunique()} artists")
    logger.info(f"  With popularity:      {n_with_pop}/{n_songs}")
    logger.info(f"  With MuLan LMC:       {n_with_mulan}/{n_songs}")
    logger.info(f"  With CLAP LMC:        {n_with_clap}/{n_songs}")
    logger.info(f"  With MERT+SBERT LMC:  {n_with_sbert}/{n_songs}")
    logger.info(f"  With segment LMC:     {n_with_seg}/{n_songs}")

    logger.info(f"\nLMC descriptives (MuLan):")
    if n_with_mulan > 0:
        logger.info(df["lmc_mulan"].describe().to_string())

    logger.info(f"\nPopularity descriptives:")
    if n_with_pop > 0:
        logger.info(df["popularity"].describe().to_string())

    logger.info(f"\nSongs per genre:")
    logger.info(df["genre"].value_counts().to_string())

    logger.info(f"\nSongs per orientation:")
    logger.info(df["orientation"].value_counts().to_string())

    complete = df[["popularity", "lmc_mulan"]].dropna()
    logger.info(f"\nComplete cases (popularity + MuLan LMC): {len(complete)}")


def combine():
    """Main entry point: build and save master results."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_master_df()
    print_summary(df)

    df.to_csv(MASTER_CSV, index=False, float_format="%.6f")
    logger.info(f"\nMaster CSV saved → {MASTER_CSV}")
    logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    combine()
