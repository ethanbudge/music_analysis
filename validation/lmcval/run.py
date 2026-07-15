"""
run.py — compute LMC for every model x prompt x unit and write the three CSVs.

Output CSVs (validation/results/):
  song_wide.csv     one row per cover        : artist + 12 score columns
  segment_wide.csv  one row per (cover,seg)  : artist, segment_label, position + 12 scores
  line_by_line.csv  one row per (cover,line) : artist, line_index, position, line_text + 12 scores

Score columns are "<model>__<prompt>" for the 4 models x 3 prompts = 12 columns
(NaN where a model could not be loaded/ran). LMC = cosine(audio, text) inside each
model's own space. We report scores, not p-values: the point is the ORDERING —
which covers / segments / lines each metric considers more vs. less congruent.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from . import config, units as units_mod

logger = logging.getLogger(__name__)

_BASE_COLS = {
    "song":    ["artist"],
    "segment": ["artist", "segment_label", "position_pct", "n_lines"],
    "line":    ["artist", "line_index", "position_pct", "is_chorus", "line_text"],
}


def score_columns() -> list[str]:
    """The 12 '<model>__<prompt>' score-column names, in a stable order."""
    return [f"{m}__{p}" for m in config.MODEL_KEYS for p in config.PROMPT_KEYS]


def _row_cosine(A: np.ndarray, T: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    an = np.linalg.norm(A, axis=1, keepdims=True)
    tn = np.linalg.norm(T, axis=1, keepdims=True)
    A = A / np.where(an > 1e-12, an, np.nan)
    T = T / np.where(tn > 1e-12, tn, np.nan)
    return np.sum(A * T, axis=1)


def score_level(units: list, models: dict, level: str, prompts: dict | None = None) -> pd.DataFrame:
    """Score every unit at one level; return the wide DataFrame for its CSV."""
    if not units:
        logger.warning("No %s-level units to score.", level)
        return pd.DataFrame(columns=_BASE_COLS[level] + score_columns())

    wavs = [u.audio for u in units]
    sr = config.BASE_SR
    N = len(units)
    cols: dict[str, np.ndarray] = {}

    for mk in config.MODEL_KEYS:
        if mk not in models:
            for pk in config.PROMPT_KEYS:
                cols[f"{mk}__{pk}"] = np.full(N, np.nan)
            continue
        model = models[mk]
        logger.info("  [%s] embedding %d %s-level audio units…",
                    config.MODEL_DISPLAY.get(mk, mk), N, level)
        try:
            A = model.embed_audio_batch(wavs, sr)
            for pk in config.PROMPT_KEYS:
                texts = [config.format_prompt(pk, u.text, level, prompts) for u in units]
                T = model.embed_text_batch(texts)
                cols[f"{mk}__{pk}"] = np.round(_row_cosine(A, T), 6)
        except Exception as e:                                 # noqa: BLE001
            logger.warning("  [%s] failed on %s level (%s) — its columns are NaN; "
                           "other models continue.", config.MODEL_DISPLAY.get(mk, mk), level, e)
            for pk in config.PROMPT_KEYS:
                cols[f"{mk}__{pk}"] = np.full(N, np.nan)

    # base identity / position columns
    base = {c: [u.position.get(c, getattr(u, c, None)) if c != "artist" else u.artist
                for u in units]
            for c in _BASE_COLS[level]}
    df = pd.DataFrame({**base, **{k: cols[k] for k in score_columns()}})
    return df


def run_all(tracks, models, prompts: dict | None = None, write: bool = True) -> dict:
    """Build units at all three levels, score them, and (optionally) write CSVs."""
    config.ensure_dirs()
    out = {}
    files = {"song": "song_wide.csv", "segment": "segment_wide.csv", "line": "line_by_line.csv"}
    for level in ("song", "segment", "line"):
        us = units_mod.build_units(tracks, level)
        df = score_level(us, models, level, prompts)
        out[level] = df
        if write:
            path = config.RESULTS_DIR / files[level]
            df.to_csv(path, index=False)
            logger.info("Wrote %s (%d rows, %d cols).", path, len(df), df.shape[1])
    return out


def full_run(limit: int | None = None, mock: bool = False,
             model_names: list | None = None, prompts: dict | None = None,
             download: bool = True) -> dict:
    """Acquire the playlist, load the models, score everything. One-call convenience."""
    from lmc.utils import setup_logging
    from . import acquire, models as models_mod
    setup_logging()
    print(config.summary())
    tracks = acquire.build_tracks(config.PLAYLIST_ID, limit=limit, download=download)
    models = models_mod.load_models(model_names, mock=mock)
    return run_all(tracks, models, prompts=prompts)


# ── legible summaries (for the notebook) ─────────────────────────────────────────
def song_ranking(df_song: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    """Covers ranked by congruence. by='model' averages over prompts per model;
    by='all' gives one overall mean. Higher = judged more congruent."""
    sc = [c for c in score_columns() if c in df_song.columns]
    g = df_song.set_index("artist")[sc]
    if by == "all":
        return g.mean(axis=1).sort_values(ascending=False).to_frame("LMC_mean")
    per_model = pd.DataFrame(
        {config.MODEL_DISPLAY[m]: g[[f"{m}__{p}" for p in config.PROMPT_KEYS
                                     if f"{m}__{p}" in g.columns]].mean(axis=1)
         for m in config.MODEL_KEYS})
    per_model["OVERALL"] = per_model.mean(axis=1)
    return per_model.sort_values("OVERALL", ascending=False).round(4)


def rank_agreement(df: pd.DataFrame, id_col: str = "artist") -> pd.DataFrame:
    """Convert each metric column to a rank (1 = most congruent) so you can eyeball
    how consistently the metrics order the items."""
    sc = [c for c in score_columns() if c in df.columns]
    ranks = df[sc].rank(ascending=False, method="min")
    ranks.insert(0, id_col, df[id_col].values)
    return ranks


def plot_heatmap(df: pd.DataFrame, index: str = "artist", title: str = "LMC by model × prompt",
                 ax=None):
    """Heatmap of the 12 score columns for a quick visual read (needs matplotlib)."""
    import matplotlib.pyplot as plt
    sc = [c for c in score_columns() if c in df.columns]
    M = df.set_index(index)[sc]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, max(3, 0.5 * len(M))))
    im = ax.imshow(M.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sc))); ax.set_xticklabels(sc, rotation=90, fontsize=8)
    ax.set_yticks(range(len(M)));  ax.set_yticklabels(M.index, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.02)
    return ax
