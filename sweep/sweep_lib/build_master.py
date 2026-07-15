"""
build_master.py — assemble results/master_results_sweep.csv for the Stan battery.

Takes the EXISTING master_results.csv (which already carries the outcome, recovered
genre/orientation, song age, MERT PCA controls, and mood columns) and merges on the
new sweep LMC columns from the `lmc_sweep` table, pivoted wide as
`<model>_<prompt>_<method>` (e.g. `mulan_raw_song`, `clamp3_idea_line_buf10`).

The result has exactly the columns sweep/R/run_models_sweep.R expects: the same
control/outcome block as the observational runs, plus 4×3×6 = 72 LMC columns.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from . import config
from lmc import db as projdb

logger = logging.getLogger(__name__)


def _sweep_wide() -> pd.DataFrame:
    """lmc_sweep (long) → wide DataFrame keyed by track_id."""
    with projdb.connect() as conn:
        df = pd.read_sql_query(
            "SELECT track_id, model, prompt, method, value FROM lmc_sweep", conn)
    if df.empty:
        return pd.DataFrame({"track_id": []})
    df["col"] = df["model"] + "_" + df["prompt"] + "_" + df["method"]
    wide = df.pivot_table(index="track_id", columns="col", values="value", aggfunc="first")
    return wide.reset_index()


def build(require_existing_master: bool = True) -> dict:
    """Write results/master_results_sweep.csv; return a small summary."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = config.RESULTS_DIR / "master_results.csv"
    if not existing.exists():
        msg = (f"{existing} not found — build the observational master first "
               f"(combine.build_master()) so the sweep can reuse its controls/outcome.")
        if require_existing_master:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        base = pd.DataFrame({"track_id": []})
    else:
        base = pd.read_csv(existing)

    wide = _sweep_wide()
    master = base.merge(wide, on="track_id", how="left") if not base.empty else wide

    out = config.MASTER_SWEEP_CSV
    master.sort_values("track_id").to_csv(out, index=False)

    lmc_cols = [c for c in config.all_lmc_columns() if c in master.columns]
    # Global complete-case count across ALL sweep columns + controls (what the Stan
    # battery will fit on) — a quick heads-up on how much CLaMP 3/MS-CLAP loss costs.
    ctrl = [c for c in master.columns if c.startswith("mert_pc")]
    needed = [c for c in (["spotify_popularity", "genre"] + ctrl + lmc_cols) if c in master.columns]
    complete = int(master.dropna(subset=needed).shape[0]) if needed else 0

    logger.info("Wrote %s (%d songs, %d LMC cols present of %d).",
                out, len(master), len(lmc_cols), len(config.all_lmc_columns()))
    logger.info("Global complete-case corpus (all %d LMC cols + controls present): %d songs.",
                len(lmc_cols), complete)
    return {"path": str(out), "n_songs": len(master),
            "lmc_cols_present": len(lmc_cols), "lmc_cols_expected": len(config.all_lmc_columns()),
            "global_complete_case": complete}


def missing_report() -> pd.DataFrame:
    """Per (model, prompt, method): how many songs have a value — spot gaps/failures."""
    with projdb.connect() as conn:
        df = pd.read_sql_query(
            "SELECT model, prompt, method, COUNT(*) AS n, "
            "SUM(value IS NULL) AS n_null FROM lmc_sweep GROUP BY model, prompt, method", conn)
    return df.sort_values(["model", "prompt", "method"]).reset_index(drop=True)
