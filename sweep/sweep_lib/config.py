"""
config.py — the sweep grid + paths, layered on the existing pipeline + lmcval.

Reuses `lmc` (corpus DB, audio, MERT, chorus, slicing) and `lmcval` (the 4 model
wrappers + the 3 prompt templates) as the single sources of truth, so the sweep
stays consistent with both the observational pipeline and the POC validation test.
"""

from __future__ import annotations
import sys
from pathlib import Path

# ── make `lmc` (src/) and `lmcval` (validation/) importable however we're launched ──
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "validation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lmc import config as lmc_config          # noqa: E402
from lmcval import config as lmcval_config     # noqa: E402  (prompts, BASE_SR, model wrappers)

REPO_ROOT   = _REPO_ROOT
RESULTS_DIR = lmc_config.RESULTS_DIR
PROJECT_DB  = lmc_config.PROJECT_DB

# ── the grid ──────────────────────────────────────────────────────────────────────
MODELS  = ["mulan", "clap", "msclap", "clamp3"]     # 4  (lmcval model keys)
PROMPTS = ["raw", "contains", "idea"]                # 3  (lmcval prompt keys)

# Line context windows (seconds of symmetric padding). We use ±1 / ±5 / ±10 s — NOT
# "exact" — and NO curvature this time, matching the requested design.
LINE_WINDOWS = {"line_buf1": 1.0, "line_buf5": 5.0, "line_buf10": 10.0}

SCALAR_METHODS = ["song", "seg_chorus", "seg_nonchorus"]
LINE_METHODS   = list(LINE_WINDOWS)                  # line_buf1, line_buf5, line_buf10
ALL_METHODS    = SCALAR_METHODS + LINE_METHODS       # 6 scalars per (model, prompt)

# The Stan structures (5 fits per model×prompt embedding):
#   track model on:  song, line_buf1, line_buf5, line_buf10
#   segment model on: seg_chorus + seg_nonchorus
TRACK_MEASURES = ["song"] + LINE_METHODS             # 4 track fits
# (segment is handled from seg_chorus + seg_nonchorus by model_segment_v4)

BASE_SR = lmcval_config.BASE_SR                      # load/slice audio once at this SR

# ── outputs ────────────────────────────────────────────────────────────────────────
MASTER_SWEEP_CSV = RESULTS_DIR / "master_results_sweep.csv"
STAN_OUTPUT_DIR  = REPO_ROOT / "sweep" / "output"    # sweep Stan fits live here

# ── column naming (must match sweep/R exactly) ─────────────────────────────────────
def lmc_col(model: str, prompt: str, method: str) -> str:
    """e.g. ('mulan','raw','song') -> 'mulan_raw_song'; ('clap','idea','seg_chorus')."""
    return f"{model}_{prompt}_{method}"


def embedding(model: str, prompt: str) -> str:
    """The R-side 'embedding' token, e.g. 'mulan_raw'."""
    return f"{model}_{prompt}"


def all_lmc_columns() -> list[str]:
    return [lmc_col(m, p, meth) for m in MODELS for p in PROMPTS for meth in ALL_METHODS]


def format_prompt(prompt_key: str, text: str, level: str) -> str:
    """Reuse lmcval's templates: 'song' wording at song level, 'song segment' else."""
    return lmcval_config.format_prompt(prompt_key, text, level)


def summary() -> str:
    return (
        f"Models   : {', '.join(MODELS)}\n"
        f"Prompts  : {', '.join(PROMPTS)}\n"
        f"Methods  : {', '.join(ALL_METHODS)}\n"
        f"Grid     : {len(MODELS)}×{len(PROMPTS)} embeddings × 5 structures = "
        f"{len(MODELS)*len(PROMPTS)*5} Stan fits\n"
        f"Project DB : {PROJECT_DB}\n"
        f"Sweep CSV  : {MASTER_SWEEP_CSV}\n"
        f"Stan output: {STAN_OUTPUT_DIR}"
    )
