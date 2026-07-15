"""
config.py — configuration for the lmcmetrics package.

Reuses the observational arm's paths and DB (single source of truth) and adds a
small set of knobs for the new metrics. New artifacts are written under
`data/lmcmetrics/` and `results/lmcmetrics/` so they never collide with the
existing pipeline outputs.

Env overrides (all optional):
  LMCMETRICS_TEXT_ENCODER   HuggingFace/sentence-transformers model id for lyrics.
  LMCMETRICS_AUDIO_REP      which cached audio representation to use as the audio
                            tower: 'mert' (default, 1024-d) | 'mulan' | 'clap'.
  LMCMETRICS_PROJ_DIM       LyricLMC shared-space dimension (default 256).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# ── Make the sibling `lmc` package importable no matter how we're launched ───────
# lmcmetrics/config.py -> parents[1] is <repo>/src, which holds both `lmc` and us.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lmc import config as lmc_config          # noqa: E402  (path set above)

# ── Paths (piggyback on the existing layout) ─────────────────────────────────────
REPO_ROOT      = lmc_config.REPO_ROOT
DATA_DIR       = lmc_config.DATA_DIR
RESULTS_DIR    = lmc_config.RESULTS_DIR
PROJECT_DB     = lmc_config.PROJECT_DB
EMBEDDINGS_DIR = lmc_config.EMBEDDINGS_DIR
MERT_DIR       = lmc_config.MERT_DIR

# New, package-owned output roots (gitignored via data/ and results/).
METRICS_DATA_DIR   = DATA_DIR / "lmcmetrics"
LYRICS_VEC_DIR     = METRICS_DATA_DIR / "lyrics"      # cached lyric text-tower vectors
LYRICLMC_RUNS_DIR  = METRICS_DATA_DIR / "lyriclmc"    # one folder per training run
METRICS_RESULTS_DIR = RESULTS_DIR / "lmcmetrics"      # eval tables, per-song scores

# ── Lyric text tower (the fix for MuLan's caption-trained text encoder) ───────────
# Default: a small, fast, robust general sentence encoder. Swap via the env var.
#   Topic-leaning stronger option : "intfloat/e5-base-v2"
#   Affect-leaning research option : an emotion-tuned encoder (document your choice)
TEXT_ENCODER = os.getenv("LMCMETRICS_TEXT_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")

# Which cached audio representation feeds LyricLMC's audio tower and the geometry
# comparisons. MERT is the default: it is audio-only, rich, already cached by the
# pipeline (data/embeddings/mert/<id>.npy), and independent of the MuLan/CLAP space
# we are trying to move away from.
AUDIO_REP = os.getenv("LMCMETRICS_AUDIO_REP", "mert")   # 'mert' | 'mulan' | 'clap'

# ── LyricLMC (Step 3) hyper-parameters ───────────────────────────────────────────
LYRICLMC = {
    "proj_dim":    int(os.getenv("LMCMETRICS_PROJ_DIM", "256")),  # shared-space dim d
    "hidden_dim":  512,       # projection-head hidden width
    "dropout":     0.1,
    "batch_size":  256,       # in-batch negatives = batch_size - 1
    "epochs":      60,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "val_frac":    0.15,      # held-out fraction for early stopping / honest AUC
    "init_logit_scale": 2.6592,  # = ln(1/0.07), the CLIP initialisation
    "max_logit_scale":  4.6052,  # = ln(100), clamp so the temperature can't collapse
    "patience":    12,        # early-stop patience (epochs w/o val-AUC improvement)
    "seed":        42,
}

# ── Evaluation (Step 2) defaults ──────────────────────────────────────────────────
EVAL = {
    "recall_at":       (1, 5, 10),
    "impostor_cap":    None,   # None = use ALL other songs as impostors (fine < ~5k)
    "seed":            42,
}

# ── Geometry (Step 4) defaults ────────────────────────────────────────────────────
GEOMETRY = {
    "rdm_metric":  "cosine",   # dissimilarity for RSA: 'cosine' | 'euclidean'
    "local_k":     None,       # None = per-song RSA over ALL other songs (robust);
                               # an int = restrict to that many nearest audio neighbours
}


def ensure_dirs() -> None:
    """Create the package's output directories (safe to call repeatedly)."""
    for d in (METRICS_DATA_DIR, LYRICS_VEC_DIR, LYRICLMC_RUNS_DIR, METRICS_RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    return (
        f"Repo root       : {REPO_ROOT}\n"
        f"Project DB      : {PROJECT_DB}\n"
        f"Text encoder    : {TEXT_ENCODER}\n"
        f"Audio rep       : {AUDIO_REP}\n"
        f"LyricLMC dim    : {LYRICLMC['proj_dim']}\n"
        f"Runs dir        : {LYRICLMC_RUNS_DIR}\n"
        f"Results dir     : {METRICS_RESULTS_DIR}"
    )
