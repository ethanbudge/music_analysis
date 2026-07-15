"""
config.py — configuration for the lmcval spot-test harness.

Reuses the main pipeline's config/creds (single source of truth) and adds the
validation-specific knobs: the playlist under test, the four models, the three
prompt templates, and the three segmentation levels.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# ── make the main `lmc` package importable however we're launched ────────────────
# validation/lmcval/config.py -> parents[2] is the repo root; src/ holds `lmc`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _load_dotenv() -> str | None:
    """Load a .env into the environment BEFORE lmc.config reads SPOTIFY_* at import.

    Searches (first match wins, existing env vars are NOT overridden):
      $LMCVAL_ENV  →  <repo>/.env  →  <repo>/notebooks/.env  →  ./.env
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return None
    for cand in (os.getenv("LMCVAL_ENV"), _REPO_ROOT / ".env",
                 _REPO_ROOT / "notebooks" / ".env", Path.cwd() / ".env"):
        if cand and Path(cand).is_file():
            load_dotenv(cand, override=False)
            return str(cand)
    return None


ENV_FILE = _load_dotenv()

from lmc import config as lmc_config       # noqa: E402  (path + env set above)

REPO_ROOT = _REPO_ROOT

# ── playlist under test ───────────────────────────────────────────────────────────
# "Mad World" — 5 covers with identical lyrics but very different musical treatments,
# a clean probe for whether a metric ranks congruent arrangements above incongruent
# ones (e.g. the sparse Gary Jules piano version vs. the brighter synthpop original).
PLAYLIST_URL = "https://open.spotify.com/playlist/6zQOgZsQJaif2Ma5NidPfc"
PLAYLIST_ID = os.getenv("LMCVAL_PLAYLIST_ID", "6zQOgZsQJaif2Ma5NidPfc")

# ── paths (validation-owned; gitignored) ──────────────────────────────────────────
VAL_DIR      = REPO_ROOT / "validation"
DATA_DIR     = VAL_DIR / "data"
AUDIO_DIR    = DATA_DIR / "audio"          # <slug>.mp3
LYRICS_DIR   = DATA_DIR / "lyrics"         # <slug>.lrc  (per-cover synced lyrics)
CACHE_DIR    = DATA_DIR / "cache"          # manifest.json, temp embedding I/O
RESULTS_DIR  = VAL_DIR / "results"         # the three output CSVs + summaries

# ── audio ─────────────────────────────────────────────────────────────────────────
BASE_SR = 48_000        # songs are loaded/sliced once at this SR; each model
                        # resamples to its own rate. 48 kHz = highest any model needs.
LINE_WINDOW_PAD = 0.0   # seconds of context padding around each line (0 = exact line).
                        # Set >0 (e.g. 1.0) to give line-level audio some surroundings.

# ── the four models under test ────────────────────────────────────────────────────
# Keys are used in the CSV score-column names (e.g. "mulan__raw").
MODEL_KEYS = ["mulan", "laion_clap", "ms_clap", "clamp3"]
MODEL_DISPLAY = {
    "mulan":      "MuQ-MuLan",
    "laion_clap": "LAION-CLAP (music)",
    "ms_clap":    "Microsoft CLAP (2023)",
    "clamp3":     "CLaMP 3",
}

# Microsoft CLAP version: '2022' | '2023' | 'clapcap'. 2023 is the strongest general.
MSCLAP_VERSION = os.getenv("LMCVAL_MSCLAP_VERSION", "2023")

# CLaMP 3 is script-based: clone https://github.com/sanderwood/clamp3 and point here.
# If unset or the run fails, CLaMP 3 columns are filled with NaN (a warning is logged)
# so the other three models still produce complete results.
CLAMP3_REPO   = os.getenv("LMCVAL_CLAMP3_REPO", "")          # path to the cloned repo
CLAMP3_PYTHON = os.getenv("LMCVAL_CLAMP3_PYTHON", sys.executable)  # its python (own env OK)
CLAMP3_SCRIPT = "clamp3_embd.py"

# ── the three prompt templates ────────────────────────────────────────────────────
# {unit} is "song" at the song level and "song segment" at the segment/line levels.
# {text} is the (raw) lyric text of the unit.
PROMPT_KEYS = ["raw", "contains", "idea"]
PROMPTS = {
    "raw":      "{text}",
    "contains": "a {unit} that contains the lyrics {text}",
    "idea":     "a {unit} representing the idea of the following lyrics: {text}",
}
PROMPT_DISPLAY = {
    "raw":      "raw lyrics",
    "contains": "'a {unit} that contains the lyrics …'",
    "idea":     "'a {unit} representing the idea of …'",
}

# Optional extra prompts you can add to PROMPTS to widen the sweep (see README /
# my recommendations). Enable by merging into PROMPTS before a run.
OPTIONAL_PROMPTS = {
    "label":  "lyrics: {text}",                                   # minimal framing
    "mood":   "a {unit} whose mood matches the lyrics {text}",    # affect-oriented
    "about":  "a {unit} about {text}",                            # topical framing
}


def unit_word(level: str) -> str:
    """'song' for the song level; 'song segment' for segment / line levels."""
    return "song" if level == "song" else "song segment"


def format_prompt(prompt_key: str, text: str, level: str,
                  prompts: dict | None = None) -> str:
    tpl = (prompts or PROMPTS)[prompt_key]
    if "{unit}" in tpl:
        return tpl.format(unit=unit_word(level), text=text)
    return tpl.format(text=text)


def ensure_dirs() -> None:
    for d in (DATA_DIR, AUDIO_DIR, LYRICS_DIR, CACHE_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    return (
        f"Playlist   : {PLAYLIST_URL}\n"
        f"Models     : {', '.join(MODEL_DISPLAY[k] for k in MODEL_KEYS)}\n"
        f"Prompts    : {', '.join(PROMPT_KEYS)}\n"
        f"Levels     : song, segment, line\n"
        f"MS-CLAP    : version {MSCLAP_VERSION}\n"
        f"CLaMP 3    : {'repo=' + CLAMP3_REPO if CLAMP3_REPO else 'NOT configured (columns will be NaN)'}\n"
        f"Results dir: {RESULTS_DIR}\n"
        f".env file  : {ENV_FILE or 'none found'}\n"
        f"Spotify    : {'creds set' if lmc_config.SPOTIFY_CLIENT_ID else 'MISSING (needed to read the playlist)'}"
    )
