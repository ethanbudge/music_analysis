"""
config.py — Central configuration for the Lyric-Music Congruence (LMC) pipeline.

Everything downstream keys off the LRCLIB data dump: we sample tracks that have
time-synced lyrics, then enrich those tracks with audio, popularity, and mood.

Paths are resolved relative to the repository root so the project is portable.
API credentials are read from the environment (never hard-coded — see README).
"""

from __future__ import annotations
import os
import glob
from pathlib import Path

# ─── Repository layout ──────────────────────────────────────────────────────────
# config.py lives at  <repo>/src/lmc/config.py  → repo root is two parents up.
REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_DIR    = Path(os.getenv("LMC_DATA_DIR",    REPO_ROOT / "data"))
RESULTS_DIR = Path(os.getenv("LMC_RESULTS_DIR", REPO_ROOT / "results"))

AUDIO_DIR      = DATA_DIR / "audio"          # downloaded MP3s, named <track_id>.mp3
EMBEDDINGS_DIR = DATA_DIR / "embeddings"     # cached per-song .npz embeddings
PROJECT_DB     = DATA_DIR / "project.db"     # progress / corpus tracking (small)

# The LRCLIB dump is large and gitignored. We auto-detect it in DATA_DIR or the
# repo root so the pipeline works whether you leave it where it downloaded or
# move it under data/. Override with the LRCLIB_DUMP environment variable.
def _find_lrclib_dump() -> Path | None:
    env = os.getenv("LRCLIB_DUMP")
    if env:
        return Path(env)
    for base in (DATA_DIR, REPO_ROOT):
        hits = sorted(glob.glob(str(base / "lrclib-db-dump-*.sqlite3")))
        if hits:
            return Path(hits[-1])          # newest dump by name
    return None

LRCLIB_DUMP = _find_lrclib_dump()

# ─── API credentials (read from environment) ─────────────────────────────────────
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID",     "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
YOUTUBE_API_KEY       = os.getenv("YOUTUBE_API_KEY",       "")   # optional
LASTFM_API_KEY        = os.getenv("LASTFM_API_KEY",        "")   # optional

# ─── Embedding models ────────────────────────────────────────────────────────────
# Only the two joint audio-text models are retained; the MERT+SBERT late-fusion
# baseline from the original study has been removed.
MODELS = {
    "mulan": {
        "name":     "MuQ-MuLan",
        "hf_id":    "OpenMuQ/MuQ-MuLan-large",
        "audio_sr": 24_000,
        "dim":      512,
    },
    "clap": {
        "name":     "LAION-CLAP (music)",
        "hf_id":    "laion/larger_clap_music",
        "audio_sr": 48_000,
        "dim":      512,
        "chunk_s":  10.0,    # CLAP works on ~10 s windows; long audio is chunked
    },
}

# ─── Line-level audio context windows ────────────────────────────────────────────
# For each synced lyric line we embed the audio under several padding regimes.
# "exact"  — only the audio spanning the line's own timestamp range.
# "buf1/5/10" — symmetric padding of N seconds before and after the line.
CONTEXT_WINDOWS = {
    "exact": 0.0,
    "buf1":  1.0,
    "buf5":  5.0,
    "buf10": 10.0,
}

# ─── Corpus sampling filters ─────────────────────────────────────────────────────
# Applied when drawing songs from the LRCLIB synced-lyric universe.
SAMPLE_FILTERS = {
    "min_duration_s":  45.0,    # skip clips / interludes
    "max_duration_s":  900.0,   # skip mixes / very long tracks
    "min_synced_lines": 5,      # need enough lines for line-level analysis
    "require_ascii_ratio": 0.5, # crude English/Latin-script bias for embeddings/search
    # Live / alternate recordings carry synced timing for *that* performance, which
    # is hard to match to a clean studio audio source — exclude them by title.
    "exclude_title_keywords": [
        "live", "karaoke", "remix", "sped up", "slowed", "instrumental",
        "acoustic version", "commentary", "demo", "rehearsal", "8d audio",
        "nightcore", "cover)", "(cover", "reverb",
    ],
}

# ─── Mood / acoustic feature columns (librosa) ───────────────────────────────────
MOOD_COLUMNS = [
    "mood_happy", "mood_sad", "mood_relaxed",
    "mood_aggressive", "mood_party",
    "danceability", "voice_instrumental",
]

# ─── LMC method names ────────────────────────────────────────────────────────────
# Methods written to the long-format LMC table. Line-window methods are produced
# per context window in CONTEXT_WINDOWS.
LMC_METHODS = (
    ["song"] +
    [f"line_{w}" for w in CONTEXT_WINDOWS] +
    ["seg_chorus", "seg_nonchorus"]
)


def ensure_dirs() -> None:
    """Create all output directories (safe to call repeatedly)."""
    for d in (DATA_DIR, RESULTS_DIR, AUDIO_DIR, EMBEDDINGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for key in MODELS:
        (EMBEDDINGS_DIR / key).mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """Human-readable configuration summary for notebooks / logs."""
    dump = str(LRCLIB_DUMP) if LRCLIB_DUMP else "NOT FOUND (set LRCLIB_DUMP)"
    return (
        f"Repo root     : {REPO_ROOT}\n"
        f"LRCLIB dump   : {dump}\n"
        f"Project DB    : {PROJECT_DB}\n"
        f"Audio dir     : {AUDIO_DIR}\n"
        f"Embeddings    : {EMBEDDINGS_DIR}\n"
        f"Results dir   : {RESULTS_DIR}\n"
        f"Spotify creds : {'set' if SPOTIFY_CLIENT_ID else 'MISSING'}\n"
        f"YouTube key   : {'set' if YOUTUBE_API_KEY else 'optional/unset'}\n"
        f"Last.fm key   : {'set' if LASTFM_API_KEY else 'optional/unset'}"
    )
