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

# ─── Joint audio-text embedding models (used to compute LMC) ─────────────────────
# Two joint models. CLAP is loaded via the official `laion_clap` package (the HF
# `transformers` ClapModel silently failed to load this checkpoint's projection
# weights — see git history / MODEL_NOTES). Point LMC_CLAP_CKPT at the downloaded
# music checkpoint (music_audioset_epoch_15_esc_90.14.pt); otherwise laion_clap
# downloads its default general checkpoint (not music-specialised).
MODELS = {
    "mulan": {
        "name":     "MuQ-MuLan",
        "hf_id":    "OpenMuQ/MuQ-MuLan-large",
        "audio_sr": 24_000,
        "dim":      512,
    },
    "clap": {
        "name":     "LAION-CLAP (music)",
        "amodel":   "HTSAT-base",                 # audio backbone for laion_clap
        "ckpt":     "",                           # filled by _find_clap_ckpt() below
        "enable_fusion": False,
        "audio_sr": 48_000,
        "dim":      512,
        "chunk_s":  10.0,    # CLAP works on ~10 s windows; long audio is chunked
    },
}


def _find_clap_ckpt() -> str:
    """Locate the LAION-CLAP *music* checkpoint without re-downloading.

    Order: LMC_CLAP_CKPT env var → the shared HuggingFace cache (where
    `lukewys/laion_clap` lands, reused across conda envs) → "" (then _CLAP falls
    back to laion_clap's default general checkpoint with a warning).
    """
    env = os.getenv("LMC_CLAP_CKPT")
    if env:
        return env
    pat = str(Path.home() / ".cache/huggingface/hub/models--lukewys--laion_clap"
              / "snapshots/*/music_audioset_*.pt")
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else ""


MODELS["clap"]["ckpt"] = _find_clap_ckpt()

# ─── MERT audio controls (replaces / augments the librosa mood proxies) ──────────
# MERT is an audio-only self-supervised representation. Because it is a DIFFERENT
# representation than the MuLan/CLAP space that LMC lives in, its principal
# components are valid *off-LMC-path* controls (they soak up production/era/genre
# nuisance variance without partialling out LMC's own inputs — see MODEL_NOTES §4.5).
MERT = {
    "name":     "MERT-v1-330M",
    "hf_id":    "m-a-p/MERT-v1-330M",
    "audio_sr": 24_000,
    "dim":      1024,
    "chunk_s":  10.0,        # embed in ~10 s windows + average — a whole song in one
                            #   forward pass blows the Apple-MPS buffer cap (>11 GiB).
    "pca_k":    10,          # number of MERT principal components used as controls
}
MERT_DIR = EMBEDDINGS_DIR / "mert"   # cached per-song MERT vectors (<track_id>.npy)

# ─── Genre ensemble ──────────────────────────────────────────────────────────────
# Coarse genre clusters (the modelling vocabulary). Genre is recovered by a cascade
# (see genre.py): Spotify artist tags → MusicBrainz/Discogs tags → zero-shot from
# the MuLan/CLAP embedding, recording the source and a confidence for each song.
GENRE_VOCAB = ["hip-hop", "country", "folk", "electronic", "rock", "r&b", "latin", "pop"]
GENRE_ZEROSHOT_MODEL  = "mulan"   # which joint embedding backs zero-shot genre
GENRE_ZEROSHOT_PROMPT = "This is a {genre} song."
MUSICBRAINZ_APP = ("lmc-research", "0.1", "ethanbudge2000@gmail.com")  # MB API user-agent

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
    for d in (DATA_DIR, RESULTS_DIR, AUDIO_DIR, EMBEDDINGS_DIR, MERT_DIR):
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
