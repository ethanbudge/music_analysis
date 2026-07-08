"""
config.py — Configuration for the generative LMC arm (`lmcgen`).

Paths are resolved relative to the repository root (this file lives at
<repo>/src/lmcgen/config.py) so the project is portable. Every heavy artifact
(generated audio, embeddings, anchors) lands under data/generation/, which is
gitignored; the small tidy result tables land under results/generation/.

ACE-Step 1.5 and MuQ-MuLan are only imported when actually generating / scoring
(see acestep.py, mulan.py) so importing this package is cheap and dependency-free.
"""
from __future__ import annotations
import os
from pathlib import Path

# ─── Repository layout ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]

GEN_DIR       = Path(os.getenv("LMCGEN_DIR", REPO_ROOT / "data" / "generation"))
AUDIO_DIR     = GEN_DIR / "audio"          # generated choruses  <lyric>__<music>__<seed>.wav
TUNE_DIR      = GEN_DIR / "tuning"         # candidate clips from the recipe search
ANCHOR_DIR    = GEN_DIR / "anchors"        # cached MuLan/CLAP emotion-anchor vectors (.npz)
EMB_DIR       = GEN_DIR / "embeddings"     # cached embeddings of generated clips (.npz)
RESULTS_DIR   = Path(os.getenv("LMCGEN_RESULTS_DIR", REPO_ROOT / "results" / "generation"))
FIGURE_DIR    = Path(os.getenv("LMCGEN_FIGURE_DIR", REPO_ROOT / "analysis" / "output" / "figures" / "generation"))

# ─── Experimental design ─────────────────────────────────────────────────────────
# The eight emotions (Plutchik's high-intensity petals). Order is fixed and used
# everywhere (rows/cols of the 8x8 grid, anchor order, plot axes).
EMOTIONS = ["ecstasy", "admiration", "terror", "amazement",
            "grief", "loathing", "rage", "vigilance"]

CHORUS_DURATION_S = 13.0      # short 2-line hook, repeated (ACE-Step min is 10 s). Kept
                              # roughly equal across cells to control exposure time; short
                              # so the vocal renders the words cleanly (see asr.py / WER).
GRID_SEED         = 20260702  # base seed; each cell derives a deterministic seed from this

# ─── Generation backend selector ─────────────────────────────────────────────────
# Which text-to-song engine the pipeline generates with. The pipeline is backend-
# agnostic — each backend is a client exposing generate(GenSpec)->wav; validation
# (WER / VA / MuLan LMC) is shared. "acestep" (local server), "lyria" (Google Vertex/
# Gemini), "suno" (third-party API). Pilot compares two (pipeline.pilot).
BACKEND = os.getenv("LMCGEN_BACKEND", "acestep")

# ─── Lyria 3 (Google, via the Gemini API / google-genai) ─────────────────────────
# Cleanest research licensing; takes user lyrics as [Chorus] tags in the prompt.
# Auth: GEMINI_API_KEY. `pip install google-genai`. Emotion/tempo go in the prompt.
LYRIA = {
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "model":   os.getenv("LYRIA_MODEL", "lyria-3-pro-preview"),  # or lyria-3-clip-preview (30s)
    "wav":     True,   # request WAV (Pro only); MP3 otherwise — transcoded on download either way
}

# ─── Suno v5.5 (via a third-party API provider — Suno has no official public API) ─
# Best lyric/vocal quality. Pick a provider (default api.sunoapi.org) + Bearer token.
# customMode: `prompt` is used STRICTLY as the sung lyrics; `style` carries emotion.
# LICENSING: Suno grants a commercial license, not ownership — verify it covers your
# study + stimulus sharing before relying on it.
SUNO = {
    "base_url":  os.getenv("SUNO_API_URL", "https://api.sunoapi.org"),
    "api_key":   os.getenv("SUNO_API_KEY", ""),
    "model":     os.getenv("SUNO_MODEL", "V5"),
    "poll_interval_s": 5.0,
    "poll_timeout_s":  300.0,
}

# ─── ACE-Step 1.5 settings ───────────────────────────────────────────────────────
# ACE-Step 1.5 is installed via `uv` into ITS OWN isolated environment (separate
# torch/MLX stack), not `pip install`-able into the `lmc` conda env this notebook
# runs in. Rather than importing it in-process (which risks clashing with the `lmc`
# env's carefully pinned torch/numpy — see the CLAP/numpy saga in project history),
# we talk to ACE-Step's own REST API SERVER as a separate process. Start it first:
#   cd /path/to/ACE-Step-1.5 && uv run python -m acestep.api_server
# (or ./start_api_server_macos.sh) — then leave ACESTEP_API_URL pointing at it.
# We drive the DiT directly with explicit caption / bpm / keyscale and DISABLE the
# 5Hz LM planner ("thinking") so our explicit emotion metadata is used verbatim.
# The exact request/response field names are young and under-documented — see
# acestep.py's error messages (which dump the full server response) if a call 422s,
# and cross-check against your running server's auto-docs at <ACESTEP_API_URL>/docs.
ACESTEP = {
    "api_url":         os.getenv("ACESTEP_API_URL", "http://127.0.0.1:8001"),
    "api_key":         os.getenv("ACESTEP_API_KEY", ""),      # only if the server was launched with --api-key
    "model":           os.getenv("ACESTEP_MODEL", "acestep-v15-base"),  # "" = let the server use its default
    "inference_steps": 50,        # base-model quality setting (turbo would be ~8)
    "guidance_scale":  9.0,        # raised from 7 for tighter lyric/caption adherence (too
                                  # high adds vocal-formant artifacts — 8-11 is the sweet spot)
    "thinking":        bool(int(os.getenv("LMCGEN_THINKING", "0"))),  # 5Hz LM planner.
                                  # v2 run showed bpm/keyscale were IGNORED with this off
                                  # (measured tempo didn't track requested) — the planner
                                  # likely consumes the numeric metadata. A/B test it via
                                  # pipeline.ab_tempo_test() before a full regen. Default
                                  # off; set LMCGEN_THINKING=1 to let the planner apply metas.
    "sample_rate":     48_000,    # canonical local sample rate; every clip is transcoded to this on download
    "poll_interval_s": 3.0,       # how often to poll /query_result while a clip generates
    "poll_timeout_s":  1800.0,    # per-clip ceiling (30 min) — generous for a fanless 16 GB Mac
    "request_timeout_s": 30.0,    # per-HTTP-request read timeout. The server can't answer
                                  # while MLX compute holds the GIL, so poll timeouts here are
                                  # EXPECTED and retried (see acestep._poll) up to poll_timeout_s.
}

# ─── MuLan-tuned recipe search ───────────────────────────────────────────────────
# For each musical emotion we generate a few candidate recipes x seeds, embed each
# with MuLan, and keep the caption that lands closest to that emotion's anchor.
# DEFAULT OFF: tuning requires MuLan and the ACE-Step server to be loaded at the SAME
# time, which OOMs a 16 GB Mac. With it off, each emotion uses its hand-written
# default caption (already emotion-targeted). Only enable on a machine with plenty of
# RAM (or a CUDA box) where the two models comfortably co-reside.
TUNE = {
    "candidates_per_emotion": 3,   # how many caption variants to try (see emotions.py captions)
    "seeds_per_candidate":    1,   # generations per candidate during tuning
    "enabled":  bool(int(os.getenv("LMCGEN_TUNE", "0"))),  # off unless LMCGEN_TUNE=1
}

# ─── Validation ──────────────────────────────────────────────────────────────────
# MuLan is the primary validator (same space as the observational arm). CLAP is an
# optional *independent* second validator to guard against the recipe-tuning /
# validation circularity (tuning maximises a MuLan quantity, so a MuLan-only check
# is partly self-fulfilling; CLAP was tuned on nothing here).
VALIDATE = {
    "use_clap": bool(int(os.getenv("LMCGEN_USE_CLAP", "0"))),
}

# ─── Lyric-intelligibility screening (ASR / WER) ─────────────────────────────────
# During generation, transcribe each clip with Whisper and score word error rate
# (WER) vs the target hook. We generate up to `max_takes` per cell (different seeds)
# and KEEP THE LOWEST-WER take — a relative screen, because sung-audio WER is
# inherently high (25-50% even when clearly intelligible). `accept_wer` early-stops
# a cell once a take is good enough. Small (~few hundred MB, CPU/int8) so it's fine
# to run in the generation phase alongside the ACE-Step server; MuLan stays out.
ASR = {
    "enabled":      bool(int(os.getenv("LMCGEN_ASR", "1"))),   # WER screening on by default
    "model_size":   os.getenv("LMCGEN_ASR_MODEL", "small"),    # tiny|base|small|medium (small = good balance)
    "device":       "cpu",
    "compute_type": "int8",
    "language":     "en",
    "beam_size":    5,
    "max_takes":    3,     # generations per cell; keep the lowest-WER one
    "accept_wer":   0.34,  # early-stop a cell once a take reaches this WER (vs SINGLE hook)
    "keep_takes":   False, # if True, keep every take on disk (else only the chosen one)
    # Hallucination guards — Whisper invents filler ("thanks for watching") on quiet/
    # buried-vocal clips, which wrecked the v2 WER. Drop low-confidence / no-speech
    # segments and flag the clip as vocal-absent instead of scoring the hallucination.
    "no_speech_threshold": 0.6,   # drop segments whose no_speech_prob exceeds this
    "logprob_threshold":  -1.0,   # drop segments whose avg_logprob is below this
    "vad_filter":          False, # Silero VAD (can clip singing — off by default)
    "vocal_separation":    False, # run Demucs to isolate the vocal stem before ASR
                                  # (needs `pip install demucs`; much cleaner, slower)
}

# ─── Dry-run switch ──────────────────────────────────────────────────────────────
# When True, acestep.generate_clip() synthesises a cheap emotion-dependent waveform
# instead of calling the 3.5B model. This lets the ENTIRE pipeline (lyrics, anchors,
# MuLan validation, stats, plots, notebook) run end-to-end for plumbing checks
# before committing hours of MPS generation. Flip to False for the real study.
DRY_RUN = bool(int(os.getenv("LMCGEN_DRY_RUN", "1")))


def ensure_dirs() -> None:
    """Create all output directories (safe to call repeatedly)."""
    for d in (GEN_DIR, AUDIO_DIR, TUNE_DIR, ANCHOR_DIR, EMB_DIR, RESULTS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """Human-readable configuration summary for the notebook / logs."""
    return (
        f"Repo root      : {REPO_ROOT}\n"
        f"Generation dir : {GEN_DIR}\n"
        f"Results dir    : {RESULTS_DIR}\n"
        f"Figures dir    : {FIGURE_DIR}\n"
        f"Emotions       : {', '.join(EMOTIONS)}\n"
        f"Grid           : {len(EMOTIONS)} lyrics x {len(EMOTIONS)} music = "
        f"{len(EMOTIONS) ** 2} hooks (~{CHORUS_DURATION_S:.0f}s each, single genre)\n"
        f"ACE-Step model : {ACESTEP['model'] or '(server default)'} via {ACESTEP['api_url']} "
        f"(steps={ACESTEP['inference_steps']}, guidance={ACESTEP['guidance_scale']}, "
        f"thinking={ACESTEP['thinking']})\n"
        f"Lyric screening: {'on' if ASR['enabled'] else 'off'} "
        f"(Whisper {ASR['model_size']}, best of {ASR['max_takes']} takes by WER)\n"
        f"Recipe tuning  : {'on' if TUNE['enabled'] else 'off'} "
        f"({TUNE['candidates_per_emotion']} candidates x {TUNE['seeds_per_candidate']} seeds)\n"
        f"CLAP validator : {'on' if VALIDATE['use_clap'] else 'off'}\n"
        f"DRY_RUN        : {DRY_RUN}   {'(MOCK audio — plumbing only)' if DRY_RUN else '(REAL ACE-Step generation)'}"
    )
