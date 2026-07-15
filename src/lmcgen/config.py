"""
config.py — Configuration for the generative LMC arm (`lmcgen`), VA-quadrant design.

The generative arm manufactures stimuli for a human-validation survey of Lyric-Music
Congruence. The design is a 2x2 valence/arousal (VA) circumplex:

    4 VA corners (music targets)  x  16 two-line lyrics (4 authored per corner)
    x  4 repetitions              =  256 song-lyric pairs.

Songs are generated with Google **Lyria 3 Clip** via the Gemini API (google-genai).
Lyria takes lyrics + style as prompt TEXT only — it cannot consume a target embedding,
has no seed and does not reproduce — so each VA corner is hit by prompt engineering and
validated post-hoc against a MuLan quadrant anchor (audio-vs-anchor cosine) plus an
independent librosa acoustic-VA measure. Per the study design we generate exactly
`REPS_PER_CELL` takes per cell and validate all of them; winners are chosen afterwards.

Paths resolve relative to the repo root so the project is portable. Heavy artifacts
(audio, embeddings, anchors) land under data/generation/ (gitignored); the tidy result
tables land under results/generation/.
"""
from __future__ import annotations
import os
from pathlib import Path

# ─── Repository layout ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]

GEN_DIR       = Path(os.getenv("LMCGEN_DIR", REPO_ROOT / "data" / "generation"))
AUDIO_DIR     = GEN_DIR / "audio"          # generated songs  <lyric_id>__<music_q>__rep<k>.wav
ANCHOR_DIR    = GEN_DIR / "anchors"        # cached MuLan/CLAP quadrant-anchor vectors (.npz)
EMB_DIR       = GEN_DIR / "embeddings"     # cached embeddings of generated clips + lyrics (.npz)
SURVEY_DIR    = GEN_DIR / "survey"         # exported, loudness-normalised survey stimuli + manifest
RESULTS_DIR   = Path(os.getenv("LMCGEN_RESULTS_DIR", REPO_ROOT / "results" / "generation"))
FIGURE_DIR    = Path(os.getenv("LMCGEN_FIGURE_DIR", REPO_ROOT / "analysis" / "output" / "figures" / "generation"))

# ─── Experimental design ─────────────────────────────────────────────────────────
# The four extreme VA corners (circumplex). Order is fixed and used everywhere
# (rows/cols of the 4x4 matrix, anchor order, plot axes).
#   hvha = high valence / high arousal   (happy, excited, euphoric)
#   hvla = high valence / low  arousal   (calm, content, serene)
#   lvha = low  valence / high arousal   (angry, afraid, tense)
#   lvla = low  valence / low  arousal   (sad, weary, hopeless)
QUADRANTS = ["hvha", "hvla", "lvha", "lvla"]

LYRICS_PER_QUADRANT = 4     # 4 lyrics x 4 corners = 16 frozen lyric stimuli (fixed; see lyrics.py)

# ─── Smoke-test toggles ───────────────────────────────────────────────────────────
# Both default to the full study design (4 x 4 -> 256 songs). Lower either one to run
# a small, cheap smoke test before spending API quota on the full batch — e.g. 1 lyric
# per quadrant x 1 rep = 16 songs, one per cell of the 4x4 congruence matrix. Read live
# at call time by generate.build_specs(), so set them directly in the notebook:
#     from lmcgen import config
#     config.ACTIVE_LYRICS_PER_QUADRANT = 1   # how many of the 4 authored lyrics/corner to use
#     config.REPS_PER_CELL = 1                # generations per (lyric x music-quadrant) cell
# generate.active_lyrics() clamps ACTIVE_LYRICS_PER_QUADRANT to [1, LYRICS_PER_QUADRANT].
ACTIVE_LYRICS_PER_QUADRANT = int(os.getenv("LMCGEN_LYRICS_PER_QUADRANT", str(LYRICS_PER_QUADRANT)))
REPS_PER_CELL       = int(os.getenv("LMCGEN_REPS_PER_CELL", "4"))
                            # takes per (lyric x music-quadrant) cell -> at defaults 16*4*4 = 256 songs
CANDIDATES_PER_SLOT = int(os.getenv("LMCGEN_CANDIDATES", "1"))
                            # 1 = generate each rep once (the study default: no selection).
                            # >1 = best-of-N per rep, keeping the take closest to target
                            # (lowest WER + highest anchor cosine). Available but off.

CLIP_DURATION_S = 30.0      # Lyria 3 Clip is a fixed 30 s clip; used for the mock + validation windows
SAMPLE_RATE     = 44_100    # Lyria 3 Clip is 44.1 kHz; canonical local SR (every clip transcoded to this)

# ─── Lyria 3 Clip (Google, via the Gemini API / google-genai) ────────────────────
# client.interactions.create(model=..., input=<prompt with [Chorus] lyrics>) -> base64
# MP3 in interaction.output_audio.data (+ echoed lyrics in interaction.output_text).
# Voice / tempo / key / mood are expressed in natural language in the prompt; there are
# NO seed / negative-prompt / embedding fields. Auth: GEMINI_API_KEY. `pip install
# google-genai`. All generated audio carries a SynthID watermark (note for redistribution).
LYRIA = {
    "api_key":         os.getenv("GEMINI_API_KEY", ""),
    "model":           os.getenv("LYRIA_MODEL", "lyria-3-clip-preview"),  # 30 s clip
    "max_retries":     4,        # transient / rate-limit retries (exponential backoff)
    "retry_backoff_s": 8.0,      # base backoff; doubles each retry
}

# A single fixed voice descriptor, prepended to every prompt so the singer stays ~constant
# across the 256 clips. Lyria can't GUARANTEE an identical voice from text (voice
# consistency is secondary to song fidelity here), but a fixed, detailed descriptor helps.
VOICE_BLURB = ("Solo female vocalist, warm natural alto, clear and well-enunciated diction, "
               "no autotune, singing the lyrics exactly as written")

# ─── Lyric-intelligibility screening (ASR / WER) ─────────────────────────────────
# Transcribe each clip with Whisper and score word error rate (WER) vs the target
# couplet. WER is a per-clip lyric-presence control (sung-audio WER is inherently high,
# 25-50% even when clearly intelligible, so read it as a relative screen). When
# CANDIDATES_PER_SLOT>1, the lowest-WER take is kept per rep. Small (CPU/int8).
ASR = {
    "enabled":      bool(int(os.getenv("LMCGEN_ASR", "1"))),
    "model_size":   os.getenv("LMCGEN_ASR_MODEL", "small"),    # tiny|base|small|medium
    "device":       "cpu",
    "compute_type": "int8",
    "language":     "en",
    "beam_size":    5,
    "max_takes":    3,     # only used when CANDIDATES_PER_SLOT>1 (best-of-N per rep)
    "accept_wer":   0.34,  # early-stop a best-of-N search once a take reaches this WER
    "keep_takes":   False,
    # Hallucination guards — Whisper invents filler ("thanks for watching") on quiet/
    # buried-vocal clips. Drop low-confidence / no-speech segments; flag vocal-absent.
    "no_speech_threshold": 0.6,
    "logprob_threshold":  -1.0,
    "vad_filter":          False,
    "vocal_separation":    False,   # run Demucs to isolate the vocal stem before ASR
}

# ─── Validation ──────────────────────────────────────────────────────────────────
# MuLan is the primary embedding validator (same space as the observational arm).
# CLAP is an optional independent second validator. Acoustic VA (librosa, va.py) is the
# genre-robust emotion instrument and is always computed.
VALIDATE = {
    "use_clap": bool(int(os.getenv("LMCGEN_USE_CLAP", "0"))),
}

# ─── Survey export ───────────────────────────────────────────────────────────────
SURVEY = {
    "target_lufs":  -14.0,   # integrated loudness to normalise every stimulus to
    "per_cell":     1,       # how many of the REPS_PER_CELL takes to export per cell (the winners)
}

# ─── Dry-run switch ──────────────────────────────────────────────────────────────
# True -> synthesise a cheap VA-dependent mock clip instead of calling Lyria, so the
# whole pipeline (lyrics, anchors, generation, WER/MuLan/VA validation, stats, plots,
# notebook) runs end-to-end for plumbing checks before spending API quota. False -> real.
DRY_RUN = bool(int(os.getenv("LMCGEN_DRY_RUN", "1")))


def ensure_dirs() -> None:
    """Create all output directories (safe to call repeatedly)."""
    for d in (GEN_DIR, AUDIO_DIR, ANCHOR_DIR, EMB_DIR, SURVEY_DIR, RESULTS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """Human-readable configuration summary for the notebook / logs."""
    active_per_corner = max(1, min(ACTIVE_LYRICS_PER_QUADRANT, LYRICS_PER_QUADRANT))
    n_lyrics_active = len(QUADRANTS) * active_per_corner
    n_songs = n_lyrics_active * len(QUADRANTS) * REPS_PER_CELL
    is_smoke = (active_per_corner < LYRICS_PER_QUADRANT) or (REPS_PER_CELL < 4)
    return (
        f"Repo root      : {REPO_ROOT}\n"
        f"Generation dir : {GEN_DIR}\n"
        f"Results dir    : {RESULTS_DIR}\n"
        f"Figures dir    : {FIGURE_DIR}\n"
        f"VA corners     : {', '.join(QUADRANTS)}\n"
        f"Design         : {n_lyrics_active} lyrics ({active_per_corner}/{LYRICS_PER_QUADRANT} per corner) x "
        f"{len(QUADRANTS)} music corners x {REPS_PER_CELL} reps = {n_songs} songs "
        f"(~{CLIP_DURATION_S:.0f}s each)" + ("   [SMOKE TEST]" if is_smoke else "") + "\n"
        f"Backend        : Lyria — {LYRIA['model']} (candidates/slot={CANDIDATES_PER_SLOT})\n"
        f"Lyric screening: {'on' if ASR['enabled'] else 'off'} (Whisper {ASR['model_size']})\n"
        f"CLAP validator : {'on' if VALIDATE['use_clap'] else 'off'}\n"
        f"GEMINI_API_KEY : {'set' if LYRIA['api_key'] else 'MISSING — set it in notebooks/.env'}\n"
        f"DRY_RUN        : {DRY_RUN}   {'(MOCK audio — plumbing only)' if DRY_RUN else '(REAL Lyria generation)'}"
    )
