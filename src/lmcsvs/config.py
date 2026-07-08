"""
config.py — Configuration for the SVS (singing-voice-synthesis) route.

Reuses the emotion definitions and 2-line hooks from `lmcgen` so this is a drop-in
alternative renderer, not a from-scratch redesign. Everything is deterministic; the
only non-code asset is the fixed voice you assign in the SVS app.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SVS_DIR    = Path(os.getenv("LMCSVS_DIR", REPO_ROOT / "data" / "svs"))
SCORE_DIR  = SVS_DIR / "scores"      # exported MusicXML, one per grid cell
AUDIO_DIR  = SVS_DIR / "audio"       # rendered vocals (from the SVS app) go here
MIX_DIR    = SVS_DIR / "mixed"       # vocal + backing mixes (later stage)
RESULTS_DIR = Path(os.getenv("LMCSVS_RESULTS_DIR", REPO_ROOT / "results" / "svs"))

# ─── Which emotions (grid rows/cols) ─────────────────────────────────────────────
# Defaults to the full lmcgen set. Set LMCSVS_EMOTIONS to a comma list to subset —
# e.g. "rage,ecstasy,grief,admiration" for a 2x2 valence×arousal design.
def _emotions() -> list[str]:
    from lmcgen.config import EMOTIONS
    env = os.getenv("LMCSVS_EMOTIONS")
    return [e.strip() for e in env.split(",")] if env else list(EMOTIONS)

EMOTIONS = _emotions()

# ─── Fixed voice (assigned once in the SVS app) ──────────────────────────────────
# We can't set the voice bank from MusicXML, but we record the intended one so the
# workflow is documented and reproducible. Pick ONE and never change it across cells.
VOICE = {
    "engine":     os.getenv("LMCSVS_ENGINE", "synthv"),   # synthv | ace-studio | diffsinger
    "voice_name": os.getenv("LMCSVS_VOICE", "(assign one fixed voice in the app)"),
    "language":   "english",
}

# ─── Melody design (valence/arousal → notes) ─────────────────────────────────────
MELODY = {
    "beats_per_bar":    4,
    "bars":             2,        # a 2-line hook spans ~2 bars
    "base_octave":      4,        # register floor (tonic sits here at arousal 0)
    "register_arousal": 12,       # semitones the register rises across arousal 0→1 (an octave,
                                  # so arousal dominates the key's tonic placement)
    "seed":             7,        # base seed; each emotion offsets it (reproducible)
    "melisma_prob":     0.0,      # v1: one syllable per note (no melisma)
}

# ─── Export ──────────────────────────────────────────────────────────────────────
EXPORT = {
    "also_midi": bool(int(os.getenv("LMCSVS_MIDI", "0"))),   # also write a .mid (needs `mido`)
}

# ─── DiffSinger (headless, open-source render) ───────────────────────────────────
# DiffSinger renders our scores to vocals from Python — no GUI. It needs a voicebank
# (acoustic model + vocoder + phoneme dictionary) and the inference code, which you
# install separately; point these at your install. `.ds` is DiffSinger's score format.
# NOTE: the .ds schema + phoneme set are voicebank-specific — verify against yours.
DIFFSINGER = {
    "repo":   os.getenv("DIFFSINGER_REPO", ""),   # cloned OpenVPI DiffSinger dir
    "exp":    os.getenv("DIFFSINGER_EXP", ""),     # acoustic experiment name (folder under checkpoints/)
    "spk":    os.getenv("DIFFSINGER_SPK", ""),     # speaker name for multi-speaker banks (optional)
    "python": os.getenv("DIFFSINGER_PYTHON", "python"),  # interpreter for DiffSinger's own env
}
DS_DIR = SVS_DIR / "ds"                                     # exported .ds score files


def ensure_dirs() -> None:
    for d in (SVS_DIR, SCORE_DIR, AUDIO_DIR, MIX_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    n = len(EMOTIONS)
    return (
        f"Repo root   : {REPO_ROOT}\n"
        f"SVS dir     : {SVS_DIR}\n"
        f"Scores      : {SCORE_DIR}  (MusicXML for the SVS app)\n"
        f"Emotions    : {', '.join(EMOTIONS)}\n"
        f"Grid        : {n} lyrics x {n} melodies = {n * n} scores\n"
        f"Engine      : {VOICE['engine']}  (voice: {VOICE['voice_name']})\n"
        f"Melody      : {MELODY['bars']} bars, base octave {MELODY['base_octave']}, "
        f"one syllable/note\n"
        f"Also MIDI   : {EXPORT['also_midi']}"
    )
