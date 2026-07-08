"""
asr.py — Lyric-intelligibility screening via ASR (Whisper) + word error rate.

The experiment needs the sung lyrics to be *exactly* the intended ones. ACE-Step
(like any music generator) doesn't guarantee that, so we screen every clip: run
Whisper on it, transcribe the vocal, and compute the word error rate (WER) against
the target hook. `pipeline` uses this to pick the best of N takes per cell (see
Transcriber.best_of + pipeline._generate_cell_screened) and records the WER as a
per-clip intelligibility control you can report.

Uses `faster-whisper` (CTranslate2) — it runs efficiently on CPU/int8 without
loading torch, so it's light on a 16 GB Mac and doesn't fight the other models.
Install once in the `lmc` env:  pip install faster-whisper

Note on singing: ASR WER on *sung* audio is inherently high (melisma, held notes,
pitch) — 25-50% even for clearly intelligible vocals. So WER here is best used as a
RELATIVE screen (keep the lowest-WER take), not an absolute pass/fail bar. The
config thresholds reflect that.
"""
from __future__ import annotations
import logging
import re

from . import config as C

logger = logging.getLogger(__name__)


class Transcriber:
    """Lazy Whisper wrapper. Loads the model once; transcribes audio files to text."""

    def __init__(self, model_size: str | None = None):
        self.model_size = model_size or C.ASR["model_size"]
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:                                   # noqa: BLE001
            raise ImportError(
                "faster-whisper is required for lyric-intelligibility screening.\n"
                "Install it in the lmc env:  pip install faster-whisper\n"
                "(or disable screening: config.ASR['enabled'] = False / env LMCGEN_ASR=0)"
            ) from e
        logger.info("Loading Whisper (%s, cpu/int8) for lyric screening…", self.model_size)
        self._model = WhisperModel(self.model_size, device=C.ASR["device"],
                                   compute_type=C.ASR["compute_type"])

    def transcribe(self, audio_path) -> str:
        return self.transcribe_detailed(audio_path)["text"]

    def transcribe_detailed(self, audio_path) -> dict:
        """Transcribe with hallucination guards. Returns {text, vocal_present,
        mean_logprob, dropped}. Segments that look like non-speech (high
        no_speech_prob, low avg_logprob) or match a known hallucination phrase are
        dropped; if nothing survives, vocal_present=False (the vocal was buried/absent
        rather than mis-sung)."""
        self._load()
        path = _maybe_separate_vocals(audio_path)
        segments, _ = self._model.transcribe(
            str(path), language=C.ASR["language"], beam_size=C.ASR["beam_size"],
            condition_on_previous_text=False, vad_filter=C.ASR["vad_filter"],
            no_speech_threshold=C.ASR["no_speech_threshold"],
            log_prob_threshold=C.ASR["logprob_threshold"])
        kept, logps, dropped = [], [], 0
        for s in segments:
            nsp = getattr(s, "no_speech_prob", None)
            alp = getattr(s, "avg_logprob", None)
            txt = s.text.strip()
            if (nsp is not None and nsp > C.ASR["no_speech_threshold"]) or \
               (alp is not None and alp < C.ASR["logprob_threshold"]) or \
               _HALLUCINATION.search(txt):
                dropped += 1
                continue
            kept.append(txt)
            if alp is not None:
                logps.append(alp)
        text = " ".join(kept).strip()
        return {"text": text, "vocal_present": bool(text),
                "mean_logprob": (sum(logps) / len(logps)) if logps else None,
                "dropped": dropped}

    def wer_of(self, audio_path, reference: str) -> dict:
        """Transcribe `audio_path` and return
        {wer, transcript, vocal_present, mean_logprob}."""
        d = self.transcribe_detailed(audio_path)
        return {"wer": word_error_rate(reference, d["text"]),
                "transcript": d["text"], "vocal_present": d["vocal_present"],
                "mean_logprob": d["mean_logprob"]}


# Common Whisper hallucinations on non-speech / buried-vocal audio.
_HALLUCINATION = re.compile(
    r"(thanks?\s+(you\s+)?for\s+watching|please\s+subscribe|like\s+and\s+subscribe"
    r"|see\s+you\s+(next|in)|thank\s+you\s+for\s+your|subtitles?\s+by|www\.|\.com)",
    re.IGNORECASE)


def _maybe_separate_vocals(audio_path):
    """If config.ASR['vocal_separation'], isolate the vocal stem with Demucs (cached
    next to the clip). Falls back to the original file if Demucs isn't installed."""
    if not C.ASR["vocal_separation"]:
        return audio_path
    from pathlib import Path
    src = Path(audio_path)
    out = src.with_name(src.stem + ".vocals.wav")
    if out.exists():
        return out
    try:
        import demucs.separate  # noqa: F401
        import subprocess, sys, tempfile, shutil, os
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals",
                            "-o", td, str(src)], check=True, capture_output=True)
            hits = list(Path(td).rglob("vocals.wav"))
            if hits:
                shutil.copyfile(hits[0], out)
                return out
    except Exception as e:                                          # noqa: BLE001
        logger.warning("  vocal separation unavailable (%s); using full mix", e)
    return audio_path


# ─── text normalisation + WER (self-contained; no jiwer dependency) ──────────────
_PUNCT = re.compile(r"[^a-z0-9\s]")


def _normalise(text: str) -> list[str]:
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER = Levenshtein word-edit distance / #reference words (normalised:
    lowercased, punctuation stripped). Clamped to [0, 1] for reporting."""
    ref = _normalise(reference)
    hyp = _normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    d = _edit_distance(ref, hyp)
    return min(1.0, d / len(ref))


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance (iterative, O(len(a)*len(b)) time, O(len(b)) space)."""
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,          # deletion
                         cur[j - 1] + 1,       # insertion
                         prev[j - 1] + (wa != wb))  # substitution
        prev = cur
    return prev[-1]


def reference_from_lyrics(acestep_lyrics: str) -> str:
    """Strip ACE-Step structure tags (e.g. [Chorus]) to get the plain sung text that
    the transcript should be compared against."""
    lines = [ln for ln in acestep_lyrics.splitlines()
             if ln.strip() and not re.fullmatch(r"\[.*\]", ln.strip())]
    return " ".join(lines)
