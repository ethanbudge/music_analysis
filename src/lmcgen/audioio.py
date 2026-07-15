"""
audioio.py — Generation spec + audio I/O helpers shared by the generative arm.

Kept deliberately dependency-light: importing this module pulls in only numpy.
librosa / soundfile are imported lazily inside the functions that need them, so the
package imports cheaply and the dry-run path works even without an audio backend.

  GenSpec              one song to generate (the Lyria prompt + where it lands +
                       its target valence/arousal, used for logging and the mock).
  _mock_waveform       a cheap, VA-dependent placeholder clip for dry-run plumbing
                       (major/minor + brightness from valence, tempo/noise from
                       arousal). Not meant to *sound* like the target — only to be a
                       valid file that varies systematically so nothing downstream is
                       degenerate.
  _write_wav           write a mono waveform to WAV (soundfile, stdlib wave fallback).
  _transcode_to_wav    decode whatever the API returned (mp3) and re-save as canonical
                       WAV so every clip on disk — real or mock — is byte-identical in
                       format for the MuLan / librosa / Whisper validators.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GenSpec:
    """One song to generate."""
    prompt: str            # the full Lyria `input` (voice descriptor + style + lyrics)
    sung_text: str         # plain sung lyric text (WER reference) — no [Chorus] tag
    out_path: Path
    music_quadrant: str    # target music quadrant code (e.g. "hvha")
    valence: float         # target valence in [0,1] — logging + dry-run synth
    arousal: float         # target arousal in [0,1]
    bpm: int = 0           # representative tempo (mock realism / prompt text)
    keyscale: str = ""     # representative key/mode, e.g. "E major" (mock realism)
    lyric_id: str = ""     # which of the 16 lyrics
    rep: int = 0           # repetition index within the cell


# ─── canonical WAV writer ─────────────────────────────────────────────────────────
def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(path), wav, sr)
    except Exception:                                              # noqa: BLE001
        # Fallback: 16-bit PCM WAV via the stdlib (no soundfile dependency).
        import wave
        pcm = np.clip(wav, -1, 1)
        pcm = (pcm * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())


def _transcode_to_wav(src: Path, dst: Path, sr: int) -> None:
    """Decode whatever the generator returned (likely mp3) and re-save as canonical
    mono WAV at `sr`, so real and mock clips share one format downstream."""
    import librosa
    wav, _ = librosa.load(str(src), sr=sr, mono=True)
    _write_wav(dst, wav.astype(np.float32), sr)


# ─── VA-dependent mock waveform (dry-run only) ───────────────────────────────────
_NOTE_HZ = {"C": 261.63, "D": 293.66, "E": 329.63, "F": 349.23,
            "G": 392.00, "A": 440.00, "B": 493.88}


def _mock_waveform(spec: GenSpec, sr: int, duration: float, seed: int = 0) -> np.ndarray:
    """Cheap placeholder: valence -> major/minor + brightness, arousal -> tempo + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(float(duration) * sr)) / sr

    root = _NOTE_HZ.get((spec.keyscale[:1] or "A").upper(), 220.0) / 2.0   # down an octave
    major = ("major" in spec.keyscale.lower()) or (spec.valence >= 0.5 and not spec.keyscale)
    third = root * (2 ** (4 / 12)) if major else root * (2 ** (3 / 12))     # major/minor 3rd
    fifth = root * (2 ** (7 / 12))
    valence, arousal = float(spec.valence), float(spec.arousal)

    n_harm = 1 + int(round(3 * valence))          # brighter with higher valence
    chord = np.zeros_like(t)
    for f0 in (root, third, fifth):
        for h in range(1, n_harm + 1):
            chord += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    chord /= np.max(np.abs(chord) + 1e-9)

    bpm = float(spec.bpm) if spec.bpm else (60 + 120 * arousal)   # tempo rises with arousal
    beat_hz = bpm / 60.0
    env = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * beat_hz * t), 0, 1) ** (1 + 2 * arousal)

    noise = arousal * 0.15 * rng.standard_normal(t.shape)
    wav = 0.8 * env * chord + noise
    fade = int(0.05 * sr)                          # gentle fade to avoid clicks
    if fade and wav.size > 2 * fade:
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
    return (wav / (np.max(np.abs(wav)) + 1e-9) * 0.9).astype(np.float32)
