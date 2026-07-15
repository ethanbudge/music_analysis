"""
units.py — turn each Track into scored UNITS at three segmentation levels.

A Unit bundles the audio slice, the raw lyric text, and position metadata for one
scorable item:

  song      the whole song  vs  the whole lyrics
  segment   chorus / non-chorus (labelled 'chorus' / 'verse')  — reuses chorus flags
  line      each synced line  vs  its own audio window

Audio is loaded once per track at BASE_SR and sliced by timestamp with the main
pipeline's lmc.embeddings._slice, so timing matches the LRC exactly.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np

from . import config
from lmc.embeddings import _slice
from lmc.utils import lrc_to_plaintext

logger = logging.getLogger(__name__)


@dataclass
class Unit:
    level: str            # 'song' | 'segment' | 'line'
    slug: str
    artist: str
    audio: np.ndarray     # waveform at config.BASE_SR
    text: str             # raw lyric text of this unit
    position: dict = field(default_factory=dict)   # extra CSV columns (level-specific)


def _load_audio(path: str) -> np.ndarray | None:
    import librosa
    try:
        wav, _ = librosa.load(path, sr=config.BASE_SR, mono=True)
        return wav
    except Exception as e:                                     # noqa: BLE001
        logger.warning("  audio load failed (%s): %s", path, e)
        return None


def build_units(tracks, level: str) -> list[Unit]:
    """Build all Units at the given level across the usable tracks."""
    assert level in ("song", "segment", "line")
    sr = config.BASE_SR
    units: list[Unit] = []

    for tk in tracks:
        if not tk.ok:
            continue
        wav = _load_audio(tk.audio_path)
        if wav is None or len(wav) < int(0.2 * sr):
            continue
        dur = len(wav) / sr
        lines = tk.lines

        if level == "song":
            units.append(Unit("song", tk.slug, tk.artist, wav,
                              lrc_to_plaintext(tk.synced_lyrics)))

        elif level == "segment":
            flags = tk.chorus_flags or [False] * len(lines)
            for label, want in (("chorus", True), ("verse", False)):
                idxs = [i for i, f in enumerate(flags) if bool(f) == want and i < len(lines)]
                if not idxs:
                    continue
                seg_wav = np.concatenate(
                    [_slice(wav, sr, lines[i]["start"], lines[i]["end"], dur) for i in idxs])
                text = " ".join(lines[i]["text"] for i in idxs)
                mids = [(lines[i]["start"] + (lines[i]["end"] or lines[i]["start"])) / 2 for i in idxs]
                units.append(Unit("segment", tk.slug, tk.artist, seg_wav, text,
                                  {"segment_label": label,
                                   "position_pct": round(100 * float(np.mean(mids)) / dur, 2),
                                   "n_lines": len(idxs)}))

        else:  # line
            pad = config.LINE_WINDOW_PAD
            for i, ln in enumerate(lines):
                seg = _slice(wav, sr, ln["start"], ln["end"], dur, pad)
                if len(seg) < int(0.1 * sr):
                    continue
                mid = (ln["start"] + (ln["end"] or ln["start"])) / 2
                units.append(Unit("line", tk.slug, tk.artist, seg, ln["text"],
                                  {"line_index": i,
                                   "position_pct": round(100 * mid / dur, 2),
                                   "is_chorus": int(bool(tk.chorus_flags[i]))
                                                if i < len(tk.chorus_flags) else 0,
                                   "line_text": ln["text"]}))

    logger.info("Built %d %s-level units from %d tracks.",
                len(units), level, sum(t.ok for t in tracks))
    return units
