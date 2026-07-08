"""
pipeline.py — Build the grid of scores and export them for the SVS app.

For every (lyric emotion L, music emotion M) cell, assemble hook_L sung to melody_M and
export a MusicXML file named <L>__<M>.musicxml under data/svs/scores/. You then import
these into Synthesizer V Studio 2 Pro (or ACE Studio), assign ONE fixed voice, and
batch-render to data/svs/audio/<L>__<M>.wav — which the existing lmcgen WER / VA harness
can validate directly.
"""
from __future__ import annotations
import logging

from . import config as C
from . import score as score_mod
from . import musicxml

logger = logging.getLogger(__name__)


def cell_name(lyric_emotion: str, music_emotion: str) -> str:
    return f"{lyric_emotion}__{music_emotion}"


def build_scores(emotions: list[str] | None = None) -> list:
    """Assemble every grid cell → Score objects (no I/O)."""
    emotions = emotions or C.EMOTIONS
    scores = []
    for L in emotions:
        for M in emotions:
            scores.append(score_mod.assemble(L, M))
    return scores


def export_scores(emotions: list[str] | None = None) -> dict:
    """Assemble + write MusicXML for every cell. Returns {cell_name: path}."""
    C.ensure_dirs()
    emotions = emotions or C.EMOTIONS
    out = {}
    scores = build_scores(emotions)
    for s in scores:
        name = cell_name(s.lyric_emotion, s.music_emotion)
        path = C.SCORE_DIR / f"{name}.musicxml"
        musicxml.write(s, path)
        out[name] = str(path)
        if C.EXPORT["also_midi"]:
            _write_midi(s, C.SCORE_DIR / f"{name}.mid")
    logger.info("Exported %d MusicXML scores → %s", len(out), C.SCORE_DIR)
    logger.info("Next: import into %s, assign one fixed voice, batch-render to %s",
                C.VOICE["engine"], C.AUDIO_DIR)
    return out


def _write_midi(s, path) -> None:
    """Optional .mid (notes + lyric events) via `mido`, for engines that prefer MIDI."""
    try:
        import mido
    except ImportError:
        logger.warning("  mido not installed; skipping MIDI (pip install mido)")
        return
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(s.tempo_bpm)))
    for n in s.notes:
        ticks = int(n.dur16 / 4 * tpb)               # sixteenths → quarter-beats → ticks
        if n.lyric:
            tr.append(mido.MetaMessage("lyrics", text=n.lyric, time=0))
        tr.append(mido.Message("note_on", note=n.midi, velocity=80, time=0))
        tr.append(mido.Message("note_off", note=n.midi, velocity=0, time=ticks))
    mid.save(str(path))
