"""
score.py — Assemble a grid cell (hook_L sung to melody_M) into a Score.

A Score is the engine-agnostic intermediate representation: a list of notes, each with
pitch, duration, and the syllable/word/lyric-role it carries, plus tempo + key. It is
exported to MusicXML (musicxml.py) for the SVS app.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from lmcgen import emotions as emo, lyrics as lyr
from . import syllables as syl
from . import melody as mel


@dataclass
class Score:
    lyric_emotion: str
    music_emotion: str
    notes: list[mel.Note]
    tempo_bpm: int
    keyscale: str
    tonic_pc: int
    is_major: bool
    title: str = ""

    @property
    def congruent(self) -> bool:
        return self.lyric_emotion == self.music_emotion


def _syllabic_roles(sylls: list[syl.Syllable]) -> list[str]:
    """MusicXML syllabic role per syllable (single/begin/middle/end) from word grouping."""
    roles = []
    for i, s in enumerate(sylls):
        prev_same = i > 0 and sylls[i - 1].word_index == s.word_index
        next_same = i + 1 < len(sylls) and sylls[i + 1].word_index == s.word_index
        roles.append("middle" if prev_same and next_same else
                     "begin" if next_same else
                     "end" if prev_same else "single")
    return roles


def assemble(lyric_emotion: str, music_emotion: str) -> Score:
    hook = lyr.get(lyric_emotion)
    lines = hook.lines
    per_line = [syl.syllabify(ln) for ln in lines]          # syllables grouped by line
    counts = [len(s) for s in per_line]

    notes = mel.build_melody(music_emotion, counts)
    flat = [s for line in per_line for s in line]
    assert len(flat) == len(notes), f"syllable/note mismatch {len(flat)} vs {len(notes)}"
    roles = _syllabic_roles(flat)
    for note, s, role in zip(notes, flat, roles):
        note.lyric, note.word, note.syllabic = s.text, s.word, role

    e = emo.get(music_emotion)
    tonic_pc, is_major = mel.parse_keyscale(e.keyscale)
    return Score(lyric_emotion=lyric_emotion, music_emotion=music_emotion, notes=notes,
                 tempo_bpm=e.bpm, keyscale=e.keyscale, tonic_pc=tonic_pc, is_major=is_major,
                 title=f"{lyric_emotion} lyric / {music_emotion} music")
