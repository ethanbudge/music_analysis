"""
melody.py — Generate a melody per music-emotion from its valence/arousal.

The user chose to vary the melody per emotion, so each music-emotion gets its own tune
derived deterministically from its circumplex coordinates and key:

  mode / tonic   from emotions.Emotion.keyscale  ("E major" → E major scale)
  tempo          from emotions.Emotion.bpm
  register       centre pitch rises with arousal
  rhythm         high arousal → busier, even short notes; low arousal → sustained,
                 with a lengthened phrase-final note
  contour        seeded scale-degree walk, biased upward for high valence / downward
                 for low, wider leaps for high arousal; each line (measure) resolves
                 to a stable scale degree.

One note per syllable (v1). Notes carry pitch + rhythm; lyrics are attached later in
score.py. Fully deterministic (seed per emotion) → reproducible stimuli.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from .config import MELODY
from lmcgen import emotions as emo

_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]           # natural minor
_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass
class Note:
    midi: int                 # MIDI pitch
    dur16: int                # duration in sixteenth notes
    measure: int              # 0-based measure index
    lyric: str = ""           # filled in by score.assemble
    word: str = ""
    syllabic: str = "single"  # single | begin | middle | end (MusicXML)


def parse_keyscale(keyscale: str) -> tuple[int, bool]:
    """'E major' / 'D minor' → (tonic pitch-class, is_major)."""
    m = re.match(r"\s*([A-Ga-g])(#|b)?\s*(major|minor)?", keyscale or "C major")
    pc = _NOTE_PC[m.group(1).upper()] if m else 0
    if m and m.group(2) == "#":
        pc = (pc + 1) % 12
    elif m and m.group(2) == "b":
        pc = (pc - 1) % 12
    is_major = not (m and m.group(3) and m.group(3).lower() == "minor")
    return pc, is_major


def _scale_midi(tonic_pc: int, is_major: bool, centre_midi: int, span: int = 19) -> list[int]:
    """Scale pitches (MIDI) within ±span semitones of centre, ascending."""
    ints = _MAJOR if is_major else _MINOR
    lo, hi = centre_midi - span, centre_midi + span
    return sorted(m for m in range(lo, hi + 1) if (m - tonic_pc) % 12 in ints)


def _rhythm(n: int, arousal: float, phrase_final: bool) -> list[int]:
    """Split a 16-sixteenth measure into n note durations. High arousal → even/busy;
    low arousal → sustained with a long final note."""
    if n <= 0:
        return []
    base = 16 // n
    durs = [max(1, base)] * n
    slack = 16 - sum(durs)
    # distribute leftover; low arousal piles it onto the final (sustain), high arousal spreads it
    i = n - 1
    while slack > 0:
        if arousal < 0.5 or i == n - 1:
            durs[n - 1] += 1
        else:
            durs[i] += 1
            i = (i - 1) if i > 0 else n - 1
        slack -= 1
    # if we overshot (n>16), clamp to 1 and drop the overflow by shortening the tail
    while sum(durs) > 16:
        for j in range(n):
            if durs[j] > 1:
                durs[j] -= 1
                if sum(durs) == 16:
                    break
    return durs


def build_melody(emotion_name: str, syllables_per_line: list[int]) -> list[Note]:
    """Melody for `emotion_name`: one measure per line, one note per syllable."""
    e = emo.get(emotion_name)
    tonic_pc, is_major = parse_keyscale(e.keyscale)
    idx = emo.ORDER.index(emotion_name)
    import random
    rng = random.Random(MELODY["seed"] * 101 + idx)

    # Register is tied to the tonic (in the base octave) plus an arousal-driven octave
    # shift, so higher-arousal emotions sit clearly higher regardless of key.
    tonic_base = 12 * (MELODY["base_octave"] + 1) + tonic_pc
    centre = tonic_base + round(MELODY["register_arousal"] * e.arousal)
    scale = _scale_midi(tonic_pc, is_major, centre)
    tonic_choices = [m for m in scale if m % 12 == tonic_pc]
    cur = min(tonic_choices, key=lambda m: abs(m - centre)) if tonic_choices else scale[len(scale) // 2]
    pos = scale.index(cur)

    start_pos = pos
    up_bias = 0.5 + 0.15 * (e.valence - 0.5) * 2      # mild upward tilt for +valence
    max_step = 1 + round(2 * e.arousal)              # wider leaps when aroused
    band = 5                                          # keep within ±5 scale degrees of start
    lo, hi = max(0, start_pos - band), min(len(scale) - 1, start_pos + band)

    notes: list[Note] = []
    for line_i, n_syl in enumerate(syllables_per_line):
        durs = _rhythm(n_syl, e.arousal, phrase_final=(line_i == len(syllables_per_line) - 1))
        for k in range(n_syl):
            last_of_phrase = (line_i == len(syllables_per_line) - 1) and (k == n_syl - 1)
            last_of_line = (k == n_syl - 1)
            if last_of_phrase:
                pos = scale.index(min(tonic_choices, key=lambda m: abs(m - scale[pos]))) if tonic_choices else pos
            elif last_of_line:
                stable = [m for m in scale if (m - tonic_pc) % 12 in (0, _MAJOR[2] if is_major else _MINOR[2], 7)]
                pos = scale.index(min(stable, key=lambda m: abs(m - scale[pos]))) if stable else pos
            else:
                # mean-reverting biased walk so the melody stays near its (arousal-set)
                # register instead of drifting to the extremes.
                p_up = up_bias + 0.15 * (start_pos - pos) / band     # pull back toward centre
                step = rng.randint(1, max_step) * (1 if rng.random() < p_up else -1)
                pos = max(lo, min(hi, pos + step))
            notes.append(Note(midi=scale[pos], dur16=durs[k], measure=line_i))
    return notes
