"""
musicxml.py — Write a Score to MusicXML (notes + per-note lyrics + key + tempo).

MusicXML is the cleanest interchange for "notes WITH lyrics attached": Synthesizer V
Studio 2 Pro and ACE Studio both import it and place each syllable on its note, so the
performance is fully specified before the app phonemises it with the fixed voice.

Hand-rolled (stdlib only) so it stays dependency-free and testable. divisions=4, so a
note's sixteenth-count maps straight to <duration>.
"""
from __future__ import annotations
from pathlib import Path
from xml.sax.saxutils import escape

from .score import Score

_DIVISIONS = 4                       # per quarter note → 1 sixteenth = 1 division

# pitch-class → (step, alter) using sharps
_PC = {0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D", 1), 4: ("E", 0), 5: ("F", 0),
       6: ("F", 1), 7: ("G", 0), 8: ("G", 1), 9: ("A", 0), 10: ("A", 1), 11: ("B", 0)}

# major-key signature (circle-of-fifths count) by tonic pitch-class
_MAJOR_FIFTHS = {0: 0, 1: 7, 2: 2, 3: -3, 4: 4, 5: -1, 6: 6, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}

# sixteenth-count → (note-type, dots)
_TYPE = {1: ("16th", 0), 2: ("eighth", 0), 3: ("eighth", 1), 4: ("quarter", 0),
         6: ("quarter", 1), 8: ("half", 0), 12: ("half", 1), 16: ("whole", 0)}


def _fifths(tonic_pc: int, is_major: bool) -> int:
    return _MAJOR_FIFTHS[tonic_pc] if is_major else _MAJOR_FIFTHS[(tonic_pc + 3) % 12]


def _pitch_xml(midi: int) -> str:
    step, alter = _PC[midi % 12]
    octave = midi // 12 - 1
    alter_xml = f"<alter>{alter}</alter>" if alter else ""
    return f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"


def _type_xml(dur16: int) -> str:
    t, dots = _TYPE.get(dur16, ("quarter", 0))
    return f"<type>{t}</type>" + "<dot/>" * dots


def _note_xml(note) -> str:
    lyric = ""
    if note.lyric:
        lyric = (f"<lyric number=\"1\"><syllabic>{note.syllabic}</syllabic>"
                 f"<text>{escape(note.lyric)}</text></lyric>")
    return ("<note>" + _pitch_xml(note.midi) +
            f"<duration>{note.dur16}</duration><voice>1</voice>" +
            _type_xml(note.dur16) + lyric + "</note>")


def to_musicxml(score: Score) -> str:
    fifths = _fifths(score.tonic_pc, score.is_major)
    mode = "major" if score.is_major else "minor"
    # group notes by measure
    measures: dict[int, list] = {}
    for n in score.notes:
        measures.setdefault(n.measure, []).append(n)

    body = []
    for mi in sorted(measures):
        attrs = ""
        if mi == 0:
            attrs = (f"<attributes><divisions>{_DIVISIONS}</divisions>"
                     f"<key><fifths>{fifths}</fifths><mode>{mode}</mode></key>"
                     f"<time><beats>4</beats><beat-type>4</beat-type></time>"
                     f"<clef><sign>G</sign><line>2</line></clef></attributes>"
                     f"<direction placement=\"above\"><direction-type><metronome>"
                     f"<beat-unit>quarter</beat-unit><per-minute>{score.tempo_bpm}</per-minute>"
                     f"</metronome></direction-type><sound tempo=\"{score.tempo_bpm}\"/></direction>")
        notes_xml = "".join(_note_xml(n) for n in measures[mi])
        body.append(f"<measure number=\"{mi + 1}\">{attrs}{notes_xml}</measure>")

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="3.1">'
        f'<work><work-title>{escape(score.title)}</work-title></work>'
        '<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>'
        '<part id="P1">' + "".join(body) + '</part>'
        '</score-partwise>'
    )


def write(score: Score, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_musicxml(score), encoding="utf-8")
    return path
