"""
syllables.py — Split a hook into per-note syllables.

SVS engines sing one syllable per note, so we need each hook segmented into syllables
(and which word each belongs to, for the app's phonemiser). Uses `pyphen`
(Hunspell hyphenation dictionaries — deterministic, no ML, no network) when available;
otherwise a transparent vowel-group fallback so the pipeline always runs.

    pip install pyphen        # optional but recommended for accurate syllabification
"""
from __future__ import annotations
import re
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z']+")
_pyphen = None
_pyphen_tried = False


@dataclass(frozen=True)
class Syllable:
    text: str          # the syllable as sung
    word: str          # the word it belongs to (for the SVS phonemiser)
    word_index: int    # index of the word within the hook
    first_in_word: bool


def _get_pyphen():
    global _pyphen, _pyphen_tried
    if not _pyphen_tried:
        _pyphen_tried = True
        try:
            import pyphen
            _pyphen = pyphen.Pyphen(lang="en_US")
        except Exception:                                          # noqa: BLE001
            _pyphen = None
    return _pyphen


def _split_word(word: str) -> list[str]:
    """Syllabify one word. pyphen if present, else a vowel-group heuristic."""
    p = _get_pyphen()
    if p is not None:
        parts = p.inserted(word).split("-")
        return [s for s in parts if s] or [word]
    return _fallback_syllables(word)


def _fallback_syllables(word: str) -> list[str]:
    """Deterministic vowel-group syllabifier: cut after each vowel-group's following
    consonants, roughly one syllable per vowel group. Good enough to place notes; the
    SVS app re-phonemises anyway."""
    w = word.lower()
    # vowel groups → candidate syllable count
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if n == 0:
        return [word]
    # English silent-e / -es / -ed corrections (classic heuristic):
    if n > 1 and re.search(r"[^aeiouy]e$", w):                 # silent final 'e' (not 'le')
        if not w.endswith("le"):
            n -= 1
    if n > 1 and re.search(r"[^aeiouytd]ed$", w):             # '-ed' is silent except after t/d
        n -= 1
    n = max(1, n)
    if n == 1:
        return [word]
    # Segment: cut the word into n chunks, each ending after a vowel group's trailing
    # consonants (keeps chunks pronounceable for the SVS phonemiser).
    bounds = [m.end() for m in re.finditer(r"[aeiouy]+[^aeiouy]*", w)][:n]
    out, prev = [], 0
    for b in bounds[:n - 1]:
        out.append(word[prev:b]); prev = b
    out.append(word[prev:])
    return [s for s in out if s] or [word]


def syllabify(text: str) -> list[Syllable]:
    """Segment a hook (possibly multi-line) into an ordered list of Syllables."""
    words = _WORD.findall(text)
    out: list[Syllable] = []
    for wi, word in enumerate(words):
        parts = _split_word(word)
        for j, s in enumerate(parts):
            out.append(Syllable(text=s, word=word, word_index=wi, first_in_word=(j == 0)))
    return out


def count(text: str) -> int:
    return len(syllabify(text))
