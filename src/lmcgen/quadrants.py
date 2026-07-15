"""
quadrants.py — The four extreme valence/arousal (VA) corners of the circumplex.

Each corner operationalises one music *target* for generation and validation:

  valence / arousal   the extreme circumplex coordinates in [0,1] (Russell 1980) — the
                      interpretable, model-independent target for the acoustic-VA check.
  style_words         the natural-language music descriptor spliced into the Lyria
                      prompt (Lyria has no tempo/key/mood fields — only prompt text).
  anchor_prompts      short *music* descriptions embedded with the MuLan text tower and
                      averaged into that corner's ANCHOR. Because MuLan is a joint
                      audio/text space, the same anchor scores generated AUDIO — this is
                      the embedding "target" the songs are validated against (Lyria can't
                      take an embedding as input, so we select/score against it instead).
  lexicon             affect words for the model-independent lyric lexical check (lyrics.py).
  bpm / keyscale      a representative tempo + mode (prompt text + dry-run mock realism).
"""
from __future__ import annotations
from dataclasses import dataclass

from .config import QUADRANTS


@dataclass(frozen=True)
class Quadrant:
    code: str                   # one of config.QUADRANTS
    label: str                  # human-readable name
    valence: float              # 0 (very negative) .. 1 (very positive) — extreme
    arousal: float              # 0 (calm) .. 1 (highly aroused) — extreme
    style_words: str            # music descriptor for the Lyria prompt
    anchor_prompts: list[str]   # MuLan text-anchor music descriptions
    lexicon: list[str]          # affect words for the lyric lexical check
    bpm: int
    keyscale: str

    @property
    def target(self) -> tuple[float, float]:
        return (self.valence, self.arousal)


# ─── The four extreme corners ─────────────────────────────────────────────────────
_QUADRANTS: dict[str, Quadrant] = {

    "hvha": Quadrant(
        code="hvha", label="high valence / high arousal (joyful, euphoric)",
        valence=0.92, arousal=0.88,
        style_words=("bright major key, fast upbeat tempo around 150 BPM, euphoric and "
                     "celebratory, driving drums and shimmering guitars, big joyful "
                     "energetic anthem, radiant and exhilarating"),
        anchor_prompts=[
            "an upbeat, euphoric, high-energy song bursting with joy",
            "exciting joyful uptempo music, fast bright and celebratory",
            "an exhilarating happy anthem, soaring and triumphant",
            "radiant dance-pop energy, elated and full of light",
        ],
        lexicon=["alive", "gold", "bright", "shine", "celebrate", "thrill", "soaring",
                 "dancing", "fire", "high", "wild", "burning", "light", "electric",
                 "rush", "sky", "joy", "glow", "run", "fly"],
        bpm=150, keyscale="E major",
    ),

    "hvla": Quadrant(
        code="hvla", label="high valence / low arousal (calm, content, serene)",
        valence=0.90, arousal=0.12,
        style_words=("warm major key, slow gentle tempo around 68 BPM, calm peaceful and "
                     "content, soft acoustic guitar and mellow piano, tender and soothing, "
                     "intimate and relaxed"),
        anchor_prompts=[
            "a calm, peaceful, gentle and soothing song",
            "warm mellow acoustic music, soft and relaxing",
            "a serene, tender, contented slow ballad",
            "quiet major-key music, gentle and reassuring",
        ],
        lexicon=["calm", "peace", "warm", "gentle", "home", "still", "soft", "easy",
                 "slow", "rest", "hold", "close", "quiet", "breeze", "glow", "sunlight",
                 "safe", "morning", "smile", "breathe"],
        bpm=68, keyscale="C major",
    ),

    "lvha": Quadrant(
        code="lvha", label="low valence / high arousal (angry, afraid, tense)",
        valence=0.10, arousal=0.90,
        style_words=("dark minor key, fast aggressive tempo around 160 BPM, tense and "
                     "furious, distorted downtuned guitars and pounding drums, urgent "
                     "frightening and menacing, explosive and hostile"),
        anchor_prompts=[
            "an aggressive, angry, high-energy song, pounding and hostile",
            "tense frightening urgent music, dark and menacing",
            "a furious heavy track, explosive and violent",
            "panicked, fearful, fast and dark music",
        ],
        lexicon=["rage", "burn", "fire", "fight", "run", "fear", "scream", "tear",
                 "danger", "dark", "storm", "panic", "break", "blood", "war", "teeth",
                 "chase", "claw", "threat", "shadow"],
        bpm=160, keyscale="D minor",
    ),

    "lvla": Quadrant(
        code="lvla", label="low valence / low arousal (sad, weary, hopeless)",
        valence=0.10, arousal=0.12,
        style_words=("slow minor key, sparse tempo around 60 BPM, sad heavy and mournful, "
                     "quiet piano and aching strings, desolate weary and hollow, "
                     "melancholy and heartbroken"),
        anchor_prompts=[
            "a sad, slow, mournful song of loss",
            "melancholy sorrowful quiet music, aching and desolate",
            "a desolate heartbroken slow ballad",
            "lonely, weary, low-energy minor-key music",
        ],
        lexicon=["grief", "sorrow", "alone", "lonely", "tears", "empty", "gone", "ghost",
                 "fade", "cold", "quiet", "rain", "ash", "hollow", "weary", "drown",
                 "sink", "lost", "still", "ache"],
        bpm=60, keyscale="A minor",
    ),
}

# Canonical ordered list (matches config.QUADRANTS order).
ORDER: list[str] = list(QUADRANTS)
ALL: list[Quadrant] = [_QUADRANTS[c] for c in ORDER]


def get(code: str) -> Quadrant:
    return _QUADRANTS[code]


def anchor_prompts() -> dict[str, list[str]]:
    return {q.code: q.anchor_prompts for q in ALL}


def valence_arousal() -> dict[str, tuple[float, float]]:
    return {q.code: q.target for q in ALL}


def nearest_quadrant(valence: float, arousal: float) -> str:
    """Which corner a measured (valence, arousal) point is closest to (Euclidean).
    Used as a model-independent manipulation check on the acoustic VA of a clip."""
    return min(ORDER, key=lambda c: (_QUADRANTS[c].valence - valence) ** 2
               + (_QUADRANTS[c].arousal - arousal) ** 2)


def _validate_definitions() -> None:
    assert set(_QUADRANTS) == set(QUADRANTS), "quadrant set mismatch vs config.QUADRANTS"
    for q in ALL:
        assert 0.0 <= q.valence <= 1.0 and 0.0 <= q.arousal <= 1.0, f"{q.code}: VA out of range"
        assert len(q.anchor_prompts) >= 3, f"{q.code}: needs >=3 anchor prompts"
        assert len(q.lexicon) >= 8, f"{q.code}: lexicon too small"
        assert q.style_words, f"{q.code}: needs style_words"


_validate_definitions()
