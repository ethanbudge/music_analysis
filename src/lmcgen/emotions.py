"""
emotions.py — The eight target emotions and everything that operationalises them.

DESIGN (v2, single-genre): to remove the emotion<->genre confound and equalise
lyric intelligibility, ALL eight emotions are rendered in ONE genre — 90s
alternative rock — with a FIXED voice. Emotion is varied *within* the genre via
tempo, mode (major/minor), dynamics and intensity, not by switching styles. Every
ACE-Step caption is therefore built as:

    VOICE_BLURB  +  GENRE_BASE  +  <emotion_style>

so the singer and the band stay constant and only the emotional colour changes.

Each emotion still carries:
  valence / arousal   circumplex coordinates (Russell 1980; Scherer 2005) — the
                      interpretable, model-independent fallback.
  anchor_prompts      short descriptors embedded with the MuLan *text* tower and
                      averaged into that emotion's ANCHOR (scores lyrics AND audio).
  lexicon             Plutchik / NRC EmoLex words for the model-independent lexical
                      check (see lyrics.py).
  emotion_style       the *within-rock* caption modifiers (tempo/mode/dynamics).
  bpm / keyscale      the concrete tempo + mode levers for this emotion.
"""
from __future__ import annotations
from dataclasses import dataclass

from .config import EMOTIONS

# ─── Fixed controls shared by every generation ───────────────────────────────────
# One singer, described the same way everywhere, so the voice is held ~constant
# across all 64 clips. (ACE-Step can't guarantee an identical voice from text alone —
# voice-consistency is also screened downstream — but a fixed, detailed descriptor
# plus a single genre gets it close.) Keep this SHORT; ACE-Step caps captions at 512
# chars and we prepend it to every emotion caption.
VOICE_BLURB = ("Lead vocal: one female alto, warm and natural, clear enunciated diction, "
               "light vibrato, no pitch-correction or effects, dry and up-front, no backing vocals")

# The single genre. Vocal-forward and rhythmic so lyrics lock to the beat; spans
# aggressive -> ballad so it can carry the full emotional range.
GENRE_BASE = ("90s alternative rock, live band, electric guitars, real drums and bass, "
              "clear intelligible lead vocal front and center, minimal reverb")


@dataclass(frozen=True)
class Emotion:
    name: str
    plutchik_dyad: str          # the underlying basic emotion (Plutchik primary)
    valence: float              # 0 (very negative) .. 1 (very positive)
    arousal: float              # 0 (calm) .. 1 (highly aroused)
    anchor_prompts: list[str]
    lexicon: list[str]
    emotion_style: str          # within-rock emotional colour (tempo/mode/dynamics/intensity)
    bpm: int
    keyscale: str               # ACE-Step keyscale, e.g. "A minor"

    @property
    def caption(self) -> str:
        """Full ACE-Step caption: fixed voice + fixed genre + this emotion's colour."""
        return f"{VOICE_BLURB}. {GENRE_BASE}, {self.emotion_style}."

    @property
    def captions(self) -> list[str]:
        """Compat with the (default-off) tuning search: a single candidate."""
        return [self.caption]


# ─── The eight emotions (all within 90s alternative rock) ────────────────────────
_EMOTIONS: dict[str, Emotion] = {

    "ecstasy": Emotion(
        name="ecstasy", plutchik_dyad="joy", valence=0.92, arousal=0.85,
        anchor_prompts=[
            "an ecstatic, euphoric song bursting with joy",
            "uplifting jubilant music, radiant and celebratory",
            "exhilarating happy anthem, soaring and triumphant",
            "blissful, elated, full of light and energy",
        ],
        lexicon=["joy", "ecstasy", "euphoria", "bliss", "elated", "radiant", "glow",
                 "celebrate", "alive", "soaring", "delight", "jubilant", "thrill",
                 "shine", "gold", "burning"],
        emotion_style="bright major key, fast and driving, soaring anthemic uplifting "
                      "chorus, jubilant and euphoric, big triumphant energy",
        bpm=150, keyscale="E major",
    ),

    "admiration": Emotion(
        name="admiration", plutchik_dyad="trust", valence=0.78, arousal=0.42,
        anchor_prompts=[
            "a warm, admiring song full of trust and devotion",
            "reverent, heartfelt music expressing loyalty and respect",
            "tender heartfelt ballad of faith and gratitude",
            "sincere, uplifting, devoted and steadfast",
        ],
        lexicon=["trust", "admire", "devotion", "loyal", "faithful", "honest",
                 "steady", "reverent", "grateful", "sincere", "respect", "believe",
                 "true", "noble", "cherish", "hold"],
        emotion_style="warm major key, mid-tempo, heartfelt and sincere, earnest steady "
                      "delivery, devoted and reverent, gentle swell",
        bpm=92, keyscale="G major",
    ),

    "terror": Emotion(
        name="terror", plutchik_dyad="fear", valence=0.10, arousal=0.90,
        anchor_prompts=[
            "a terrifying song full of dread and panic",
            "frightening, tense and menacing music",
            "a fearful track racing with danger and alarm",
            "nightmarish, chilling, breathless and afraid",
        ],
        lexicon=["fear", "terror", "dread", "panic", "horror", "afraid", "shudder",
                 "tremble", "nightmare", "shadow", "scream", "run", "danger",
                 "flee", "chilling", "dark"],
        emotion_style="dark minor key, fast and urgent, tense dissonant guitars, "
                      "driving panicked energy, ominous and frightening",
        bpm=156, keyscale="D minor",
    ),

    "amazement": Emotion(
        name="amazement", plutchik_dyad="surprise", valence=0.62, arousal=0.85,
        anchor_prompts=[
            "an astonishing, awe-struck song full of wonder",
            "breathtaking music, sudden and dazzling",
            "a track bursting with surprise and amazement",
            "wide-eyed, spectacular, swelling with awe",
        ],
        lexicon=["surprise", "amazement", "wonder", "awe", "astonish", "sudden",
                 "unbelievable", "dazzling", "spectacular", "gasp", "marvel",
                 "breathtaking", "shock", "stunned", "glow", "world"],
        emotion_style="bright major key, dynamic with sudden dramatic swells, wide-eyed "
                      "and wondrous, building to an awe-struck soaring chorus",
        bpm=120, keyscale="A major",
    ),

    "grief": Emotion(
        name="grief", plutchik_dyad="sadness", valence=0.10, arousal=0.20,
        anchor_prompts=[
            "a grieving, sorrowful song of loss and mourning",
            "a slow, tearful ballad full of heartbreak",
            "melancholy music, aching and desolate",
            "a mournful lament, lonely and heavy with sorrow",
        ],
        lexicon=["grief", "sorrow", "mourning", "loss", "tears", "weep", "lonely",
                 "heartbreak", "ache", "silence", "empty", "gone", "ghost", "fade",
                 "quiet", "rain"],
        emotion_style="slow minor key, mournful rock ballad, sparse aching verses and a "
                      "heavy swelling chorus, heartbroken and desolate",
        bpm=68, keyscale="C minor",
    ),

    "loathing": Emotion(
        name="loathing", plutchik_dyad="disgust", valence=0.15, arousal=0.55,
        anchor_prompts=[
            "a song dripping with disgust and contempt",
            "a sneering, revolted track full of loathing",
            "bitter, repulsed, dark and contemptuous music",
            "a venomous, disgusted, seething sound",
        ],
        lexicon=["disgust", "loathing", "revulsion", "contempt", "sick", "vile",
                 "rot", "poison", "venom", "sneer", "repulsed", "filthy", "scorn",
                 "shame", "wretched", "done"],
        emotion_style="mid-tempo minor key, gritty sneering attitude, sludgy distorted "
                      "guitars, bitter and contemptuous, seething and disgusted",
        bpm=100, keyscale="F minor",
    ),

    "rage": Emotion(
        name="rage", plutchik_dyad="anger", valence=0.12, arousal=0.95,
        anchor_prompts=[
            "a furious, raging song full of anger and fury",
            "aggressive, violent music, pounding and hostile",
            "an enraged track of shouting fury and defiance",
            "wrathful, explosive, seething with rage",
        ],
        lexicon=["anger", "rage", "fury", "wrath", "furious", "hostile", "burn",
                 "fire", "roar", "explode", "scream", "violent", "tear", "line",
                 "defiant", "storm"],
        emotion_style="fast and heavy, distorted downtuned guitars, pounding drums, "
                      "aggressive shouted chorus, furious and explosive, minor key",
        bpm=168, keyscale="E minor",
    ),

    "vigilance": Emotion(
        name="vigilance", plutchik_dyad="anticipation", valence=0.55, arousal=0.65,
        anchor_prompts=[
            "a tense, watchful song full of anticipation",
            "a driving, alert track building with expectation",
            "suspenseful music, poised and ready, on the edge",
            "focused, determined, coiled and waiting",
        ],
        lexicon=["anticipation", "vigilance", "expect", "ready", "watch", "alert",
                 "await", "poised", "brace", "coming", "storm", "prepare", "focus",
                 "edge", "counting", "steady"],
        emotion_style="driving mid-fast tempo, insistent palm-muted guitar pulse, tense "
                      "and coiled, building anticipation, alert and expectant, minor-tinged",
        bpm=132, keyscale="A minor",
    ),
}

# Canonical ordered list/tuple (matches config.EMOTIONS order).
ORDER: list[str] = list(EMOTIONS)
ALL: list[Emotion] = [_EMOTIONS[name] for name in ORDER]


def get(name: str) -> Emotion:
    return _EMOTIONS[name]


def anchor_prompts() -> dict[str, list[str]]:
    return {e.name: e.anchor_prompts for e in ALL}


def caption_candidates(name: str) -> list[str]:
    return list(_EMOTIONS[name].captions)


def valence_arousal() -> dict[str, tuple[float, float]]:
    return {e.name: (e.valence, e.arousal) for e in ALL}


def _validate_definitions() -> None:
    """Sanity checks run at import: names line up, fields are well-formed."""
    assert set(_EMOTIONS) == set(EMOTIONS), "emotion set mismatch vs config.EMOTIONS"
    for e in ALL:
        assert e.emotion_style, f"{e.name}: needs an emotion_style"
        assert len(e.caption) <= 512, f"{e.name}: caption exceeds ACE-Step's 512-char limit"
        assert 0.0 <= e.valence <= 1.0 and 0.0 <= e.arousal <= 1.0, f"{e.name}: V/A out of range"
        assert len(e.anchor_prompts) >= 3, f"{e.name}: needs >=3 anchor prompts"
        assert len(e.lexicon) >= 8, f"{e.name}: lexicon too small"


_validate_definitions()
