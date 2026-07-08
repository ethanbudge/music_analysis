"""
lyrics.py — The eight chorus stimuli (lyric emotion held fixed within a chorus).

These are ORIGINAL, clean, non-derivative choruses written for this study. They
are deliberately *not* modelled on any existing song; word choice is grounded in
each emotion's Plutchik / NRC EmoLex vocabulary (see emotions.Emotion.lexicon) so
that each chorus lands unambiguously in one emotion. Two independent lines of
evidence back that placement:

  1. Lexical (model-independent) — `lexical_alignment()` counts how many of each
     emotion's lexicon words appear in each chorus; the target emotion should win.
  2. Embedding (MuLan) — mulan/pipeline embed each chorus's text and score cosine
     against the eight emotion anchors; the target anchor should be nearest. This
     is the "cosine-similarity evidence" the study requires, and it is computed on
     the TEXT side only, so it is independent of the music generation.

Because these are the experimental stimuli (the research design itself), they live
in code and are version-controlled — unlike scraped third-party lyrics, which the
repo's .gitignore deliberately excludes.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from .config import EMOTIONS
from . import emotions as emo


@dataclass(frozen=True)
class Chorus:
    emotion: str          # target emotion (one of config.EMOTIONS)
    text: str             # the chorus lyric (newline-separated lines)
    rationale: str        # why these words target this emotion

    @property
    def lines(self) -> list[str]:
        return [ln.strip() for ln in self.text.strip().splitlines() if ln.strip()]

    @property
    def plain(self) -> str:
        return " ".join(self.lines)

    def acestep_lyrics(self, repeats: int = 2) -> str:
        """ACE-Step lyric field. Short 2-line hooks are repeated so the clip reaches
        ~10-14 s and the hook lands twice (catchier, and it reinforces intelligibility).
        The repeated block is what gets *sung*, so WER screening compares against it."""
        block = "\n".join(self.lines)
        return "[Chorus]\n" + ("\n".join([block] * max(1, repeats)))


# ─── The eight choruses ──────────────────────────────────────────────────────────
# Two-line hooks. Kept short, catchy, monosyllable-heavy and clearly metered so the
# vocal renders the words cleanly (see WER screening) — and repeated at generation
# time (Chorus.acestep_lyrics) so the hook lands twice.
CHORUSES: dict[str, Chorus] = {

    "ecstasy": Chorus(
        emotion="ecstasy",
        text="""
        Hands to the sky, we're burning gold tonight
        Alive, alive, everything's alight
        """,
        rationale="Joy words (gold, alive, burning, light, sky) at high valence/arousal "
                  "give a euphoric, celebratory hook.",
    ),

    "admiration": Chorus(
        emotion="admiration",
        text="""
        I'd follow you through fire, through the cold
        You're the truest heart I'll ever hold
        """,
        rationale="Trust/devotion words (follow, truest, heart, hold) convey warm, "
                  "loyal admiration.",
    ),

    "terror": Chorus(
        emotion="terror",
        text="""
        Run, don't look back, it's closing in
        Something in the dark knows where I've been
        """,
        rationale="Fear cues (run, dark, closing in) build dread and panic at high "
                  "arousal, low valence.",
    ),

    "amazement": Chorus(
        emotion="amazement",
        text="""
        Out of nowhere, it took my breath away
        Eyes wide open, nothing looks the same today
        """,
        rationale="Surprise/awe words (out of nowhere, breath away, eyes wide) mark "
                  "sudden amazement.",
    ),

    "grief": Chorus(
        emotion="grief",
        text="""
        The house is quiet where you used to be
        I keep your ghost for company
        """,
        rationale="Loss/mourning imagery (quiet, ghost, used to be) evokes sorrow at "
                  "low valence and low arousal.",
    ),

    "loathing": Chorus(
        emotion="loathing",
        text="""
        Keep your poison, keep your rotten name
        I'm done breathing in your shame
        """,
        rationale="Disgust/contempt words (poison, rotten, done, shame) convey "
                  "revulsion and scorn.",
    ),

    "rage": Chorus(
        emotion="rage",
        text="""
        Burn it down, I'm done, I've crossed the line
        Every wall you built is mine
        """,
        rationale="Anger words (burn, done, crossed the line) give a furious, defiant "
                  "high-arousal hook.",
    ),

    "vigilance": Chorus(
        emotion="vigilance",
        text="""
        Eyes on the door, I'm counting down
        Ready for the storm to come around
        """,
        rationale="Anticipation cues (eyes on, counting down, ready, storm coming) "
                  "convey alert, coiled expectancy.",
    ),
}

ORDER: list[str] = list(EMOTIONS)
ALL: list[Chorus] = [CHORUSES[name] for name in ORDER]


def get(emotion: str) -> Chorus:
    return CHORUSES[emotion]


# ─── Lexical (model-independent) alignment evidence ──────────────────────────────
_WORD_RE = re.compile(r"[a-z']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _stem_match(tok: str, root: str) -> bool:
    """Stem-tolerant match: equal, or the shorter is a >=4-char prefix of the longer.

    Handles dance/dancing, roar/roaring, burn/burning without over-matching short
    words (min shared prefix of 4 chars).
    """
    if tok == root:
        return True
    short, long = (tok, root) if len(tok) <= len(root) else (root, tok)
    return len(short) >= 4 and long.startswith(short)


def _lexicon_hits(tokens: list[str], lexicon: list[str]) -> int:
    """Count tokens matching any lexicon entry (each token counts at most once)."""
    roots = [w.lower() for w in lexicon]
    return sum(any(_stem_match(tok, root) for root in roots) for tok in tokens)


def lexical_alignment():
    """
    Score every chorus against every emotion's lexicon.

    Returns a pandas DataFrame indexed by chorus emotion, columns = emotion
    lexicons, values = number of matching words, plus 'predicted' (argmax) and
    'correct' (predicted == target). This is the model-INDEPENDENT evidence that
    each chorus sits in its intended emotion.
    """
    import pandas as pd
    rows = {}
    for ch in ALL:
        toks = _tokens(ch.plain)
        rows[ch.emotion] = {e.name: _lexicon_hits(toks, e.lexicon) for e in emo.ALL}
    df = pd.DataFrame(rows).T[ORDER]           # rows=target chorus, cols=lexicon
    df.index.name = "chorus_emotion"
    df["predicted"] = df[ORDER].idxmax(axis=1)
    df["correct"] = df["predicted"] == df.index
    return df


def _validate_definitions() -> None:
    assert set(CHORUSES) == set(EMOTIONS), "chorus set mismatch vs config.EMOTIONS"
    for ch in ALL:
        assert len(ch.lines) >= 2, f"{ch.emotion}: hook needs >=2 lines"
        assert ch.rationale, f"{ch.emotion}: missing rationale"


_validate_definitions()
