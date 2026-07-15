"""
lyrics.py — The sixteen two-line lyric stimuli (four per VA corner).

These are ORIGINAL, clean, non-derivative couplets written for this study — short
enough to be simple, long enough to carry a clear semantic viewpoint. Word choice is
grounded in each corner's valence/arousal vocabulary (quadrants.Quadrant.lexicon) so
each couplet lands unambiguously in one corner. Two independent lines of evidence back
that placement:

  1. Lexical (model-independent) — `lexical_alignment()` counts how many of each
     corner's lexicon words appear in each couplet; the target corner should win.
  2. VA (lexicon-VAD) — `va_alignment()` scores each couplet's valence/arousal with
     the va.py lexicon and checks its nearest corner is the target.
  (A third, MuLan text-anchor cosine, is available in the notebook — text side only,
   so it is independent of the music generation.)

Because these are the experimental stimuli (the design itself) they live in code and
are version-controlled. Lyric ids are "<corner>_<n>", e.g. "hvha_1".
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from .config import QUADRANTS, LYRICS_PER_QUADRANT
from . import quadrants as q


@dataclass(frozen=True)
class Couplet:
    lyric_id: str         # "<corner>_<n>", e.g. "hvha_1"
    quadrant: str         # target VA corner (one of config.QUADRANTS)
    text: str             # the two-line lyric (newline-separated)

    @property
    def lines(self) -> list[str]:
        return [ln.strip() for ln in self.text.strip().splitlines() if ln.strip()]

    @property
    def plain(self) -> str:
        """Plain sung text (WER reference)."""
        return " ".join(self.lines)

    def chorus_block(self) -> str:
        """The lyrics as a Lyria `[Chorus]` block."""
        return "[Chorus]\n" + "\n".join(self.lines)


# ─── The sixteen couplets (4 per corner) ─────────────────────────────────────────
_TEXT: dict[str, list[str]] = {

    "hvha": [   # high valence / high arousal — joyful, euphoric
        "Hands in the air, we're burning gold tonight\nAlive, alive, the whole world's alight",
        "Turn it up loud, we're dancing through the fire\nHigher and higher, we never tire",
        "Feel the rush, the city's ours to run\nChasing the light, we're second to none",
        "Electric hearts and a sky full of flame\nShout it out loud, we're never the same",
    ],

    "hvla": [   # high valence / low arousal — calm, content, serene
        "Soft morning light, your hand holds mine\nSlow and warm, everything's fine",
        "Rest your head, the world is still\nQuiet and safe on this gentle hill",
        "A calm breeze hums, the evening's kind\nPeace in my chest, ease in my mind",
        "Home at last and the fire burns low\nClose and warm, nowhere to go",
    ],

    "lvha": [   # low valence / high arousal — angry, afraid, tense
        "Burn it down, I'll fight you to the end\nTear down every wall you defend",
        "Run, don't stop, the dark is closing in\nSomething with teeth knows where I've been",
        "Blood in my mouth, I'm ready for the war\nBreak down the walls, I want more",
        "Fists and fire, the storm is at the door\nScream until my throat is raw",
    ],

    "lvla": [   # low valence / low arousal — sad, weary, hopeless
        "The house is quiet where you used to be\nI keep your ghost for company",
        "Cold in the rain, the streetlights fade\nAlone with the wreck that we made",
        "Empty hands and a hollow chest\nToo tired now to even rest",
        "Gone, all gone, the ashes fall\nI don't feel much of anything at all",
    ],
}


def _build() -> dict[str, Couplet]:
    out: dict[str, Couplet] = {}
    for code in QUADRANTS:
        texts = _TEXT[code]
        for i, t in enumerate(texts, 1):
            lid = f"{code}_{i}"
            out[lid] = Couplet(lyric_id=lid, quadrant=code, text=t)
    return out


COUPLETS: dict[str, Couplet] = _build()
ORDER: list[str] = list(COUPLETS)                 # grouped by corner, then 1..4
ALL: list[Couplet] = [COUPLETS[lid] for lid in ORDER]


def get(lyric_id: str) -> Couplet:
    return COUPLETS[lyric_id]


def by_quadrant(code: str) -> list[Couplet]:
    return [c for c in ALL if c.quadrant == code]


# ─── Lexical (model-independent) alignment evidence ──────────────────────────────
_WORD_RE = re.compile(r"[a-z']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _stem_match(tok: str, root: str) -> bool:
    """Stem-tolerant match: equal, or the shorter is a >=4-char prefix of the longer
    (burn/burning, tear/tears) without over-matching short words."""
    if tok == root:
        return True
    short, long = (tok, root) if len(tok) <= len(root) else (root, tok)
    return len(short) >= 4 and long.startswith(short)


def _lexicon_hits(tokens: list[str], lexicon: list[str]) -> int:
    roots = [w.lower() for w in lexicon]
    return sum(any(_stem_match(tok, root) for root in roots) for tok in tokens)


def lexical_alignment():
    """Score every couplet against every corner's lexicon. Returns a DataFrame indexed
    by lyric_id, columns = corner lexicons, values = matching-word counts, + 'quadrant'
    (target), 'predicted' (argmax) and 'correct'."""
    import pandas as pd
    rows = {}
    for c in ALL:
        toks = _tokens(c.plain)
        rows[c.lyric_id] = {qq.code: _lexicon_hits(toks, qq.lexicon) for qq in q.ALL}
    df = pd.DataFrame(rows).T[q.ORDER]
    df.index.name = "lyric_id"
    df.insert(0, "quadrant", [COUPLETS[i].quadrant for i in df.index])
    df["predicted"] = df[q.ORDER].idxmax(axis=1)
    df["correct"] = df["predicted"] == df["quadrant"]
    return df


def va_alignment():
    """Score each couplet's valence/arousal with the va.py lexicon and check its nearest
    VA corner is the target. Returns a DataFrame indexed by lyric_id."""
    import pandas as pd
    from . import va
    rows = []
    for c in ALL:
        v, a, n = va.lyric_va(c.plain)
        pred = q.nearest_quadrant(v, a)
        rows.append({"lyric_id": c.lyric_id, "quadrant": c.quadrant,
                     "lyric_v": round(v, 3), "lyric_a": round(a, 3), "n_matched": n,
                     "predicted": pred, "correct": pred == c.quadrant})
    return pd.DataFrame(rows).set_index("lyric_id")


def _validate_definitions() -> None:
    assert set(_TEXT) == set(QUADRANTS), "lyric corner set mismatch vs config.QUADRANTS"
    for code in QUADRANTS:
        assert len(_TEXT[code]) == LYRICS_PER_QUADRANT, \
            f"{code}: need exactly {LYRICS_PER_QUADRANT} couplets"
    for c in ALL:
        assert len(c.lines) == 2, f"{c.lyric_id}: must be exactly two lines"


_validate_definitions()
