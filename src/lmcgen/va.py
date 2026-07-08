"""
va.py — Valence/Arousal instrument for the generative arm (Tier-1 fix).

MuLan reads genre/timbre, so once genre is held constant it can no longer see the
emotion manipulation (the v2 run's manipulation check came out at chance). Emotion
*within* a fixed genre lives in valence/arousal, which DOES vary by our tempo / mode
/ dynamics design. This module measures VA on both sides so congruence can be scored
in VA space instead of MuLan space:

  audio_va(path)   valence/arousal of a clip from librosa acoustics — REUSES
                   lmc.mood.extract_features (tempo, energy, mode, brightness) and
                   reduces its mood tags to the 2 circumplex axes (Russell/Thayer).
  lyric_va(text)   valence/arousal of a lyric from a bundled compact VAD lexicon
                   (ANEW / NRC-VAD-style word norms). Coarse on short hooks — a full
                   NRC-VAD lexicon or a GoEmotions/VAD model would sharpen it (noted).

Both return coordinates in [0,1] (0.5 = neutral), matching emotions.Emotion.valence
/arousal, so audio/lyric VA can be compared directly to the design targets.
"""
from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)


# ─── audio VA (reuses the observational arm's librosa mood features) ─────────────
def audio_va(audio_path) -> tuple[float, float] | None:
    """(valence, arousal) in [0,1] from acoustics. None if the file can't be read.

    valence  ← the happy↔sad axis (mode + brightness + tempo/energy tilt)
    arousal  ← the energy axis (aggressive/party vs relaxed)
    """
    from lmc.mood import extract_features
    feats = extract_features(str(audio_path))
    if not feats:
        return None
    valence = _clip01(0.5 + 0.5 * (feats["mood_happy"] - feats["mood_sad"]))
    arousal = _clip01((feats["mood_aggressive"] + feats["mood_party"]
                       + (1.0 - feats["mood_relaxed"])) / 3.0)
    return valence, arousal


# ─── lyric VA (bundled compact VAD lexicon) ──────────────────────────────────────
# Approximate valence/arousal norms in [0,1] for common affect words (ANEW / NRC-VAD
# style; 0.5 = neutral). Deliberately compact + transparent — a stopgap. For a thesis,
# swap in the full NRC-VAD lexicon (Mohammad 2018, ~20k words) or a text VAD model.
_VAD: dict[str, tuple[float, float]] = {
    # positive / high-arousal
    "alive": (.85, .72), "gold": (.80, .55), "bright": (.78, .60), "alight": (.72, .62),
    "sky": (.68, .45), "joy": (.90, .70), "happy": (.88, .65), "love": (.87, .60),
    "shine": (.80, .58), "celebrate": (.88, .72), "thrill": (.80, .80), "soaring": (.78, .70),
    "hope": (.78, .55), "dream": (.72, .50), "smile": (.85, .55), "light": (.72, .50),
    "wonder": (.82, .62), "wide": (.58, .58), "open": (.62, .48), "today": (.60, .45),
    # trust / warmth (mid arousal)
    "follow": (.58, .42), "truest": (.78, .42), "true": (.75, .40), "heart": (.72, .55),
    "hold": (.62, .40), "faithful": (.78, .40), "trust": (.78, .45), "honest": (.75, .42),
    "steady": (.65, .35), "cherish": (.82, .50), "peace": (.78, .28), "calm": (.75, .25),
    "warm": (.75, .40), "company": (.66, .40),
    # fear / high-arousal negative
    "run": (.38, .75), "dark": (.30, .48), "closing": (.32, .58), "fear": (.18, .78),
    "terror": (.12, .85), "dread": (.15, .72), "panic": (.15, .82), "scream": (.22, .85),
    "shadow": (.30, .52), "afraid": (.20, .72), "danger": (.20, .78), "storm": (.32, .72),
    "flee": (.25, .78), "chase": (.30, .70), "nightmare": (.15, .75),
    # anger / disgust
    "burn": (.30, .82), "burning": (.35, .80), "fire": (.42, .78), "rage": (.15, .88),
    "fury": (.15, .85), "anger": (.18, .80), "hate": (.12, .68), "fight": (.28, .78),
    "tear": (.30, .70), "crossed": (.42, .55), "poison": (.15, .62), "rotten": (.15, .52),
    "shame": (.20, .55), "disgust": (.18, .62), "sick": (.22, .55), "venom": (.15, .62),
    "vile": (.15, .58), "scorn": (.20, .55), "wall": (.45, .40),
    # sadness / low-arousal negative
    "quiet": (.45, .22), "ghost": (.28, .45), "grief": (.15, .48), "sorrow": (.18, .42),
    "lonely": (.22, .38), "alone": (.28, .40), "tears": (.24, .52), "cry": (.22, .55),
    "gone": (.28, .40), "lost": (.28, .45), "empty": (.25, .38), "rain": (.42, .35),
    "used": (.48, .35), "cold": (.35, .38), "pain": (.20, .60), "fade": (.35, .35),
    # anticipation / neutral-ish
    "ready": (.62, .60), "counting": (.50, .52), "watch": (.52, .50), "eyes": (.52, .48),
    "door": (.50, .42), "come": (.55, .45), "around": (.50, .40), "wait": (.45, .40),
    "signal": (.52, .55), "edge": (.42, .60), "brace": (.40, .62),
    # generic
    "night": (.48, .45), "down": (.40, .42), "back": (.48, .40), "away": (.40, .45),
    "everything": (.55, .50), "nothing": (.35, .42), "something": (.48, .45),
    "name": (.50, .40), "house": (.55, .32), "hands": (.55, .50), "done": (.45, .48),
    "line": (.50, .42), "mine": (.55, .45), "world": (.58, .50), "breath": (.55, .52),
    "nowhere": (.32, .50), "same": (.48, .35), "knows": (.52, .42), "built": (.55, .40),
}


_WORD = re.compile(r"[a-z']+")

# The active lexicon. Defaults to the bundled stopgap; use_nrc_vad() swaps in the full
# NRC-VAD lexicon (Mohammad 2018) for far better coverage on real lyrics.
_ACTIVE_VAD: dict[str, tuple[float, float]] = dict(_VAD)


def use_nrc_vad(path: str | None = None) -> int:
    """Load the full NRC-VAD lexicon and use it for lyric_va(). Pass the path to
    `NRC-VAD-Lexicon.txt` (term<TAB>valence<TAB>arousal<TAB>dominance, values in [0,1]),
    or set env LMCGEN_NRC_VAD. Download: https://saifmohammad.com/WebPages/nrc-vad.html
    (free for research). Returns the number of words loaded. Falls back silently to the
    bundled lexicon if the file isn't found."""
    import os
    global _ACTIVE_VAD
    path = path or os.getenv("LMCGEN_NRC_VAD")
    if not path or not os.path.exists(path):
        logger.warning("NRC-VAD not found (%s); keeping the bundled stopgap lexicon.", path)
        return len(_ACTIVE_VAD)
    lex: dict[str, tuple[float, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                try:
                    lex[parts[0].lower()] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    continue                       # header / malformed line
    if lex:
        _ACTIVE_VAD = lex
        logger.info("Loaded NRC-VAD: %d words.", len(lex))
    return len(_ACTIVE_VAD)


def lyric_va(text: str) -> tuple[float, float, int]:
    """(valence, arousal, n_matched) — mean VA over lexicon-matched words. Falls back
    to neutral (0.5, 0.5) when no words match (reported via n_matched=0)."""
    lex = _ACTIVE_VAD
    toks = _WORD.findall(text.lower())
    hits = [lex[t] for t in toks if t in lex]
    # light stemming: try trimming a trailing 's'/'ing'/'ed' if no exact hit
    if not hits:
        for t in toks:
            for stem in (t.rstrip("s"), t[:-3] if t.endswith("ing") else t,
                         t[:-2] if t.endswith("ed") else t):
                if stem in lex:
                    hits.append(lex[stem]); break
    if not hits:
        return 0.5, 0.5, 0
    v = sum(h[0] for h in hits) / len(hits)
    a = sum(h[1] for h in hits) / len(hits)
    return v, a, len(hits)


# ─── congruence in VA space ──────────────────────────────────────────────────────
def va_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two VA points (0 = identical, ~1.41 = opposite corner)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def va_congruence(a: tuple[float, float], b: tuple[float, float]) -> float:
    """VA-space congruence in [0,1]: 1 = same emotion point, 0 = opposite corner."""
    return 1.0 - va_distance(a, b) / (2.0 ** 0.5)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
