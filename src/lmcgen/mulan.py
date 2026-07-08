"""
mulan.py — Joint-embedding scorer + emotion-anchor builder.

Reuses the observational arm's embedders (`lmc.embeddings._MuLan` / `_CLAP`) so the
generative arm scores stimuli in *exactly* the same space the main study measures
LMC in. Everything torch/muq/laion_clap is imported lazily by those classes, so
importing this module stays cheap until you actually build a Scorer.

Two things live here:

  Scorer          Wraps one loaded embedder and exposes embed_text(str) -> [D],
                  embed_audio_file(path) -> [D], and cosine helpers.

  build_anchors   For each emotion, embeds its anchor prompts (emotions.py) with the
                  TEXT tower and averages them into a unit anchor vector. Because
                  MuLan/CLAP are joint spaces, the same anchor scores both lyric text
                  and generated audio. Anchors are cached to data/generation/anchors/.
"""
from __future__ import annotations
import logging
import numpy as np

from .config import ANCHOR_DIR
from . import emotions as emo

logger = logging.getLogger(__name__)


class Scorer:
    """A loaded joint embedder ('mulan' or 'clap') with text/audio embedding helpers."""

    def __init__(self, model_key: str = "mulan", device: str | None = None):
        from lmc.utils import get_device
        from lmc.embeddings import _MuLan, _CLAP
        self.model_key = model_key
        self.device = device or get_device()
        cls = {"mulan": _MuLan, "clap": _CLAP}[model_key]
        self._emb = cls(self.device)
        self.sr = self._emb.sr
        self.dim = self._emb.dim

    # ── embedding ────────────────────────────────────────────────────────────────
    def embed_text(self, text: str) -> np.ndarray | None:
        return self._emb.embed_text(text)

    def embed_audio(self, wav: np.ndarray) -> np.ndarray | None:
        return self._emb.embed_audio(wav)

    def embed_audio_file(self, path) -> np.ndarray | None:
        import librosa
        try:
            wav, _ = librosa.load(str(path), sr=self.sr, mono=True)
        except Exception as e:                                     # noqa: BLE001
            logger.warning("  audio load failed for %s: %s", path, e)
            return None
        return self.embed_audio(wav)

    # ── anchors ──────────────────────────────────────────────────────────────────
    def build_anchors(self, force: bool = False) -> dict[str, np.ndarray]:
        """Build (and cache) one unit anchor vector per emotion from its prompts."""
        from lmc.utils import load_song_embeddings, save_song_embeddings
        ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
        path = ANCHOR_DIR / f"anchors_{self.model_key}.npz"
        if not force and path.exists():
            cached = load_song_embeddings(path)
            if cached and set(cached) >= set(emo.ORDER):
                logger.info("Anchors[%s]: loaded cached.", self.model_key)
                return {name: cached[name] for name in emo.ORDER}

        logger.info("Anchors[%s]: embedding %d emotion prompt sets…", self.model_key, len(emo.ALL))
        anchors: dict[str, np.ndarray] = {}
        for e in emo.ALL:
            vecs = [self.embed_text(p) for p in e.anchor_prompts]
            vecs = [v for v in vecs if v is not None]
            if not vecs:
                raise RuntimeError(f"anchor embedding failed for {e.name} ({self.model_key})")
            anchors[e.name] = _unit(np.mean(np.stack(vecs), axis=0))
        save_song_embeddings(path, anchors)
        return anchors

    # ── scoring ──────────────────────────────────────────────────────────────────
    def score_against_anchors(self, vec: np.ndarray,
                              anchors: dict[str, np.ndarray]) -> dict[str, float]:
        """Cosine of `vec` against every emotion anchor (emotion -> cosine)."""
        from lmc.utils import cosine_sim
        return {name: cosine_sim(vec, anchors[name]) for name in emo.ORDER}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n else v.astype(np.float32)


def argmax_emotion(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Convenience cosine that doesn't require a Scorer (delegates to lmc.utils)."""
    from lmc.utils import cosine_sim
    return cosine_sim(a, b)
