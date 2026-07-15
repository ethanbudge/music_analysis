"""
lyric_encoder.py — the NEW lyric text tower.

This is the direct fix for MuLan's core weakness: MuLan's text encoder was trained
on music *descriptions*, so lyrics are out-of-distribution for it. Here we embed
lyrics with a proper *sentence* encoder that was trained on natural language, so the
lyric side is in-distribution. These vectors feed both LyricLMC (Step 3) and the
geometry comparison (Step 4).

Default model: `sentence-transformers/all-MiniLM-L6-v2` (384-d, tiny, fast, CPU-fine).
Swap it via config.TEXT_ENCODER / env LMCMETRICS_TEXT_ENCODER, e.g.
  - "intfloat/e5-base-v2"                       (stronger, topic-leaning)
  - an emotion-tuned encoder of your choice     (affect-leaning research variant)

We prefer the `sentence-transformers` package; if it is not installed we fall back
to a plain HuggingFace `transformers` model with mean-pooling, so the code still
runs. Vectors are cached to data/lmcmetrics/lyrics/<track_id>.npy (resumable).
"""

from __future__ import annotations
import logging

import numpy as np

from . import config
from lmc import db as projdb
from lmc.utils import lrc_to_plaintext

logger = logging.getLogger(__name__)


def _vec_path(track_id: int):
    return config.LYRICS_VEC_DIR / f"{track_id}.npy"


# ── encoder wrapper (sentence-transformers, with a transformers fallback) ─────────
class LyricEncoder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or config.TEXT_ENCODER
        from lmc.utils import get_device
        self.device = device or get_device()
        self._st = None       # sentence-transformers model
        self._hf = None       # (tokenizer, model) fallback
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading lyric encoder (sentence-transformers): %s", self.model_name)
            self._st = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:                                 # noqa: BLE001
            logger.warning("sentence-transformers unavailable (%s); using transformers "
                           "mean-pooling fallback.", e)
            import torch
            from transformers import AutoTokenizer, AutoModel
            self.torch = torch
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._hf = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @property
    def dim(self) -> int:
        if self._st is not None:
            return int(self._st.get_sentence_embedding_dimension())
        return int(self._hf.config.hidden_size)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of documents -> [len(texts), D] float32 (L2-normalised)."""
        if self._st is not None:
            v = self._st.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        # transformers fallback: mean-pool last hidden state over real tokens.
        import numpy as _np
        out = []
        for t in texts:
            enc = self._tok(t, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with self.torch.no_grad():
                hs = self._hf(**enc).last_hidden_state[0]            # [T, H]
            mask = enc["attention_mask"][0].unsqueeze(-1).float()   # [T, 1]
            vec = (hs * mask).sum(0) / mask.sum().clamp(min=1.0)
            vec = vec.cpu().numpy()
            out.append(vec / (np.linalg.norm(vec) + 1e-12))
        return _np.asarray(out, dtype=_np.float32)


# ── driver: cache one lyric vector per song ───────────────────────────────────────
def _song_lyrics(row) -> str:
    """Prefer stored plain lyrics; otherwise strip timestamps from synced LRC."""
    txt = (row["plain_lyrics"] or "").strip() if "plain_lyrics" in row.keys() else ""
    if not txt:
        txt = lrc_to_plaintext(row["synced_lyrics"] or "")
    return txt


def embed_pending(limit: int | None = None, force: bool = False,
                  model_name: str | None = None, batch_size: int = 64) -> dict:
    """Compute + cache a lyric vector for every corpus song that lacks one."""
    config.ensure_dirs()
    with projdb.connect() as conn:
        songs = [dict(r) for r in conn.execute(
            "SELECT track_id, plain_lyrics, synced_lyrics FROM songs")]

    todo = [s for s in songs if force or not _vec_path(s["track_id"]).exists()]
    todo = [s for s in todo if _song_lyrics(_Row(s)).strip()]
    if limit:
        todo = todo[:limit]
    if not todo:
        logger.info("Lyric vectors: nothing pending.")
        return {"attempted": 0, "done": 0, "dim": None}

    enc = LyricEncoder(model_name)
    logger.info("Lyric vectors: %d songs with %s (dim=%d).", len(todo), enc.model_name, enc.dim)

    ok = 0
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        texts = [_song_lyrics(_Row(s)) for s in chunk]
        vecs = enc.encode(texts)
        for s, v in zip(chunk, vecs):
            np.save(_vec_path(s["track_id"]), v.astype(np.float32))
            ok += 1
        logger.info("  [%d/%d]", min(start + batch_size, len(todo)), len(todo))

    logger.info("Lyric vectors done: %d songs (dim=%d).", ok, enc.dim)
    return {"attempted": len(todo), "done": ok, "dim": enc.dim}


def load_vectors(track_ids: list[int]) -> dict[int, np.ndarray]:
    out = {}
    for tid in track_ids:
        p = _vec_path(tid)
        if p.exists():
            out[tid] = np.load(p)
    return out


class _Row(dict):
    """Tiny shim so dict rows expose .keys() like sqlite3.Row in _song_lyrics."""
    def keys(self):
        return super().keys()
    def __getitem__(self, k):
        return super().get(k)
