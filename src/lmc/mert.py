"""
mert.py — Per-song MERT audio embeddings, cached as control features.

MERT-v1-330M is an audio-only self-supervised model. We compute one 1024-d vector
per song (mean-pooled over time and layers) and cache it to data/embeddings/mert/.
combine.build_master() then PCA-reduces these to `mert_pc01..K` columns that the
Stan models use as controls (see config.MERT, run_models.R control toggle).

Why MERT for controls: it is a DIFFERENT representation than the MuLan/CLAP space
that LMC is computed in, so its components absorb production / era / genre nuisance
variance WITHOUT partialling out LMC's own inputs (avoids the over-control /
collider problem of using PCs of the LMC-generating embedding — MODEL_NOTES §4.5).

Resumable: a song is skipped if its .npy already exists (unless force=True).
"""

from __future__ import annotations
import logging

import numpy as np

from .config import MERT, MERT_DIR
from .utils import get_device
from . import db as projdb

logger = logging.getLogger(__name__)


def _vec_path(track_id: int):
    return MERT_DIR / f"{track_id}.npy"


def extract_pending(limit: int | None = None, force: bool = False,
                    device: str | None = None) -> dict:
    """Compute + cache the MERT vector for every song with audio but no vector."""
    import librosa
    from .embeddings import _MERT
    MERT_DIR.mkdir(parents=True, exist_ok=True)
    device = device or get_device()

    with projdb.connect() as conn:
        have_audio = [dict(r) for r in projdb.songs_with_audio(conn)]
        cached = set() if force else {
            r["track_id"] for r in conn.execute("SELECT track_id FROM mert")}

    todo = [s for s in have_audio if force or s["track_id"] not in cached]
    todo = [s for s in todo if force or not _vec_path(s["track_id"]).exists()]
    if limit:
        todo = todo[:limit]
    if not todo:
        logger.info("MERT: nothing pending.")
        return {"attempted": 0, "done": 0}

    logger.info("MERT: %d songs (device=%s).", len(todo), device)
    emb = _MERT(device)

    ok = 0
    for i, song in enumerate(todo, 1):
        tid = song["track_id"]
        logger.info("[%d/%d] %s", i, len(todo), tid)
        try:
            wav, _ = librosa.load(song["file_path"], sr=MERT["audio_sr"], mono=True)
        except Exception as e:                                 # noqa: BLE001
            logger.warning("  audio load failed: %s", e)
            continue
        vec = emb.embed_audio(wav)
        if vec is None or not np.any(vec):
            logger.warning("  [mert] empty embedding for %s — skipping.", tid)
            continue
        path = _vec_path(tid)
        np.save(path, vec.astype(np.float32))
        with projdb.connect() as conn:
            projdb.upsert(conn, "mert", {
                "track_id": tid, "path": str(path), "dim": int(vec.shape[0])})
        ok += 1

    logger.info("MERT done: %d songs.", ok)
    return {"attempted": len(todo), "done": ok}


def load_vectors(track_ids: list[int]) -> dict[int, np.ndarray]:
    """Load cached MERT vectors for the given tracks (missing ones are skipped)."""
    out = {}
    for tid in track_ids:
        p = _vec_path(tid)
        if p.exists():
            out[tid] = np.load(p)
    return out
