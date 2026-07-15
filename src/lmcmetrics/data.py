"""
data.py — load matched (audio, lyric) representations from the existing caches,
and generate synthetic data so the whole package can be tested with no corpus.

A PairSet is the common currency: N songs, each with an audio vector and a lyric
vector (possibly different dimensionalities), plus optional group labels (e.g.
genre) used for hard-negative evaluation.

Loaders
-------
load_joint_pairset(model)      audio = bundle 'audio_full', text = bundle 'text_full'
                               for MuLan or CLAP (SAME 512-d joint space). Use this to
                               evaluate the CURRENT metric and the centering fix.
load_lyriclmc_pairset(audio)   audio = MERT (or MuLan/CLAP audio), text = the NEW
                               lyric sentence vectors. Different dims. Use this for
                               LyricLMC (Step 3) and the geometry comparison (Step 4).

Synthetic
---------
synthetic_pairset(...)         different-dim audio/text with a tunable congruence
                               signal — exercises LyricLMC + geometry.
synthetic_joint_pairset(...)   same-dim audio/text with a simulated modality gap —
                               exercises the cosine harness + the centering fix.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np

from . import config
from lmc import db as projdb
from lmc import mert as mert_mod
from lmc.utils import load_song_embeddings, embedding_path
from . import lyric_encoder

logger = logging.getLogger(__name__)


@dataclass
class PairSet:
    track_ids: list[int]
    audio: np.ndarray                     # [N, audio_dim]
    text: np.ndarray                      # [N, text_dim]
    groups: np.ndarray | None = None      # [N] optional labels (e.g. genre)
    audio_kind: str = "unknown"
    text_kind: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.track_ids)

    @property
    def same_space(self) -> bool:
        return self.audio.shape[1] == self.text.shape[1]


# ── real loaders ──────────────────────────────────────────────────────────────────
def _genre_map() -> dict[int, str]:
    try:
        with projdb.connect() as conn:
            return {r["track_id"]: (r["genre"] or "unknown")
                    for r in conn.execute("SELECT track_id, genre FROM genre")}
    except Exception:                                          # noqa: BLE001
        return {}


def load_joint_pairset(model: str = "mulan", limit: int | None = None) -> PairSet:
    """Audio_full vs text_full from the cached MuLan/CLAP bundles (same 512-d space)."""
    with projdb.connect() as conn:
        tids = [r["track_id"] for r in conn.execute(
            "SELECT track_id FROM embeddings WHERE model=? ORDER BY track_id", (model,))]
    if limit:
        tids = tids[:limit]
    gmap = _genre_map()
    ids, A, T, G = [], [], [], []
    for tid in tids:
        bundle = load_song_embeddings(embedding_path(config.EMBEDDINGS_DIR, model, tid))
        if bundle is None:
            continue
        a, t = bundle.get("audio_full"), bundle.get("text_full")
        if a is None or t is None or not np.any(a) or not np.any(t):
            continue
        ids.append(tid); A.append(a); T.append(t); G.append(gmap.get(tid, "unknown"))
    if not ids:
        raise RuntimeError(f"No usable {model} bundles found — has the pipeline run?")
    logger.info("Loaded joint PairSet [%s]: %d songs.", model, len(ids))
    return PairSet(ids, np.stack(A), np.stack(T), np.asarray(G),
                   audio_kind=f"{model}_audio", text_kind=f"{model}_text")


def load_lyriclmc_pairset(audio: str | None = None, limit: int | None = None) -> PairSet:
    """Audio (MERT by default) vs the NEW lyric sentence vectors — different dims.

    Requires lyric vectors to be cached first (lyric_encoder.embed_pending()).
    """
    audio = audio or config.AUDIO_REP
    with projdb.connect() as conn:
        tids = [r["track_id"] for r in conn.execute("SELECT track_id FROM songs ORDER BY track_id")]
    if limit:
        tids = tids[:limit]

    lyric_vecs = lyric_encoder.load_vectors(tids)
    if audio == "mert":
        audio_vecs = mert_mod.load_vectors(tids)
    else:                                            # 'mulan' | 'clap' audio_full
        audio_vecs = {}
        for tid in tids:
            b = load_song_embeddings(embedding_path(config.EMBEDDINGS_DIR, audio, tid))
            if b is not None and np.any(b.get("audio_full")):
                audio_vecs[tid] = b["audio_full"]

    gmap = _genre_map()
    ids, A, T, G = [], [], [], []
    for tid in tids:
        if tid in audio_vecs and tid in lyric_vecs:
            ids.append(tid); A.append(audio_vecs[tid]); T.append(lyric_vecs[tid])
            G.append(gmap.get(tid, "unknown"))
    if not ids:
        raise RuntimeError(
            "No songs have BOTH an audio vector and a lyric vector. Run "
            "lyric_encoder.embed_pending() (and MERT extraction) first.")
    logger.info("Loaded LyricLMC PairSet: %d songs (audio=%s d=%d, text d=%d).",
                len(ids), audio, np.stack(A).shape[1], np.stack(T).shape[1])
    return PairSet(ids, np.stack(A), np.stack(T), np.asarray(G),
                   audio_kind=audio, text_kind="lyric_encoder")


# ── synthetic generators (for selftest / plumbing checks, no corpus needed) ──────
def synthetic_pairset(n: int = 300, dim_a: int = 1024, dim_t: int = 384,
                      signal: float = 0.8, latent_k: int = 16, n_groups: int = 4,
                      seed: int = 0) -> PairSet:
    """Different-dim audio/text with a controllable congruence signal in [0,1].

    signal=1 -> perfectly congruent matched pairs (AUC should approach 1);
    signal=0 -> unrelated (AUC ~ 0.5). Used to test LyricLMC and geometry.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, latent_k))                    # shared latent
    Z2 = rng.standard_normal((n, latent_k))                   # text-private latent
    Zt = signal * Z + np.sqrt(max(1e-9, 1 - signal ** 2)) * Z2
    Wa = rng.standard_normal((latent_k, dim_a))
    Wt = rng.standard_normal((latent_k, dim_t))
    A = Z @ Wa + 0.3 * rng.standard_normal((n, dim_a))
    T = Zt @ Wt + 0.3 * rng.standard_normal((n, dim_t))
    G = rng.integers(0, n_groups, size=n).astype(str)
    return PairSet(list(range(n)), A.astype(np.float32), T.astype(np.float32), G,
                   audio_kind="synthetic_audio", text_kind="synthetic_text",
                   meta={"signal": signal})


def synthetic_joint_pairset(n: int = 300, dim: int = 64, signal: float = 0.8,
                            gap: float = 3.0, latent_k: int = 16, n_groups: int = 4,
                            seed: int = 0) -> PairSet:
    """Same-dim audio/text with a simulated MODALITY GAP (a constant text offset).

    Tests the cosine harness and demonstrates that the Step-1 centering fix recovers
    AUC lost to the gap.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, latent_k))
    Z2 = rng.standard_normal((n, latent_k))
    Zt = signal * Z + np.sqrt(max(1e-9, 1 - signal ** 2)) * Z2
    W = rng.standard_normal((latent_k, dim))
    A = Z @ W + 0.3 * rng.standard_normal((n, dim))
    T = Zt @ W + 0.3 * rng.standard_normal((n, dim))
    T = T + gap * rng.standard_normal(dim)                    # inject a constant offset
    G = rng.integers(0, n_groups, size=n).astype(str)
    return PairSet(list(range(n)), A.astype(np.float32), T.astype(np.float32), G,
                   audio_kind="synthetic_audio", text_kind="synthetic_text",
                   meta={"signal": signal, "gap": gap})
