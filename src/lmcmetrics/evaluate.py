"""
evaluate.py — Step 2: the matched-vs-mismatched retrieval harness (the YARDSTICK).

This is the most important piece of infrastructure in the package. It lets you
score ANY congruence metric — the current MuLan cosine, CLAP, a centered variant,
LyricLMC, a geometry score — on the SAME objective, automatic, whole-corpus test:

    A good congruence metric should score a song's TRUE lyrics higher than
    someone else's ("impostor") lyrics paired to the same audio.

Given a score matrix S where S[i, j] = congruence(audio_i, text_j), the diagonal
S[i, i] are the true pairs and the off-diagonal are impostors. We report:

  auc          P(true pair scores above a random impostor). 0.5 = chance,
               1.0 = perfect. This is the headline number — a metric that beats
               MuLan here is objectively better at detecting congruence.
  recall@k     how often the true partner is in the top-k retrieved.
  median_rank  median position of the true partner (1 = always first).
  mrr          mean reciprocal rank.

Directions: 'a2t' (given the audio, rank candidate lyrics) and 't2a' (given the
lyrics, rank candidate audio). We report both and their mean.

Hard mode (optional): pass `groups` (e.g. genre) and `restrict='same'` to force the
impostors to come from the SAME genre. That controls for topic/genre and isolates
finer congruence — a much stricter and more meaningful test.

Bonus per-song output: `per_song_margin` returns, for each song, how far its true
score sits above its own impostor distribution (a z-score). That is itself a
*calibrated* per-song congruence value you can correlate with the human survey.

Nothing here needs a GPU. For N up to a few thousand the full N x N matrix is fine;
set `impostor_cap` to subsample impostors for very large corpora.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


# ── build a cosine score matrix (with optional Step-1 centering) ─────────────────
def cosine_score_matrix(audio: np.ndarray, text: np.ndarray, centerer=None) -> np.ndarray:
    """S[i,j] = cosine(audio_i, text_j). audio and text must share dimensionality.

    If `centerer` (a fitted centering.Centerer) is given, the modality-gap fix is
    applied first. This is how Step 1 plugs into the evaluation.
    """
    A = np.asarray(audio, dtype=np.float64)
    T = np.asarray(text, dtype=np.float64)
    if centerer is not None:
        A = centerer.transform_audio(A)
        T = centerer.transform_text(T)
    if A.shape[1] != T.shape[1]:
        raise ValueError(
            f"cosine needs matching dims; got audio {A.shape[1]} vs text {T.shape[1]}. "
            "Use a projection scorer (e.g. LyricLMC) for cross-dim representations."
        )
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Tn = T / np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-12, None)
    return An @ Tn.T


# ── core metric ───────────────────────────────────────────────────────────────────
def retrieval_metrics(S: np.ndarray, groups=None, restrict: str | None = None,
                      recall_at=(1, 5, 10), impostor_cap: int | None = None,
                      seed: int = 42) -> dict:
    """Compute matched-vs-mismatched retrieval metrics from a score matrix S.

    S[i, j] = congruence(audio_i, text_j); higher = more congruent.
    restrict: None (all impostors) | 'same' (same-group only) | 'diff' (cross-group).
    """
    S = np.asarray(S, dtype=np.float64)
    N = S.shape[0]
    assert S.shape == (N, N), "S must be square (N songs x N candidate lyrics)."
    rng = np.random.default_rng(seed)
    groups = None if groups is None else np.asarray(groups)

    out = {}
    per_song_rank = {}
    for direction in ("a2t", "t2a"):
        M = S if direction == "a2t" else S.T   # M[i,:]: candidates scored for query i
        ranks = np.full(N, np.nan)
        aucs = np.full(N, np.nan)
        for i in range(N):
            cand = np.ones(N, dtype=bool)
            if groups is not None and restrict is not None:
                same = groups == groups[i]
                cand = same if restrict == "same" else ~same
                cand[i] = True                              # always keep the true item
            if impostor_cap is not None:
                pool = np.flatnonzero(cand & (np.arange(N) != i))
                if pool.size > impostor_cap:
                    keep = rng.choice(pool, size=impostor_cap, replace=False)
                    mask = np.zeros(N, dtype=bool)
                    mask[keep] = True
                    mask[i] = True
                    cand = mask
            idx = np.flatnonzero(cand)
            if idx.size < 2:
                continue
            row = M[i, idx]
            pos = int(np.flatnonzero(idx == i)[0])          # where the true item sits
            r = rankdata(-row, method="average")[pos]        # 1 = best; ties averaged
            ranks[i] = r
            aucs[i] = (idx.size - r) / (idx.size - 1)
        per_song_rank[direction] = ranks
        tag = "" if direction == "a2t" else "_t2a"
        out[f"auc{tag}"] = float(np.nanmean(aucs))
        out[f"median_rank{tag}"] = float(np.nanmedian(ranks))
        out[f"mrr{tag}"] = float(np.nanmean(1.0 / ranks))
        for k in recall_at:
            out[f"recall@{k}{tag}"] = float(np.nanmean(ranks <= k))

    out["auc_mean"] = 0.5 * (out["auc"] + out["auc_t2a"])
    out["n"] = int(N)
    # Calibrated per-song congruence: z of the true score vs this row's impostors.
    diag = np.diag(S)
    off_mean = (S.sum(axis=1) - diag) / (N - 1)
    off_var = (np.sum(S ** 2, axis=1) - diag ** 2) / (N - 1) - off_mean ** 2
    off_std = np.sqrt(np.clip(off_var, 1e-12, None))
    out["per_song_margin"] = (diag - off_mean) / off_std
    out["_ranks_a2t"] = per_song_rank["a2t"]
    return out


def summarize(metrics: dict) -> dict:
    """Drop the bulky per-song arrays for a compact, tabulatable summary."""
    return {k: v for k, v in metrics.items() if not k.startswith(("per_song", "_"))}


# ── compare many scorers on one PairSet ──────────────────────────────────────────
def compare_scorers(pairset, scorers: dict, groups=None, restrict: str | None = None,
                    recall_at=(1, 5, 10)) -> pd.DataFrame:
    """Run every scorer over one PairSet and return a ranked comparison table.

    `scorers` maps name -> object with `.score_matrix(pairset) -> [N,N]`
    (see scorers.py). `groups` defaults to the PairSet's own groups if present.
    """
    if groups is None:
        groups = getattr(pairset, "groups", None)
    rows = []
    for name, scorer in scorers.items():
        S = scorer.score_matrix(pairset)
        m = summarize(retrieval_metrics(S, groups=groups, restrict=restrict,
                                        recall_at=recall_at))
        m["scorer"] = name
        rows.append(m)
        logger.info("  %-24s auc=%.3f  R@1=%.3f  median_rank=%.1f",
                    name, m["auc_mean"], m.get("recall@1", float("nan")),
                    m.get("median_rank", float("nan")))
    df = pd.DataFrame(rows).set_index("scorer").sort_values("auc_mean", ascending=False)
    front = ["auc_mean", "auc", "auc_t2a", "recall@1", "recall@5", "median_rank", "mrr", "n"]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    return df[cols]
