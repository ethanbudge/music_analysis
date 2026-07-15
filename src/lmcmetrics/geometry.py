"""
geometry.py — Step 4: congruence as agreement between two embedding *geometries*.

The idea in one sentence
------------------------
Forget about putting audio and lyrics in the same space. Instead ask: **do songs
that are close in AUDIO space also tend to be close in LYRIC space?** If musical
neighbours are also lyrical neighbours, then music and words are co-varying — they
"say the same thing" as a *structural* property of the corpus. This sidesteps the
modality gap entirely (we never compare an audio vector to a text vector directly),
which is exactly why it is robust.

Two classic tools do this, and they are simpler than they sound:

1. RSA — Representational Similarity Analysis (Kriegeskorte et al., 2008).
   * Build the audio similarity matrix SA (SA[i,j] = how similar songs i,j sound).
   * Build the lyric similarity matrix ST (how similar their lyrics are).
   * Congruence = correlation between the off-diagonal entries of SA and ST.
   One number for the whole corpus. High = music and lyric geometries agree.

2. CKA — Centered Kernel Alignment (Kornblith et al., ICML 2019).
   * A normalised similarity-of-similarity-matrices score in [0, 1].
   * Crucially INVARIANT to rotation and isotropic scaling of either space, so it
     is safe to compare two different modalities with different dimensionalities.
   One number for the whole corpus. 1 = identical geometry (up to rotation/scale).

3. LOCAL RSA (per song) — our per-song congruence score.
   For song i, correlate {how similar i is to every other song in AUDIO} with
   {how similar i is to every other song in LYRICS}. A high value means song i
   lives in a neighbourhood where music and lyrics co-vary — locally congruent.
   This yields ONE scalar per song, so it can become an `lmc_*` column and be
   correlated with the human survey.

   Caveat to state in the write-up: local RSA measures congruence *relative to the
   corpus covariation structure*, not the absolute semantic match of a song's own
   lyrics to its own music (that is what LyricLMC's pair score does). The two are
   complementary; report both.

All functions take plain [N, D] float matrices (audio and lyric can have different
D). Nothing here needs a GPU or torch.
"""

from __future__ import annotations
import logging

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


# ── similarity / dissimilarity matrices ──────────────────────────────────────────
def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Full [N, N] cosine-similarity matrix for rows of X."""
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(n > 1e-12, n, 1.0)
    return Xn @ Xn.T


def dissimilarity_matrix(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """[N, N] representational dissimilarity matrix (RDM)."""
    if metric == "cosine":
        return 1.0 - cosine_similarity_matrix(X)
    if metric == "euclidean":
        X = np.asarray(X, dtype=np.float64)
        sq = np.sum(X ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        return np.sqrt(np.maximum(d2, 0.0))
    raise ValueError(f"unknown metric {metric!r}")


def _upper_tri(M: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle of a square matrix (the unique pairs)."""
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


# ── RSA (global) ──────────────────────────────────────────────────────────────────
def rsa(audio: np.ndarray, lyric: np.ndarray, metric: str = "cosine",
        method: str = "spearman") -> float:
    """Global RSA: correlation between the audio RDM and the lyric RDM.

    Returns a scalar in [-1, 1] (Spearman by default — rank correlation is the
    standard RSA choice because only the *ordering* of dissimilarities matters).
    """
    da = _upper_tri(dissimilarity_matrix(audio, metric))
    dl = _upper_tri(dissimilarity_matrix(lyric, metric))
    if method == "spearman":
        da, dl = rankdata(da), rankdata(dl)
    # Pearson on (optionally ranked) vectors.
    da = da - da.mean()
    dl = dl - dl.mean()
    denom = np.linalg.norm(da) * np.linalg.norm(dl)
    return float(da @ dl / denom) if denom > 0 else 0.0


# ── CKA (global) ──────────────────────────────────────────────────────────────────
def linear_cka(audio: np.ndarray, lyric: np.ndarray) -> float:
    """Linear CKA in [0, 1]. Invariant to rotation and isotropic scaling.

    CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)   (columns centered).
    Computed in the efficient feature-space form (no N x N matrix needed).
    """
    X = np.asarray(audio, dtype=np.float64)
    Y = np.asarray(lyric, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.sum((X.T @ Y) ** 2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def _rbf_gram(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (X @ X.T), 0.0)
    if sigma is None:                       # median heuristic
        med = np.median(d2[np.triu_indices_from(d2, k=1)])
        sigma = np.sqrt(med / 2.0) if med > 0 else 1.0
    return np.exp(-d2 / (2.0 * sigma ** 2))


def kernel_cka(audio: np.ndarray, lyric: np.ndarray, sigma: float | None = None) -> float:
    """RBF-kernel CKA in [0, 1] — captures nonlinear geometry agreement.

    Heavier (builds N x N Gram matrices) but a good robustness check on linear CKA.
    """
    K = _rbf_gram(audio, sigma)
    L = _rbf_gram(lyric, sigma)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Lc = H @ K @ H, H @ L @ H
    hsic_kl = np.sum(Kc * Lc)
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(hsic_kl / denom) if denom > 0 else 0.0


# ── Local RSA (per-song congruence) ──────────────────────────────────────────────
def local_rsa(audio: np.ndarray, lyric: np.ndarray, metric: str = "cosine",
              k: int | None = None) -> np.ndarray:
    """Per-song congruence: for each song i, rank-correlate its similarity profile
    in audio space with its similarity profile in lyric space.

    Parameters
    ----------
    k : if None, use every other song (robust, recommended). If an int, restrict to
        song i's k nearest AUDIO neighbours (a stricter, more local notion).

    Returns
    -------
    scores : np.ndarray [N]  — Spearman rho per song, in [-1, 1]. Higher = song i
             sits where music and lyrics co-vary (more locally congruent).
    """
    SA = cosine_similarity_matrix(audio) if metric == "cosine" else -dissimilarity_matrix(audio, metric)
    SL = cosine_similarity_matrix(lyric) if metric == "cosine" else -dissimilarity_matrix(lyric, metric)
    N = SA.shape[0]
    scores = np.full(N, np.nan)
    for i in range(N):
        others = np.arange(N) != i
        a = SA[i, others]
        l = SL[i, others]
        if k is not None and k < a.size:
            # Restrict to the k most audio-similar neighbours of i.
            idx = np.argsort(-a)[:k]
            a, l = a[idx], l[idx]
        if a.size < 3 or np.allclose(a, a[0]) or np.allclose(l, l[0]):
            continue
        ar, lr = rankdata(a), rankdata(l)
        ar = ar - ar.mean()
        lr = lr - lr.mean()
        denom = np.linalg.norm(ar) * np.linalg.norm(lr)
        scores[i] = (ar @ lr / denom) if denom > 0 else np.nan
    return scores


# ── convenience: everything at once ──────────────────────────────────────────────
def geometry_report(audio: np.ndarray, lyric: np.ndarray, metric: str = "cosine",
                    k: int | None = None, with_kernel: bool = False) -> dict:
    """Compute global CKA + RSA and per-song local RSA in one call."""
    out = {
        "n": int(np.asarray(audio).shape[0]),
        "linear_cka": linear_cka(audio, lyric),
        "rsa_spearman": rsa(audio, lyric, metric=metric, method="spearman"),
    }
    if with_kernel:
        out["kernel_cka"] = kernel_cka(audio, lyric)
    ls = local_rsa(audio, lyric, metric=metric, k=k)
    out["local_rsa"] = ls
    out["local_rsa_mean"] = float(np.nanmean(ls))
    return out


# ── optional: Gromov-Wasserstein (needs POT: `pip install pot`) ───────────────────
def gromov_wasserstein(audio: np.ndarray, lyric: np.ndarray, metric: str = "cosine"):
    """Gromov-Wasserstein alignment between the two geometries (optional).

    Returns (gw_distance, coupling). The diagonal mass of the coupling is a per-song
    "does song i map to itself under the optimal cross-modal transport" signal.
    Requires the POT library; raises a clear error if it is not installed.
    """
    try:
        import ot
    except Exception as e:                                     # noqa: BLE001
        raise ImportError(
            "Gromov-Wasserstein needs the POT library. Install with `pip install pot` "
            "to enable this optional metric."
        ) from e
    Da = dissimilarity_matrix(audio, metric)
    Dl = dissimilarity_matrix(lyric, metric)
    n = Da.shape[0]
    p = np.ones(n) / n
    q = np.ones(n) / n
    coupling, log = ot.gromov.gromov_wasserstein(
        Da, Dl, p, q, "square_loss", log=True, verbose=False)
    return float(log["gw_dist"]), coupling
