"""
centering.py — Step 1: remove the "modality gap" before computing a cosine.

Why this exists
---------------
In a contrastive dual-encoder (MuLan, CLAP), audio embeddings and text embeddings
do NOT sit on top of each other — they occupy two separate "cones" with a constant
offset between them (the *modality gap*; Liang et al., NeurIPS 2022). A raw
`cos(audio, text)` therefore mixes the thing you care about (do these two agree?)
with an artefact (where the two cones happen to sit).

The cheapest, standard mitigation is to **mean-center each modality separately**
using the corpus mean, then take the cosine. This subtracts the constant offset so
the cosine reflects agreement, not geometry. Optionally we also L2-normalise or
standardise.

This is a no-training, deterministic transform. Fit it once on your corpus and
apply it to any vector (including new/generated songs).

Usage
-----
    c = Centerer().fit(audio_matrix, text_matrix)   # [N,D] each
    a2 = c.transform_audio(audio_matrix)
    t2 = c.transform_text(text_matrix)
    # now cosine(a2[i], t2[i]) is modality-gap-corrected.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[None, :] if x.ndim == 1 else x


@dataclass
class Centerer:
    """Per-modality mean-centering (optionally standardisation) for the gap fix.

    mode:
      'mean'  — subtract the per-modality corpus mean (default; removes the gap).
      'std'   — subtract mean and divide by per-dimension std (whitening-lite).
      'none'  — identity (useful as an A/B baseline in the harness).
    l2:  if True, L2-normalise each vector AFTER centering (so downstream code can
         use a plain dot product). Cosine is scale-invariant either way; this just
         makes the vectors unit-length.
    """

    mode: str = "mean"
    l2: bool = False
    audio_mean_: np.ndarray = field(default=None, repr=False)
    audio_std_:  np.ndarray = field(default=None, repr=False)
    text_mean_:  np.ndarray = field(default=None, repr=False)
    text_std_:   np.ndarray = field(default=None, repr=False)

    def fit(self, audio: np.ndarray, text: np.ndarray) -> "Centerer":
        A, T = _as2d(audio), _as2d(text)
        self.audio_mean_ = A.mean(axis=0)
        self.text_mean_ = T.mean(axis=0)
        # Guard against zero-variance dims (avoid divide-by-zero in 'std').
        self.audio_std_ = np.where(A.std(axis=0) > 1e-8, A.std(axis=0), 1.0)
        self.text_std_ = np.where(T.std(axis=0) > 1e-8, T.std(axis=0), 1.0)
        return self

    def _apply(self, x, mean, std):
        X = _as2d(x)
        if self.mode == "none":
            out = X.copy()
        elif self.mode == "mean":
            out = X - mean
        elif self.mode == "std":
            out = (X - mean) / std
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.l2:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(n > 1e-12, n, 1.0)
        return out

    def transform_audio(self, audio: np.ndarray) -> np.ndarray:
        return self._apply(audio, self.audio_mean_, self.audio_std_)

    def transform_text(self, text: np.ndarray) -> np.ndarray:
        return self._apply(text, self.text_mean_, self.text_std_)
