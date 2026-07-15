"""
scorers.py — a common interface so every congruence metric plugs into evaluate.py.

A Scorer turns a PairSet into an N x N score matrix S where S[i, j] is the
congruence of audio_i with lyrics_j. evaluate.compare_scorers() then ranks them all
on the identical matched-vs-mismatched test.

Built-in scorers
----------------
CosineScorer(center=...)   the current paradigm: cosine in a shared space (MuLan /
                           CLAP). center=None (raw, = your current LMC), 'mean'
                           (Step-1 modality-gap fix), or 'std'. Requires the PairSet's
                           audio and text to share dimensionality.
LyricLMCScorer(model)      Step-3 learned space: project both sides, then cosine.
                           Works for different-dim audio/text.

You can add your own (e.g. a VA-congruence scorer for Family B later): implement
`.score_matrix(pairset) -> np.ndarray [N, N]` and drop it into the dict.
"""

from __future__ import annotations

import numpy as np

from .centering import Centerer
from .evaluate import cosine_score_matrix


class CosineScorer:
    """Cosine in a shared space, optionally with the Step-1 centering fix.

    center: None -> raw cosine (reproduces the current MuLan/CLAP LMC exactly);
            'mean' -> subtract per-modality corpus mean first (modality-gap fix);
            'std'  -> mean-center then scale by per-dim std.
    """

    def __init__(self, center: str | None = None, l2: bool = False, name: str | None = None):
        self.center = center
        self.l2 = l2
        self.name = name or ("cosine" if center is None else f"cosine_{center}")

    def score_matrix(self, pairset) -> np.ndarray:
        centerer = None
        if self.center is not None:
            centerer = Centerer(mode=self.center, l2=self.l2).fit(pairset.audio, pairset.text)
        return cosine_score_matrix(pairset.audio, pairset.text, centerer=centerer)


class LyricLMCScorer:
    """Score with a trained LyricLMC run (projects both towers, then cosine)."""

    def __init__(self, model, name: str = "lyriclmc"):
        self.model = model            # a lyriclmc.LyricLMC handle
        self.name = name

    def score_matrix(self, pairset) -> np.ndarray:
        return self.model.score_matrix_from_reps(pairset.audio, pairset.text)


class PrecomputedScorer:
    """Wrap an already-computed N x N score matrix (e.g. from an external metric)."""

    def __init__(self, S: np.ndarray, name: str = "precomputed"):
        self.S = np.asarray(S)
        self.name = name

    def score_matrix(self, pairset) -> np.ndarray:
        return self.S
