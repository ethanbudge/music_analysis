"""
lmcmetrics — alternative Lyric-Music Congruence (LMC) metrics and their evaluation.

A *separate, self-contained* package that sits alongside the observational arm
(`lmc`) and the generative arm (`lmcgen`). It does NOT modify or delete anything in
those packages — it reads their cached artifacts (the MERT / MuLan / CLAP embeddings
and the project SQLite DB) and adds new ways to measure congruence, plus a common
yardstick to compare all of them on equal footing.

What lives here (see docs/congruence_metrics_review.md for the theory):

  centering.py   Step 1 — remove the audio/text "modality gap" before any cosine.
  evaluate.py    Step 2 — matched-vs-mismatched retrieval harness. The yardstick:
                 score ANY metric by how well it ranks a song's TRUE lyrics above
                 impostor lyrics. Fully automatic, whole corpus, no humans needed.
  lyriclmc.py    Step 3 — a small bespoke contrastive model ("LyricLMC") that learns
                 a lyric<->audio space from YOUR matched pairs. Trains only two
                 projection heads on CACHED embeddings (never raw audio).
  geometry.py    Step 4 — CKA & RSA: compare the *geometry* of the audio embedding
                 space with the *geometry* of the lyric embedding space (global +
                 a per-song "local RSA" congruence score).
  lyric_encoder  The new lyric text tower (a sentence encoder), cached per song.
  scorers.py     A common Scorer interface so every metric plugs into evaluate.py.
  data.py        Load matched (audio, lyric) pairs from the existing caches; also a
                 synthetic generator so you can test the plumbing with no corpus.
  selftest.py    Runs the whole pipeline on synthetic data (correctness check).

Nothing here writes into `data/embeddings/` or `project.db` tables owned by `lmc`.
New artifacts go under `data/lmcmetrics/` and `results/lmcmetrics/` (both gitignored),
and every LyricLMC training run is saved to its own timestamped folder so you can
keep and compare different iterations.
"""

from __future__ import annotations

__all__ = [
    "config",
    "data",
    "centering",
    "evaluate",
    "geometry",
    "lyric_encoder",
    "lyriclmc",
    "scorers",
]
