"""
lmcval — a self-contained spot-test harness for comparing LMC metrics.

This is a VALIDATION sandbox, separate from the main pipeline (`src/lmc`), the
metrics package (`src/lmcmetrics`), and the generative arm (`src/lmcgen`). It reuses
as much of the original pipeline as possible (audio download scoring, LRC parsing,
chorus detection, the MuLan/CLAP embedders, audio slicing) and adds:

  * two extra embedding models — Microsoft CLAP (msclap) and CLaMP 3,
  * three text-prompt templates for turning lyrics into a text query,
  * three segmentation levels — song-wide, segment-wide, line-by-line,

then scores lyric-music congruence as cosine(audio, text) for every
model × prompt × unit and writes three tidy CSVs plus legible ranked summaries.

The whole thing is driven from `validation/lmc_validation.ipynb`.
See `validation/README.md` for setup and interpretation.
"""

from __future__ import annotations

__all__ = ["config", "acquire", "models", "units", "run"]
