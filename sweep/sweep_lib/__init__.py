"""
sweep_lib — compute the full model × prompt × structure LMC grid on the existing corpus.

This package powers `sweep/embeddings_sweep.ipynb`. It reuses the observational
pipeline (`src/lmc`) for the corpus/DB/audio/chorus/slicing and the validation
harness (`validation/lmcval`) for the four model wrappers and the three prompt
templates, then computes one LMC scalar per (song, model, prompt, method) and
assembles a `master_results_sweep.csv` for the Stan battery in `sweep/R/`.

The grid:
  models   : mulan, clap, msclap, clamp3           (4)
  prompts  : raw, contains, idea                    (3)   -> 12 "embeddings"
  methods  : song, seg_chorus, seg_nonchorus,
             line_buf1, line_buf5, line_buf10        (6 scalars/embedding)

Which feed 5 Stan fits per embedding (song + 3 line windows as track models;
chorus/verse as one segment model) = 60 fits. No curvature models.

Everything is resumable: a (track_id, model) pair already recorded in the
`sweep_progress` table is skipped, so you can stop and restart freely, and run one
model at a time (useful because CLaMP 3 is much slower than the rest).
"""

from __future__ import annotations

__all__ = ["config", "compute", "build_master"]
