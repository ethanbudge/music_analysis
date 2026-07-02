# Archived Stan models (superseded)

These are kept for provenance only — they are **not** used by `run_models.R`,
`evaluate_models.R`, or the report (all of which target the **v4** family in
`stan/`).

| File | Family | Why superseded |
|---|---|---|
| `model_track.stan`, `model_segment.stan`, `model_timeline.stan` | **v2** (original) | Singleton artist random effect + mis-calibrated `phi` prior → E-BFMI < 0.3, high Pareto-k (see `../MODEL_NOTES.md`). |
| `model_track_v3.stan`, `model_segment_v3.stan`, `model_timeline_v3.stan` | **v3** (reparameterized) | Fixed the diagnostics (dropped artist, NCP genre, genre-varying precision). Replaced by **v4**, which adds a generic control matrix (mood/MERT toggle) and functional trajectory models, and folds the timeline summaries into scalar-on-function regression. |

The **v4** models in `stan/` are the canonical family.
