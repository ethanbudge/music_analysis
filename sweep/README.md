# sweep — the model × prompt × structure LMC battery

Expands the observational analysis to the **full grid** validated in the POC
(`validation/`), then runs the **same v4 Stan models** on it. It reuses the
existing corpus, audio, MERT controls, chorus flags, and native LRCLIB line
timestamps — it only adds the new embeddings on top.

## The grid → 60 Stan fits

| axis | values | n |
|---|---|---|
| **models** | MuQ-MuLan, LAION-CLAP, Microsoft CLAP, CLaMP 3 | 4 |
| **prompts** | raw · "a song[ segment] that contains the lyrics …" · "… representing the idea of …" | 3 |
| **structures** | song-wide, line ±1 s, line ±5 s, line ±10 s (track models) + chorus/verse (segment model) | 5 |

4 × 3 = **12 embeddings**, each fit at 5 structures = **60 fits**. No curvature models.
`LMC = cosine(audio, prompt(text))` inside each model's own space; line methods are the
**mean** cosine over a song's lines.

## Pipeline (three steps)

```
1.  sweep/embeddings_sweep.ipynb   compute the 72 LMC columns → results/master_results_sweep.csv   (Python, lmc env)
2.  sweep/R/run_models_sweep.R     fit the 60 models → sweep/output/                                (R / RStudio)
3.  sweep/lmc_report_sweep.qmd     model comparison + posterior checks                              (Quarto / RStudio)
```

### 1 — Embeddings (notebook)

Run `embeddings_sweep.ipynb` top to bottom in the **`lmc` conda env** (`pip install
msclap`; set the CLaMP 3 env vars in cell 1, as in the validation notebook).

- **Resumable**: a `(song, model)` already recorded in the `sweep_progress` table is
  skipped. Stop/restart freely; run one model at a time.
- **Run CLaMP 3 last / overnight** — it is far slower than the others (subprocess +
  MERT-95M per chunk over ~1,000 songs × line windows). Everything else is quick.
- Output: `results/master_results_sweep.csv` — the existing master's outcome / genre /
  `mert_pc*` controls **plus** 4×3×6 = 72 LMC columns (`<model>_<prompt>_<method>`).

Offline plumbing check (no corpus/models): `cd sweep && python -m sweep_lib.selftest`.

### 2 — Fit (R)

```bash
Rscript sweep/R/run_models_sweep.R                 # all, N=all, controls=mert
Rscript sweep/R/run_models_sweep.R 500 1 mert      # N, seed, controls (for a quick subset)
```

- **4 chains × (1000 warmup + 1000 sampling)**, MERT PCA controls.
- **One GLOBAL complete-case corpus**: a song must have all 60 measures present, so
  every LOO object is mutually comparable. Heavy CLaMP 3 / MS-CLAP missingness shrinks N
  for *all* fits — check the notebook's `build_master.missing_report()` and the
  `global_complete_case` count.
- The **Stan models are unchanged** (`stan/model_track_v4.stan`,
  `stan/model_segment_v4.stan`). `run_models_sweep.R` *sources* `stan/run_models.R` and
  reuses its data builders + `fit_one`; only the battery, master CSV, and output dir
  differ.
- Writes fits `track_<emb>_<measure>.rds` / `segment_<emb>.rds` (+ `.labels.rds`) and
  the LOO table `sweep/output/loo_all_sweep.csv` to `sweep/output/`.

### 3 — Report (Quarto)

Render `lmc_report_sweep.qmd` (RStudio, or `quarto render`). It gives the 60-fit LOO
comparison (heatmap + ranked table + marginal "which model/prompt/structure wins"),
LMC effect sizes across the grid, a best-fit spotlight, and posterior predictive checks.

## Files

```
sweep/
  embeddings_sweep.ipynb        Step 1 — compute the LMC grid (resumable)
  sweep_lib/
    config.py     the grid, paths, column naming (must match sweep/R)
    compute.py    resumable per-(song,model) embedding → cosine → lmc_sweep table
    build_master.py  lmc_sweep + existing master → results/master_results_sweep.csv
    selftest.py   offline mock test of the compute + master plumbing
  R/
    run_models_sweep.R      Step 2 — sources stan/run_models.R, fits the 60-model battery
    report_helpers_sweep.R  sources analysis/report_helpers.R; overrides tag parsing for {model}_{prompt}
  lmc_report_sweep.qmd      Step 3 — model comparison + posterior checks
  output/                   Stan fits + LOO tables (gitignored)
```

## Reuse map (nothing duplicated)

- **corpus / audio / DB / chorus / slicing** → `src/lmc` (`db`, `chorus`, `utils.parse_lrc`,
  `embeddings._slice`)
- **the 4 model wrappers + 3 prompt templates** → `validation/lmcval` (`models`, `config.format_prompt`)
- **Stan data builders + `fit_one` + Stan models** → `stan/run_models.R`, `stan/model_{track,segment}_v4.stan`
- **fit extraction + plots + PPC** → `analysis/report_helpers.R`

New artifacts (`sweep/output/`, `results/master_results_sweep.csv`, the `lmc_sweep` /
`sweep_progress` DB tables) are additive — the observational pipeline is untouched.
```
