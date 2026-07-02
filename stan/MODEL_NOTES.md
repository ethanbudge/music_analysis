> **Update (v4 family).** The models in `stan/` are now the **v4** family. v4 keeps
> every v3 fix below and adds: (1) a **generic control matrix** `X[N,K]` so the
> control set is a runtime toggle (`controls = mood | mert | both`, default MERT —
> see `run_models.R`); (2) **functional (scalar-on-function) trajectory models**
> (`model_curve_v4`, `model_segment_curve_v4`) that replace the old two-stage
> timeline summaries with a penalized-spline β(t); and (3) an experimental
> one-stage line-level model (`model_line_curve_v4`). v2/v3 sources are in
> `stan/archive/`. The diagnosis below (written for v3) still explains *why* the
> structure is what it is.
>
> **Curvature update.** The functional models originally used a *centred* RW2
> P-spline, which funnelled (`sigma_b`↔spline coefficients) → 310 divergences /
> E-BFMI 0.37 / R̂ 1.05 on `curve`. They now use a **non-centred RW2** (level +
> slope + standardised innovations), which fixed it (≈0–9 divergences, E-BFMI
> 0.7+). Two **genre-varying** curvature models were added so β(t) can differ by
> genre (partially pooled): `model_curve_poly_v4` (hierarchical orthogonal-
> polynomial / "quadratic by genre" — robust, `POLY_DEGREE=2`) and
> `model_curve_hier_v4` (hierarchical spline — flexible). Both emit `beta_t_genre`
> for per-genre β_g(t) plots. The trajectory LOO set is now
> segment / curve / segment+curve / polycurve / hiercurve.

# Modeling notes: why the LMC models mis-fit, and how to reparameterize

*Deep dive prompted by **high Pareto-k everywhere** and **E-BFMI < 0.3 on every
chain**. These two diagnostics, together, are not "tune the sampler" problems —
they are telling you the **model geometry and likelihood are mis-specified for
this data**. Below is what the current corpus actually looks like, why each
diagnostic is firing, and a prioritized set of construction options.*

All numbers below are from the current `results/master_results.csv`
(N = 550 complete songs, MuLan measures only — see §0).

---

## 0. First, a blocker for the CLAP comparison

`master_results.csv` **has no `clap_*` columns yet** — only `mulan_*`. The runner
now supports `clap`/`both`, but it will warn and skip CLAP until you compute the
embeddings and rebuild the master table:

```python
embeddings.embed_pending("clap")
alignment.compute_pending()
combine.build_master()
```

Once those columns exist, `Rscript stan/run_models.R both` fits MuLan **and** CLAP
on **one shared complete-case corpus** (the songs that have every measure for both
embeddings). That shared corpus is the only way `loo_compare` across embeddings is
valid — LOO differences are meaningless if the two models saw different rows.

---

## 1. What the data actually looks like

| Quantity | Value | Implication |
|---|---|---|
| N (complete cases) | 550 | small for a 9-genre × varying-slope hierarchy |
| `spotify_popularity/100` | mean **0.41**, sd **0.22** | nicely interior — **not** zero-inflated |
| exact 0 / exact 1 | **7** / **0** | boundary mass is negligible; plain Beta is fine *if* phi is right |
| **artists** | **510 for 550 songs; 94% appear once**, max 4 | a near **per-observation** random effect |
| **genres** | 9, but **`unknown` = 38%** (209/550) | one "group" is a junk-drawer |
| **orientation** | narrative 43%, production 19%, **`unknown` 38%** | 38% is currently coded `0.5` — a fiction |
| data-implied global `phi` (method-of-moments) | **≈ 4** | prior is `gamma(4, 0.1)` → **mean 40, sd 20** |

Two things jump out: **the artist effect is almost one parameter per row**, and
**the `phi` prior is ~10× too tight**. Those are the headline causes.

---

## 2. Why E-BFMI < 0.3 (energy diagnostic)

Low E-BFMI means the momentum resampling in HMC can't move between energy levels —
the classic signature of a **funnel** or a poorly identified scale parameter
(Betancourt 2016, *Diagnosing Biased Inference with Divergences*). You have at
least three funnels feeding it:

1. **Artist intercepts on singleton groups (the big one).** `alpha_artist` is 510
   parameters informed, for 94% of artists, by a *single* observation. With one
   data point, an artist's intercept and the residual Beta dispersion (`phi`) are
   nearly unidentified — they trade off along a ridge, and `sigma_artist` sits in a
   funnel with the 510 `z_artist`. In a broad random sample **artist ≈ song**, so
   this term is essentially an observation-level random effect masquerading as a
   grouping factor. It contributes almost no real pooling and a lot of bad geometry.

2. **Centered genre intercept.** Every other hierarchical term is non-centered, but
   `alpha_genre ~ normal(0, sigma_genre)` (in all three models) is **centered**.
   With only 9 groups — one of them the 38% `unknown` bin — `sigma_genre` is weakly
   identified and the centered form funnels. This is a one-line fix (NCP).

3. **`phi` prior fighting the data.** `gamma(4, 0.1)` puts effectively no mass below
   ~15, but the data want `phi ≈ 4`. The sampler is dragged into the prior's tail,
   where the geometry is bad and the energy distribution is skewed. Re-centering the
   prior alone will noticeably help E-BFMI.

E-BFMI will **not** improve by raising `adapt_delta` or `max_treedepth` or running
longer — those address divergences/step size, not the energy mismatch. You have to
change the model.

---

## 3. Why Pareto-k is high almost everywhere

PSIS-LOO approximates leaving out observation *i* by importance-weighting the full
posterior. The approximation breaks (k > 0.7) when removing one point changes the
posterior a lot. Two reasons here, both expected:

1. **Parameter-per-observation ⇒ guaranteed high k.** This is a documented failure
   mode (Vehtari, Gelman & Gabry 2017; the **loo** FAQ explicitly warns about it):
   *"if you have one or more parameters per observation, the LOO approximation can
   fail."* Leaving out a singleton artist's only song means that artist's intercept
   reverts to the prior — a large posterior change — so `k_i` is large for a big
   fraction of rows. **The bad Pareto-k is largely a *symptom* of the artist term**,
   not (only) of likelihood misfit. Remove that term and most k's drop.

2. **An over-precise Beta makes ordinary points look like outliers.** With `phi`
   pulled up toward ~40, the fitted Beta is ~10× too concentrated, so genuinely
   typical songs land in its tails and become individually influential. Fixing the
   `phi` prior (and letting `phi` vary — §4.2) removes much of the remaining tail.

A practical note for the comparison itself: when `k_gt_0.7 > 0`, the reported
`elpd_loo` is unreliable. Until the reparameterization lands, treat LOO rankings as
provisional, and use `evaluate_models.R`'s `k_gt_0.7` column to see which fits you
can trust. For a few bad k's, `loo(..., moment_match = TRUE)` or `reloo = TRUE`
rescues them; if a third of rows are bad, fix the model instead.

---

## 4. Construction options (prioritized)

### 4.1 Remove — or threshold — the artist level **(highest impact)**
In a broad LRCLIB sample, artist is not a usable grouping factor. Options, best
first:

- **Drop `alpha_artist` entirely.** Cleanest. Its variance is absorbed by `phi`
  (and, better, by a `phi` submodel). This single change should fix most of both
  diagnostics. *Recommended default.*
- **Threshold:** pool only artists with ≥ 3 songs into the hierarchy and assign the
  rest a shared "singleton" intercept (handled in R when building `artist_id`).
  Keeps genuine repeat-artist signal without the singleton funnel.
- **Swap in `album`** (often several tracks share an album) if you want *some*
  nesting — but check its singleton rate the same way first.

### 4.2 Model the precision `phi` (don't leave it constant or mis-priored)
Two sub-steps:

- **Re-calibrate the prior** to the data scale, e.g. `phi ~ lognormal(log(6), 0.6)`
  or `gamma(2, 0.3)` — mass over ~3–20 instead of ~40. Do a **prior predictive
  check** to confirm simulated `y` spans (0,1) with realistic spread.
- **Let `phi` depend on covariates** (a *precision submodel*, standard in beta
  regression — Ferrari & Cribari-Neto 2004; `brms`/`betareg` call this the `phi ~`
  part). Popularity is heteroscedastic — variance differs by genre and across the
  popularity range. Modeling `log(phi) = φ0 + genre-varying term (+ maybe LMC)`
  lets typical points stop looking like outliers, which is the *direct* cure for
  influential-point Pareto-k. This is the highest-leverage modeling upgrade after
  removing the artist term.

### 4.3 Non-center every hierarchical term, and right-size the genre prior
- Non-center `alpha_genre` (matches the slope terms already using NCP).
- With 9 groups, `sigma_genre` is estimable but weakly — use a tighter half-normal
  (e.g. `normal(0, 0.3)`) and consider a Student-t / hierarchical-shrinkage prior on
  the genre slopes if you keep varying slopes. With this few groups, **genre as
  fixed effects** (a plain `vector[N_genre]` with a weak normal prior, no `sigma`)
  is a legitimate, better-identified alternative — you lose partial pooling but gain
  clean geometry, and 9 levels don't need shrinkage.

### 4.4 Stop coding missingness as data
- `genre == "unknown"` (38%) and `orientation == 0.5` (38%) are **not measurements**;
  they're "we don't know." Putting `unknown` at the midpoint biases the moderation
  estimates toward zero and contaminates the genre hierarchy with a huge junk group.
  Better: (a) add an explicit `orientation_known` indicator and only apply
  `gamma_*` where known; (b) keep `unknown` genre as its own clearly-labeled
  category but **don't** let it drive `sigma_genre`; or (c) for a first clean pass,
  fit on the labeled subset and report the unknowns separately.

### 4.5 Tame the collinear controls (lower impact)
Seven librosa mood/acoustic proxies are mutually correlated, creating ridges in the
posterior. Either regularize them (a tighter shared prior, or a regularized-
horseshoe if you want selection) or reduce them to 2–3 principal components used as
controls. Won't fix E-BFMI, but tightens everything else.

### 4.6 Likelihood alternatives (only if needed after the above)
- The data are **not** zero-inflated (7 zeros, 0 ones), so you do **not** need ZOIB
  here. If a future, broader sample develops real mass at 0/1, the clean modern
  choice is **ordered beta regression** (Kubinec 2023, *Political Analysis*) — it
  handles (0,1) interior plus point masses at the bounds in *one* GLM with shared
  coefficients and no Smithson–Verkuilen fudge.
- If heavy tails persist after the `phi` submodel, a robust option is to keep the
  Beta but allow occasional outliers via a small contamination component, or move to
  a logit-normal / Student-t-on-logit response.

### 4.7 Sampler settings (after, not instead of, the above)
Once the geometry is fixed: `adapt_delta = 0.9` is plenty (0.95 was masking the
funnel), `max_treedepth = 10` should suffice, and 1000 warmup / 1000 sampling is
fine. If E-BFMI is still < 0.3 *after* reparameterizing, that's a signal the
remaining issue is the likelihood (§4.6), not tuning.

---

## 5. Suggested order of operations

The reparameterized family is now shipped and is the **runner default**:
`model_track_v3.stan`, `model_segment_v3.stan`, `model_timeline_v3.stan` (artist
term removed, all terms non-centered, genre-varying precision submodel,
re-calibrated priors — all compile via `stanc`). `run_models.R` builds the extra
`orientation_known` field automatically.

1. Add `clap_*` columns (§0) so the comparison is even possible.
2. Fit the v3 battery and check it actually fixed the diagnostics:
   ```bash
   Rscript stan/run_models.R both 500 1 v3     # v3 is also the default
   Rscript stan/evaluate_models.R 500 1
   ```
   Confirm E-BFMI > 0.3 on every chain and Pareto-k mostly < 0.7 in the
   `sampler_diagnostics.csv` / `loo_reliability.csv` tables.
3. If you want a before/after, fit the originals too (`... 500 1 v2`) and compare
   the diagnostics tables side by side.
4. Read the single `loo_compare_all` for the MuLan-vs-CLAP × measure ranking.
5. Only then reach for §4.4–4.6 refinements if diagnostics still complain (the
   `orientation_known` indicator from §4.4 is already wired into v3; §4.5–4.6 are
   not).

---

### References
- Betancourt (2016), *Diagnosing Biased Inference with Divergences* (E-BFMI, funnels).
- Vehtari, Gelman & Gabry (2017), *Practical Bayesian model evaluation using LOO-CV
  and WAIC* + the **loo** package FAQ (parameter-per-observation ⇒ high Pareto-k).
- Ferrari & Cribari-Neto (2004), *Beta regression for modelling rates and
  proportions* (precision submodel).
- Kubinec (2023), *Ordered Beta Regression* (boundary-inclusive (0,1) response).
- Stan User's Guide — *Reparameterization* (non-centered hierarchical models).
