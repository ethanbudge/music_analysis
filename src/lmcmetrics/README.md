# lmcmetrics — alternative Lyric-Music Congruence metrics

A **separate, self-contained** package that sits next to the observational arm
(`src/lmc`) and generative arm (`src/lmcgen`) **without changing or deleting either**.
It reads their cached artifacts and adds new, academically-grounded ways to measure
lyric-music congruence — plus one common yardstick to compare all of them fairly.

Motivation and the full literature basis are in
[`docs/congruence_metrics_review.md`](../../docs/congruence_metrics_review.md).
The one-line problem: your current LMC is `cos(audio, lyric_text)` in MuLan/CLAP, whose
text tower was trained on music *captions*, not *lyrics* — so lyrics are
out-of-distribution and the number is unreliable. This package fixes that four ways.

---

## The four steps (and where each lives)

| Step | Idea | File | Needs training? |
|---|---|---|---|
| **1** | Remove the audio↔text "modality gap" before any cosine | `centering.py` | no |
| **2** | **The yardstick**: does a metric rank a song's TRUE lyrics above impostor lyrics? | `evaluate.py` | no |
| **3** | **LyricLMC**: learn a lyric↔audio space from YOUR pairs (fixes the text tower) | `lyriclmc.py` | yes (tiny) |
| **4** | **CKA / RSA**: do the audio and lyric *geometries* agree? | `geometry.py` | no |

Everything is validated on synthetic data by `selftest.py` (run it first, below).

---

## Quickstart

**0. Install the one new dependency** (into your `lmc` conda env):

```bash
pip install sentence-transformers        # the new lyric text tower
# optional, only for the Gromov-Wasserstein extra in geometry.py:
# pip install pot
```

**1. Sanity-check the maths with no corpus needed:**

```bash
cd src && python -m lmcmetrics.selftest
```

You should see `ALL PASSED`. This proves the harness, centering, geometry, and
LyricLMC training all behave correctly on data with a known answer.

**2. Run the whole thing on your real corpus** (in the `lmc` env, after the
observational pipeline has produced MERT vectors + MuLan/CLAP bundles):

```python
import sys; sys.path.insert(0, "src")
from lmcmetrics import run
out = run.run_all()      # caches lyric vectors, trains LyricLMC, compares all
                         # metrics on held-out songs, writes CSVs, prints tables
```

Outputs land in `results/lmcmetrics/` (gitignored):
- `retrieval_heldout.csv` — every metric ranked by matched-vs-mismatched AUC.
- `retrieval_heldout_within_genre.csv` — the harder, within-genre version.
- `geometry_global.json` — corpus-level CKA / RSA.
- `congruence_scores.csv` — **per-song** columns to merge into `master_results.csv`.
- and one training-run folder per LyricLMC fit under `data/lmcmetrics/lyriclmc/`.

---

## Step 1 — Centering (the modality-gap fix), in plain terms

In MuLan/CLAP, audio vectors and text vectors don't actually overlap — they sit in two
separate "clouds" with a constant gap between them. A raw cosine mixes *"do these
agree?"* with *"where do the two clouds sit?"*. Centering just subtracts each cloud's
average before the cosine, so what's left is agreement. It's one line, no training, and
it never makes things worse. In the self-test it recovers AUC from 0.87 → 1.00 on data
where a gap was injected. Use it as a free upgrade to your existing MuLan/CLAP numbers
(the `mulan_centered` / `clap_centered` columns).

---

## Step 2 — The yardstick, in plain terms

This is the most important tool here. It answers *"is metric B actually better than
metric A?"* objectively, with no humans, on the whole corpus:

> A good congruence metric should score a song's **own** lyrics higher than a
> **random other song's** lyrics laid over the same audio.

Give the harness a score matrix `S` (S[i,j] = congruence of song i's audio with song
j's lyrics). The diagonal is the true pairs; everything else is "impostors". It reports:

- **AUC** — the headline. Probability the true pair beats a random impostor. 0.5 = the
  metric is useless (coin flip); 1.0 = perfect. **A metric with higher AUC is objectively
  better at detecting congruence.** This is how you prove LyricLMC beats MuLan.
- **recall@1 / @5** — how often the true partner is the top (or top-5) match.
- **median rank**, **MRR** — where the true partner lands on average.
- **per-song margin** — a bonus: for each song, how far its true score sits above its own
  impostor distribution. That's a *calibrated per-song congruence value* you can correlate
  with the human survey.

"Within-genre" mode (`restrict='same'`) forces impostors to be from the same genre — a
much stricter test that controls for topic/genre and isolates finer congruence.

---

## Step 3 — LyricLMC (your first trained model), in plain terms

**You have never trained a model, so here is exactly what happens — no jargon.**

### What we are (and are NOT) training
We are **not** retraining any big neural network. We take two things that already exist
for each song:

- its **MERT audio vector** (1024 numbers describing the sound — already cached), and
- its **lyric vector** (384 numbers describing the words — made by a sentence encoder),

and we learn two small "adapters" (called *projection heads*) that reshape each into a
shared 256-number space where **matching lyrics and music end up pointing the same way**.
That pair of adapters is the *entire* trainable model — a few hundred thousand numbers.
The big encoders stay frozen. This is why it trains in **minutes on your laptop** and
why it only ever touches cached embeddings (never audio files — see Legal below).

### How the learning works (the "contrastive" idea)
Training happens in small batches of, say, 256 songs. For each batch the model looks at
all 256×256 possible (audio, lyrics) combinations and is rewarded for making each song's
**own** pair the closest, while pushing the 255 mismatched pairings apart. Repeat over
many batches and the space self-organizes so that congruent pairs are close. The 255
mismatches per song are the "negative examples" it learns from — you don't have to label
anything.

### What you actually do
```python
from lmcmetrics import run
model, hist = run.train(run._load_reps())     # trains + saves a run folder
print(hist["best_val_auc"])                    # how well it does on held-out songs
```
Or let `run.run_all()` do it as part of the full workflow.

### How to read the result and iterate
- Each run is saved to its own folder `data/lmcmetrics/lyriclmc/run_YYYYMMDD_HHMMSS/`
  with the weights, the exact config, the train/val song split, and a `history.csv`
  learning curve. **Nothing is overwritten**, so you can train many iterations and
  compare them — exactly what you asked for.
- The number that matters is **best_val_auc** (from Step 2, on songs the model never
  trained on). Higher = better. Compare it to MuLan's AUC on the *same* held-out songs
  (the `run_all()` table does this for you, leakage-free).
- To try a variant, pass a config override, e.g. a different shared dimension:
  `run.train(reps, cfg={"proj_dim": 128, "epochs": 100})`, or swap the lyric encoder
  (below). Give it a `run_name=` to label the iteration.

### Knobs worth knowing (all in `config.LYRICLMC`)
`proj_dim` (shared space size), `batch_size` (more = more negatives = usually better),
`epochs`, `lr`, `val_frac`, `patience` (early stopping). Defaults are sensible; you can
ignore them at first.

---

## Step 4 — CKA & RSA (geometry congruence), in plain terms

**You said you'd never met CKA/RSA — here is the whole idea with an analogy.**

Imagine two maps of the same 50 cities. One map places cities by how their **music**
sounds; the other by what their **lyrics** say. If the two maps put the *same* cities
near each other — even if one map is rotated, flipped, or a different scale — then music
and lyrics are "saying the same thing" about those cities. CKA and RSA are just two ways
to measure *how similar the two maps are*, using only within-map distances. Crucially,
**they never compare an audio vector to a text vector directly**, so the modality gap
(Step 1's problem) can't contaminate them at all.

Concretely, for your corpus:

- **RSA** (Representational Similarity Analysis): build the table of "how similar is song
  i to song j **in sound**" and the table of "how similar in **lyrics**", then correlate
  the two tables. One number in [−1, 1]. High = musical neighbors are also lyrical
  neighbors.
- **CKA** (Centered Kernel Alignment): the same spirit but a normalized 0–1 score that is
  **immune to rotation and rescaling** of either space, which is what makes it safe to
  compare two different representations (1024-d audio vs 384-d lyrics). 1 = the two
  geometries are identical up to rotation/scale; 0 = unrelated.

Both are **one number for the whole corpus** — e.g. *"in this corpus, music and lyric
geometries align at CKA = 0.31."* That is already a publishable statement (and you can
compute it per genre: *"congruence geometry is tighter in folk than in EDM"*).

### The per-song version: "local RSA"
To get **one congruence number per song** (so it can be a column and be correlated with
the survey), we do RSA *locally*: for song i, we ask *"do the songs that are musically
close to i also tend to be lyrically close to i?"* If yes, song i sits in a region where
music and words co-vary — it is **locally congruent**. This is `geom_local_rsa` in the
output.

**Important honesty note to put in your thesis:** local RSA measures congruence
*relative to how music and lyrics co-vary across the corpus*, which is a **different**
(and complementary) notion from LyricLMC's *direct* "do these specific lyrics match this
specific music" score. Report both; where they agree, that convergence is itself
evidence. Neither is "the" answer — that's the whole point of a convergent battery.

---

## What to do with the outputs

1. **Believe the yardstick.** If `lyriclmc` (and/or `mulan_centered`) beats `mulan_raw`
   on held-out AUC, you have an objective, corpus-wide result that MuLan under-measures
   congruence. That is a genuine methodological contribution on its own.
2. **Merge the per-song columns** from `congruence_scores.csv` into
   `results/master_results.csv` by `track_id`, then refit the Stan models with the new
   congruence columns (`lyriclmc_song`, `geom_local_rsa`, `mulan_song_centered`) in place
   of / alongside `mulan_song`.
3. **Validate against the survey (the real gold standard).** Score the generative-arm
   stimuli with LyricLMC (`model.score_pairs(mert_vecs, lyric_vecs)`) and correlate with
   human congruence ratings; a better metric tracks humans more tightly and separates the
   congruent-diagonal from the incongruent-antidiagonal of your 4×4 design more cleanly.
   Do **not** pick the metric by how well it predicts popularity (that's circular — see
   the review memo §6).

---

## Legal — training on YouTube-sourced audio (read this)

**Not legal advice; this is an informed engineering summary. Confirm with your advisor,
department, and university library before releasing anything.** That said, the design
here was chosen specifically to keep you in the low-risk zone, and here is the reasoning:

**The key fact: LyricLMC never trains on audio.** It trains on **cached embeddings** —
the MERT audio vector (1024 numbers) and the lyric sentence vector. The trainable part is
two small projection heads. These vectors are *derived features*; they **cannot
reconstruct the original recording**, and the model is not generative (it can't produce
audio). This is materially different from, and much lower-risk than, training a generative
model on raw audio. It is exactly the posture mainstream MIR research uses.

There are three distinct legal layers; none is aggravated by this package:

1. **Copyright in the sound recordings (the training-data question).**
   - US: training ML on copyrighted works for **non-commercial research** is widely argued
     to be transformative *fair use* (17 U.S.C. §107); the "creative work" factor cuts
     against but is rarely decisive. The established MIR norm (MERT, JukeMIR, MARBLE, etc.)
     is to train on copyrighted audio for research and **distribute the learned
     representations, not the audio** — precisely because "released features and embeddings
     cannot reconstruct original recordings."
   - EU/UK: there are **explicit text-and-data-mining research exceptions** (EU DSM
     Directive Art. 3) permitting non-commercial research mining of lawfully accessible
     works.
   - Your use is non-commercial academic research → squarely in the favorable zone.

2. **YouTube Terms of Service (a contract matter, separate from copyright).** Downloading
   via yt-dlp is a ToS question you already engaged when building the observational arm.
   Training on embeddings you already hold **adds no new copyright exposure** beyond
   possessing the audio, which is a pre-existing fact of the project.

3. **Lyrics copyright.** Lyric text (LRCLIB) is used for research and never redistributed;
   the package stores only *derived* lyric vectors, not the text.

**Practical rules this package already follows — keep following them:**
- Never redistribute audio, or anything that can reconstruct it. (We only cache and use
  embeddings.)
- Keep it non-commercial and research-scoped; cite datasets (LRCLIB, MERT, etc.).
- You **may** publish/release the LyricLMC weights and the derived metrics (they don't
  reproduce audio) — but get a one-time OK from your advisor/library first, and do **not**
  release a mapping from vectors back to specific recordings.
- If the project ever turns commercial, re-evaluate — the fair-use / TDM footing changes.

**Bottom line:** because you train on embeddings and never redistribute audio, the extra
legal risk from "training a model" here is minimal and consistent with standard MIR
practice. The main thing to actually do is a quick confirmation with your institution,
which is routine for MIR work.

---

## Choices you can make

- **Lyric text encoder** (`config.TEXT_ENCODER` / env `LMCMETRICS_TEXT_ENCODER`):
  default `sentence-transformers/all-MiniLM-L6-v2` (tiny, fast). For more topical nuance:
  `intfloat/e5-base-v2`. For affect focus: an emotion-tuned encoder (document your choice).
  Re-cache lyric vectors after changing it (`lyric_encoder.embed_pending(force=True)`).
- **Audio tower** (`config.AUDIO_REP`): `mert` (default), or `mulan`/`clap` audio.
- **Shared dimension / training**: `config.LYRICLMC["proj_dim"]`, `batch_size`, `epochs`.
- **Geometry**: `config.GEOMETRY["local_k"]` (None = per-song RSA over all songs; an int =
  restrict to that many nearest audio neighbors for a stricter local notion).

---

## File map

```
src/lmcmetrics/
  config.py         paths (under data/lmcmetrics, results/lmcmetrics), model choices, knobs
  centering.py      Step 1 — per-modality mean-centering (modality-gap fix)
  evaluate.py       Step 2 — matched-vs-mismatched retrieval harness (the yardstick)
  lyric_encoder.py  the new lyric text tower (sentence encoder), cached per song
  lyriclmc.py       Step 3 — projection-head contrastive model (train / save / load / score)
  geometry.py       Step 4 — CKA, RSA (global) + local RSA (per song); optional Gromov-Wasserstein
  scorers.py        common Scorer interface so every metric plugs into evaluate.py
  data.py           load matched pairs from the caches; synthetic generators
  run.py            end-to-end driver on the real corpus (writes results/lmcmetrics/*)
  selftest.py       runs the whole pipeline on synthetic data (correctness check)
  README.md         this file
```
