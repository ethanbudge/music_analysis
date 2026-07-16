# Sound and Sense: Multimodal Lyric-Music Congruence and Streaming Popularity

*A reproducible pipeline for measuring how well a song's **lyrics** and its
**music** agree — at the whole-song, chorus, and line level — and estimating
whether that agreement predicts streaming popularity.*

---

## Overview

**Lyric-Music Congruence (LMC)** is the cosine similarity between a song's audio
and its lyric text in a *joint* audio-language embedding space. When a model has
been trained to place matching audio and text near each other, a high cosine
means the sound and the words "say the same thing"; a low cosine means they pull
in different directions.

This repository:

1. **Samples** songs that have time-synced lyrics from the public
   [LRCLIB](https://lrclib.net) data dump (≈ 19.5 M synced-lyric tracks). LRCLIB
   is the single source of truth — every other data source keys off a sampled
   track's LRCLIB ID.
2. **Gathers** the official audio (not music videos, to keep the lyric timing in
   sync) and a battery of public **popularity** signals (Spotify popularity,
   YouTube view/like/comment counts, Deezer rank, optional Last.fm).
3. **Computes LMC** with two joint embedding models — **MuQ-MuLan** and
   **LAION-CLAP** — at three granularities:
   - **song-wide** (whole audio vs. whole lyrics),
   - **segment-wide** (chorus vs. non-chorus, with algorithmic chorus detection),
   - **line-by-line** with four audio **context windows** — exact (0 s), ±1 s,
     ±5 s, ±10 s — exploiting the LRCLIB timestamps.
4. **Models** popularity as a function of LMC in a hierarchical **Beta
   regression** (Stan), recovering a genre / narrative-vs-production structure
   from Spotify metadata.

The pipeline is **fully resumable and incremental**: progress is tracked in a
small SQLite database, and every expensive artifact (audio, embeddings, mood,
popularity) is cached and never recomputed.

---

## What changed from the first-year version

| Original | Now |
|---|---|
| Curated 12-artist / 4-genre catalog, lyrics scraped from Genius | Broad random sample from the **LRCLIB** synced-lyric universe |
| WhisperX forced-alignment for line timing | **Native LRCLIB synced timestamps** (no transcription needed) |
| Genre / orientation hand-coded | **Recovered** from Spotify artist genre tags |
| Spotify popularity only | Spotify + **YouTube + Deezer + Last.fm** |
| Line LMC at a single ±5 s window | Line LMC at **exact / ±1 / ±5 / ±10 s** windows |
| Section labels from Genius `[Chorus]` headers | **Algorithmic chorus detection** from repeated synced lines |
| MuLan, CLAP, **and a MERT+SBERT baseline** | MuLan + CLAP only (**baseline removed**) |
| Per-stage JSON checkpoints | **Single SQLite project DB**, per-song caching |

---

## Repository layout

```
.
├── src/lmc/                  Python package (the pipeline)
│   ├── config.py             paths, models (MuLan/CLAP/MERT), windows, filters
│   ├── db.py                 project SQLite schema + progress queries
│   ├── lrclib.py             setup()/sample() from the LRCLIB dump
│   ├── audio.py              yt-dlp official-audio download + YouTube metrics
│   ├── popularity.py         Spotify / Deezer / Last.fm + genre keyword map
│   ├── genre.py              ensemble genre: Spotify → MusicBrainz → zero-shot
│   ├── mood.py               librosa mood proxies (control option)
│   ├── mert.py               MERT-v1-330M per-song vectors (control option)
│   ├── chorus.py             chorus detection from repeated synced lines
│   ├── embeddings.py         MuLan + CLAP (laion_clap) + MERT encoders
│   ├── alignment.py          LMC: song / line-windows / chorus vs non-chorus
│   ├── combine.py            → master_results.csv (+ MERT PCA, ensemble genre)
│   ├── utils.py              device, LRC parsing, cosine, embedding I/O
│   ├── gather.py             steps 1-4 as one batched call (run_batch)
│   └── scheduler.py          on/off + hours-window loop around gather.run_batch
├── scripts/auto_gather.py    CLI: enable/disable/set-hours/run the scheduler
├── tests/test_scheduler.py   unit tests for the scheduler's window/toggle logic
├── notebooks/pipeline.ipynb  resumable end-to-end driver
├── analysis/
│   ├── summary_stats.R       quick descriptives + figures
│   ├── report_helpers.R      fit → labelled summaries (controls, genre, β(t))
│   └── lmc_report.qmd        rendered results report (per embedding × controls)
├── stan/                     v4 model family (generic controls + functional)
│   ├── model_track_v4.stan        single-measure LMC → popularity
│   ├── model_segment_v4.stan      chorus vs non-chorus → popularity
│   ├── model_curve_v4.stan        scalar-on-function: ∫β(t)·LMC(t)  (curvature)
│   ├── model_segment_curve_v4.stan section-aware β_chorus(t)/β_nonchorus(t)
│   ├── model_line_curve_v4.stan   one-stage line-level joint model (experimental)
│   ├── run_models.R          cmdstanr runner; controls toggle (mood/mert/both)
│   ├── evaluate_models.R     family-agnostic diagnostics sweep
│   ├── MODEL_NOTES.md        why the parameterization is what it is
│   └── archive/              superseded v2 / v3 models (provenance only)
├── data/                     LRCLIB dump, audio, embeddings, project.db  (gitignored)
├── results/                  master_results.csv, lmc_lines.csv           (gitignored)
├── archive/                  the original first-year codebase (preserved, unused)
├── requirements.txt
└── README.md
```

> **Nothing dense is tracked.** `data/`, audio, embeddings, raw lyrics, the
> LRCLIB dump, and rendered reports are all `.gitignore`d.

---

## Setup

### 1. Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/OpenMuQ/MuQ.git    # MuQ-MuLan
brew install ffmpeg                                    # or: apt install ffmpeg
```

### 2. The LRCLIB dump

Download the dump from <https://lrclib.net> and place it under `data/`:

```
data/lrclib-db-dump-YYYYMMDD.sqlite3
```

(It is auto-detected; or set `LRCLIB_DUMP=/path/to/dump.sqlite3`.)

### 3. API credentials (environment variables)

```bash
export SPOTIFY_CLIENT_ID="..."        # https://developer.spotify.com/dashboard
export SPOTIFY_CLIENT_SECRET="..."
export YOUTUBE_API_KEY="..."          # optional (yt-dlp already yields most metrics)
export LASTFM_API_KEY="..."           # optional
```

### 4. R (for analysis / modeling)

```r
install.packages(c("tidyverse", "here", "cmdstanr", "loo"))
cmdstanr::install_cmdstan()
```

---

## Running the pipeline

Open `notebooks/pipeline.ipynb` and run top to bottom, or call the package
directly:

```python
import sys; sys.path.insert(0, "src")
from lmc import config, lrclib, audio, popularity, mood, chorus, embeddings, alignment, combine, db

config.ensure_dirs()
lrclib.setup()             # universe / sampled / remaining
lrclib.sample(50)          # draw a session target of NEW songs

audio.download_pending()   # official audio + YouTube metrics
popularity.fetch_pending() # Spotify popularity + recovered genre/orientation
mood.extract_pending()     # librosa mood proxies
chorus.compute_pending()   # chorus vs non-chorus labels

embeddings.embed_pending("mulan")   # cached; recompute-free
embeddings.embed_pending("clap")
alignment.compute_pending()         # all LMC measures
combine.build_master()              # results/master_results.csv

db.progress_report()       # how far each stage has gotten
```

Stop and rerun any time — each stage processes only what is still outstanding.

### Auto-gathering on a schedule

`scripts/auto_gather.py` drives steps 1-4 above (sample → audio → popularity →
mood/MERT/chorus) with a single command, in batches of 100 new songs, and can
be left running so it keeps gathering during a daily hours window as long as
the device is on. It is **off by default** — it does nothing until you `enable`
it.

```bash
python scripts/auto_gather.py status              # show the toggle + window
python scripts/auto_gather.py enable               # turn scheduled gathering on
python scripts/auto_gather.py disable              # turn it back off
python scripts/auto_gather.py set-hours 1 6        # only gather 1am-6am local time
python scripts/auto_gather.py run                  # leave this running (Ctrl-C to stop)
python scripts/auto_gather.py run --once           # run a single batch right now, for testing
```

The toggle and hours window are saved to `data/scheduler_state.json` (gitignored),
so they persist across `run` invocations. `run` wakes up once a minute, checks
the toggle and the clock, and runs a 100-song batch whenever both say go —
back-to-back until the window closes. To have it survive closing the terminal,
run it under `nohup`/`screen`/`tmux`, e.g.:

```bash
nohup python scripts/auto_gather.py run > gather.log 2>&1 &
```

Every stage it calls is already resumable and idempotent (see `db.py`), so
stopping and restarting `run` — or letting the window close mid-batch — never
duplicates work. `--batch-size` and `--poll-seconds` on `run` override the
defaults (100 songs, 60s) if you want a different cadence.

### Statistics & modeling

```bash
Rscript analysis/summary_stats.R              # descriptives + figures
Rscript stan/run_models.R both 0 42 mert      # embeddings, N(0=all), seed, controls
Rscript stan/evaluate_models.R                # diagnostics + LOO sweep
quarto render analysis/lmc_report.qmd -P embedding:mulan -P controls:mert
```

`run_models.R` fits, on one **shared corpus** (so LOO is comparable), the **track**
model once per LMC measure plus the **segment / curvature / segment+curvature**
trajectory models, for each embedding. The **control block is a toggle** —
`controls ∈ {mood, mert, both}` (default MERT) — so you can compare control sets.
`sample_corpus(df, N, seed)` and `build_corpus()` let you fit on any subset.

---

## Outputs

**`results/master_results.csv`** — one row per song:

| Column | Meaning |
|---|---|
| `track_id`, `title`, `artist`, `album`, `duration`, `n_synced_lines` | LRCLIB identity |
| `spotify_popularity` | Spotify 0–100 score (**primary outcome**) |
| `yt_view_count`, `yt_like_count`, `yt_comment_count`, `deezer_rank`, `lastfm_*` | other popularity signals |
| `genre`, `orientation` | recovered from Spotify genre tags |
| `mood_happy … voice_instrumental` | librosa mood/acoustic controls |
| `mulan_song`, `mulan_line_exact … mulan_line_buf10`, `mulan_seg_chorus`, `mulan_seg_nonchorus` | LMC measures (MuLan) |
| `clap_song`, `clap_line_*`, `clap_seg_*` | LMC measures (CLAP) |
| `song_age_years` | derived from release date |

**`results/lmc_lines.csv`** — long line-level series (`track_id, model, line_idx,
window, position_pct, is_chorus, lmc`) for the timeline analysis.

---

## Generative arm — controlled LMC stimuli (`src/lmcgen`)

Where the observational arm *measures* LMC on real songs, the generative arm
*manufactures* stimuli with LMC under experimental control, to feed a **human-validation
survey** of the construct.

**Design — a 2 × 2 valence/arousal (VA) circumplex.** Four *extreme* VA corners are the
music targets:

| corner | valence / arousal | character |
|---|---|---|
| `hvha` | high / high | joyful, euphoric |
| `hvla` | high / low  | calm, content, serene |
| `lvha` | low / high  | angry, afraid, tense |
| `lvla` | low / low   | sad, weary, hopeless |

**16 original two-line couplets** (four authored per corner, frozen in `lyrics.py`) are
each sung at **all four music corners**, with **four repetitions** per cell:

> 16 lyrics × 4 music corners × 4 reps = **256 song–lyric pairs**.

This fully crosses *lyric corner* × *music corner* (the 4 × 4 matrix): the diagonal is
congruent, the anti-diagonal maximally incongruent, so LMC varies systematically by
construction, with robust repetition so on-target winners can be picked per cell.

**Generation — Google Lyria 3 Clip** (Gemini API, `GEMINI_API_KEY`). Lyria takes the
lyrics as `[Chorus]` tags plus the musical style as prose (`generate.build_prompt`:
fixed voice descriptor + the corner's `style_words` + bpm/key). Crucially it has **no
embedding input, no seed, and does not reproduce** — voice/tempo/key/mood live only in
the prompt text. So each corner is hit by prompt engineering and *validated post-hoc*
against a MuLan target embedding (Lyria cannot consume an embedding). A single fixed
voice descriptor keeps the singer ~constant; song fidelity is prioritised over a
perfectly identical voice. All output carries a **SynthID** watermark.

**Validation.**
1. *Lyric presence* — Whisper (`asr.py`, faster-whisper) transcribes each clip; word
   error rate vs. the couplet screens whether the lyrics were sung faithfully. (The
   echoed Lyria lyrics, saved to `<clip>.lyria.txt`, are a free extra cross-check.)
2. *VA placement, embedding* — **MuLan** audio-vs-corner-anchor cosine: the anchor is
   the embedding target the song was aimed at; its argmax is the predicted corner.
3. *VA placement, acoustic* — **librosa** valence/arousal (`va.py`) → nearest corner +
   distance to the numeric target: a genre-robust, model-independent second opinion.
4. *Realised LMC* — `cos(generated-audio, own-lyric text)` in MuLan, the same definition
   as the observational arm; congruent cells should embed more similarly than incongruent.

Lyric-side placement is additionally checked model-independently (lexical count vs. each
corner's affect lexicon, and a VA-lexicon nearest-corner) in `analysis.lyric_placement`.

```
src/lmcgen/
  config.py     paths, VA design constants, Lyria config, ASR/validation knobs, DRY_RUN
  quadrants.py  the 4 extreme VA corners: coordinates, Lyria style words, MuLan anchor
                prompts, lexicon, representative bpm/key
  lyrics.py     the 16 authored 2-line couplets + lexical / VA placement checks
  audioio.py    GenSpec + audio I/O helpers (mock synth, WAV write, mp3→wav transcode)
  lyria.py      Google Lyria 3 Clip backend (real + dry-run mock), resumable
  mulan.py      Scorer + VA-corner anchor builder (reuses lmc.embeddings)
  asr.py        Whisper (faster-whisper) transcription + word-error-rate screening
  va.py         librosa acoustic VA + lexicon lyric VA + VA-congruence
  generate.py   Phase 1: build the 256 specs → generate with Lyria (resumable)
  validate.py   Phase 2: WER + MuLan + VA on every clip → results/generation/songs.csv
  analysis.py   descriptive stats, figures, winner selection, survey export
notebooks/generation_pipeline.ipynb   end-to-end driver → songs.csv + figures + survey export
```

**Dry-run first.** `LMCGEN_DRY_RUN=1` (default) synthesises cheap VA-dependent *mock*
audio so the whole pipeline runs with no API calls; every number is then a MOCK
placeholder. Flip to real by setting `LMCGEN_DRY_RUN=0`, then run
`generate.clean_generated()` **once** (so mock clips aren't reused) and re-run.

**Running for real.** Real generation is **256 preview-API calls** (one per rep;
resumable, so a stopped batch skips finished clips). No local generation model is
loaded — Lyria is remote — so MuLan (validation) and small Whisper co-reside
comfortably; there is no ACE-Step-style two-phase memory dance any more. Outputs land in
`data/generation/audio/`, `results/generation/songs.csv`, the loudness-normalised survey
stimuli + `manifest.csv` in `data/generation/survey/`, and figures in
`analysis/output/figures/generation/` (all gitignored). Set `LMCGEN_CANDIDATES=3` to keep
the best-of-N take per rep by WER instead of the default single call.

Choosing the VA design over eight discrete emotions is deliberate: music renders
valence/arousal strongly on its two dominant axes, whereas fine-grained emotions with
weaker musical signatures blur in embedding checks — so four extreme, well-separated
corners give the cleanest manipulation. Earlier generation attempts (a local ACE-Step
8 × 8 emotion grid; a singing-voice-synthesis route via Synthesizer V / DiffSinger /
OpenUtau; a Suno backend) are preserved under `Archive/legacy_generation/`,
`Archive/legacy_svs/`, and `Archive/legacy_notebooks/`.

> **KNOWN ISSUE — MuLan text-tower lyric alignment (open, project-wide).** MuLan's text
> tower is trained on music *descriptions/tags* (MusicCaps-style), not lyrics, so raw
> lyric text is out-of-distribution for it and *valence* is a stronger signal than
> fine-grained emotion in this embedding. This affects any `mulan_*` column that touches
> lyric text in BOTH arms, so the fix belongs project-wide, not patched locally. The VA
> design mitigates it (four coarse, valence-dominated corners are exactly what MuLan
> reads well), and lyric-side placement is cross-checked with model-independent lexical
> and VA-lexicon scores. Candidate fixes to evaluate: (1) prompt-ensemble the anchors and
> score by margin, not raw cosine; (2) mean-center text/audio embeddings separately (the
> tower "modality gap"); (3) add convergent text-native validators (NRC-VAD / GoEmotions)
> and treat cross-method agreement as the evidence; (4) a small human rating pass — which
> the survey this arm feeds is itself a step toward. **Deferred** pending a broader pass.

---

## Method notes & limitations

- **Audio provenance.** We prefer auto-generated `"<Artist> - Topic"` channels and
  audio uploads over music videos, and reject candidates whose duration deviates
  from the LRCLIB track duration, because video edits desync the lyric timestamps.
  Live / remix / karaoke titles are excluded at sampling time.
- **Chorus detection** is lyrics-based: the most frequently recurring multi-line
  block (or, failing that, the single most repeated line) is labeled chorus. This
  is transparent and aligns with the synced timestamps; an audio self-similarity
  approach is a natural future extension.
- **Genre / orientation recovery** maps Spotify artist genre tags to coarse
  clusters and a narrative-vs-production orientation via a keyword map
  (`popularity.GENRE_MAP`). Songs without usable tags are `unknown`.
- **Mood features** are librosa signal-processing proxies, not learned classifiers
  — fine as controls, not ground truth.
- **MuLan text-tower lyric alignment is an open issue (project-wide).** MuLan's text
  tower is trained on music descriptions/tags, not lyrics, and under-resolves
  fine-grained emotion vs. valence when given raw lyric text — see the KNOWN ISSUE
  box under "Generative arm" above for evidence and candidate fixes. Any
  interpretation of `mulan_*` LMC columns that hinges on precise lyric emotion
  (rather than valence/topic-level alignment) should flag this.

---

## Citation

If this pipeline contributes to your work, please cite the associated paper
(*Sound and Sense: Multimodal Lyric-Music Congruence and Its Effect on Streaming
Popularity*) and the underlying resources:

- **LRCLIB** — synced lyrics database, <https://lrclib.net>
- **MuQ-MuLan** — OpenMuQ team, 2024
- **LAION-CLAP** — Wu et al., ICASSP 2023

---

## Legal

Audio downloaded via yt-dlp is for **non-commercial academic research only** and
is **not redistributed**. Lyrics from LRCLIB are used under fair-use principles
for research; raw lyrics are excluded from version control.
