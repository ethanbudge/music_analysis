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
│   └── utils.py              device, LRC parsing, cosine, embedding I/O
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
