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

## Generative arm — controlled LMC stimuli (`src/lmcgen`)

Where the observational arm *measures* LMC on real songs, the generative arm
*manufactures* stimuli with LMC under experimental control, to feed a downstream
respondent study on liking / repeat-listening.

**Design.** Eight original, clean two-line **hooks** — one per Plutchik high-intensity
emotion (ecstasy, admiration, terror, amazement, grief, loathing, rage, vigilance) —
are each set to eight musical emotions, giving an **8 × 8 grid of 64 hooks**. Lyric
content is held fixed down each column; musical emotion is held fixed across each row.
The diagonal is *congruent* (lyric emotion = music emotion); off-diagonal cells are
*incongruent* by degree, so LMC varies systematically by construction.

**One genre, one voice (v2).** To avoid confounding musical emotion with *genre* (the
original design used a different genre per emotion — rage=metal, grief=ambient — so
emotion and genre were collinear, and sparse genres mumbled the lyrics), every cell is
now **90s alternative rock with a fixed lead voice**; emotion is varied *within* the
genre via tempo, mode, dynamics and intensity (`emotions.py`: `GENRE_BASE` +
`VOICE_BLURB` + per-emotion `emotion_style`). Hooks are short and repeated so the
vocal renders the words cleanly.

**Generation.** Music is generated locally with **ACE-Step 1.5** (base model, Apple
Silicon). ACE-Step conditions on *text* (a caption + lyrics) and scalar metadata
(bpm, key), **not** on a raw target embedding — so each musical emotion is a
caption / bpm / key recipe. The 5 Hz LM planner is disabled so our explicit emotion
metadata is used verbatim; `guidance_scale` is raised to ~9 for tighter lyric
adherence.

**Lyric-intelligibility screening.** Because the experiment needs the sung lyrics to
be *exactly* right and ACE-Step doesn't guarantee that, each cell is generated as
**best-of-N takes**: every take is transcribed with Whisper (`asr.py`, faster-whisper)
and scored for word error rate (WER) vs. the target hook; the clearest take is kept
and its WER is recorded as a per-clip control (`results/generation/wer.json`, `wer`
column). Sung-audio WER is inherently high (25–50% even when clearly intelligible), so
it's a *relative* screen and a go/no-go signal for whether ACE-Step is good enough
versus a paid tool (Suno/Udio have clearer vocals but less control and ToS friction
for redistributed stimuli).

ACE-Step 1.5 is installed separately via `uv` (**not** `pip install acestep` — no
such package exists) into its own environment, and is driven over its **REST API
server** as a separate process, not imported in-process. This keeps ACE-Step's
dependency stack (its own torch / MLX build) fully isolated from the `lmc` conda
env's carefully pinned torch/numpy (see the CLAP/numpy history above) — the two
never need to coexist in one interpreter. Before real generation: start the server
(`uv run python -m acestep.api_server`, or `./start_api_server_macos.sh`, from your
ACE-Step-1.5 clone) and call `acestep.check_server()` in the notebook, which fails
fast with setup instructions if it isn't reachable.

**Validation (three lines of evidence).**
1. *Lexical* (model-independent) — each chorus's words vs. the eight Plutchik/NRC
   EmoLex lexicons; the target emotion should win.
2. *Embedding, text-side* — each chorus's MuLan text embedding vs. the eight emotion
   anchors (independent of the audio, so no circularity).
3. *Cross-modal LMC* — `cos(generated-audio, lyric-text)` per cell: congruent cells
   should embed more similarly than incongruent ones (headline test), with a graded
   correlation against *designed* congruence, plus a music manipulation check and an
   optional independent **CLAP** validator.

```
src/lmcgen/
  config.py     paths, ACE-Step + MuLan + ASR settings, single-genre design, DRY_RUN
  emotions.py   8 emotions: valence/arousal, anchors, lexicon; GENRE_BASE+VOICE_BLURB
  lyrics.py     the 8 authored 2-line hooks + rationale + lexical alignment scoring
  mulan.py      Scorer + emotion-anchor builder (reuses lmc.embeddings)
  acestep.py    ACE-Step 1.5 REST client (real + dry-run mock), resumable
  asr.py        Whisper (faster-whisper) transcription + word-error-rate screening
  pipeline.py   generate_all() [best-of-N WER] → [stop server] → validate_all()
  analysis.py   descriptive statistics + figures (incl. WER)
notebooks/generation_pipeline.ipynb   end-to-end driver → stats + plots
```

**Dry-run first.** `LMCGEN_DRY_RUN=1` (default) synthesises cheap emotion-dependent
audio so the whole pipeline runs in seconds; every number is then a MOCK placeholder
and figures are labelled as such.

**Running for real on a 16 GB Mac — two memory-isolated phases.** MuLan (~a few GB)
and ACE-Step's 3.5B model (in its server process) must **never be resident at the
same time** on 16 GB — doing so spirals into swap and can crash the machine. The
pipeline enforces this by splitting into two phases that don't overlap:

1. **Generate** — start the ACE-Step API server with its optional LM disabled (we
   don't use it), then run phase 1, which is a thin HTTP client (no MuLan loaded):
   ```bash
   cd /path/to/ACE-Step-1.5 && ACESTEP_INIT_LLM=false ./start_api_server_macos.sh
   ```
   ```python
   pipeline.generate_all()      # downloads 64 clips; resumable; logs per-clip ETA
   ```
2. **Stop the ACE-Step server** (Ctrl-C — frees its memory), then validate:
   ```python
   out = pipeline.validate_all()   # loads MuLan only; embeds clips → tidy results
   ```

Recipe **tuning is off by default** (`LMCGEN_TUNE=1` to enable) because it needs both
models at once — only turn it on where RAM is ample. `pipeline.run()` still exists as
a single-process convenience for dry-run / big machines, but warns on 16 GB. Set
`LMCGEN_DRY_RUN=0` and, if you changed the port, `ACESTEP_API_URL` (default
`http://127.0.0.1:8001`); run `pipeline.clean_generated()` once when switching between
dry-run and real. Outputs land in `data/generation/audio/`, `results/generation/*.csv`,
and `analysis/output/figures/generation/*.png` (all gitignored).

### Generation backends & the pilot (`lyria.py`, `suno.py`)

The pipeline is backend-agnostic: each engine is a client exposing
`generate(GenSpec) -> wav`, and validation (WER / VA / MuLan LMC) is shared. Besides
the local `acestep` server there are two hosted text-to-song backends —
**`lyria`** (Google Lyria 3 via the Gemini API / `google-genai`, `GEMINI_API_KEY`,
lyrics passed as `[Chorus]` tags in the prompt, cleanest research licensing) and
**`suno`** (Suno v5.5 via a third-party REST provider, `SUNO_API_KEY`; best lyric
fidelity, but a *granted* commercial licence — verify it covers research + stimulus
sharing). No hosted API accepts a raw emotion embedding, so emotion is set via the
text recipes and *targeted/selected* by embedding distance to the emotion anchor.
`pipeline.pilot(backends=("lyria","suno"), emotions=(…))` generates the same hooks on
two backends and compares lyric WER + audio valence/arousal vs. the design target
(`results/generation/pilot_comparison.csv`) so you can pick one on evidence.

### SVS route — controlled singing-voice synthesis (`src/lmcsvs`)

The ACE-Step run failed the experiment's hard requirements: lyrics weren't sung
verbatim or intelligibly (fair WER ~0.73, 27% of clips had no intelligible vocal), the
voice drifted, and the requested per-emotion tempo was ignored (measured BPM didn't
track requested, corr ≈ 0.23). Text-to-song models are stochastic black boxes — exactly
the wrong tool when you need *exact lyrics + one fixed voice + reproducibility*.

**Singing-voice synthesis inverts this:** we *specify* the performance — exact notes,
exact lyrics, one fixed voice — so lyrics and voice are guaranteed by construction and
only the musical emotion is manipulated. `lmcsvs` does everything up to the vocal render,
headlessly and deterministically:

```
src/lmcsvs/
  config.py     paths, fixed-voice + emotion set (reuses lmcgen emotions/hooks)
  syllables.py  hook → per-note syllables (pyphen, with a naive fallback)
  melody.py     per-emotion melody: valence/arousal → mode, tempo, register, rhythm, contour
  score.py      assemble hook_L on melody_M → an engine-agnostic Score
  musicxml.py   Score → MusicXML (notes + lyrics + key + tempo)
  pipeline.py   export every cell → data/svs/scores/<L>__<M>.musicxml
  diffsinger.py headless render path: Score → .ds → OpenVPI DiffSinger inference
  validate.py   rendered wavs → the same WER / valence-arousal harness as lmcgen
notebooks/svs_pipeline.ipynb
```

**Two render paths.** (a) **Synthesizer V Studio 2 Pro / ACE Studio** — import the
MusicXML (Pro edition; File → Import), assign one fixed voice, batch-render. (b)
**Headless DiffSinger** (`diffsinger.py`) — fully scriptable from Python: `export_ds()`
writes `.ds` scores, `render()` shells out to a locally-installed OpenVPI DiffSinger +
English voicebank (set the `DIFFSINGER_*` env vars; `pip install g2p_en` for ARPABET).
English DiffSinger banks use a bank-specific phoneme dictionary, so verify one `.ds`
against your bank — importing the MusicXML into **OpenUtau** (free, matching DIFFS-EN
phonemizer, renders DiffSinger directly) is both the verification and a GUI fallback.

Workflow: `pipeline.export_scores()` writes 64 (or N²) MusicXML files → import them into
**Synthesizer V Studio 2 Pro** or **ACE Studio**, assign **one fixed voice**, batch-render
to `data/svs/audio/<L>__<M>.wav` → `validate.validate()` scores WER (should be ~0, since
the lyrics are sung by construction) and valence/arousal. The user chose a **controlled
voice** and **melody varied per emotion**; the melody encodes emotion on its strong axes
(tempo + mode exact per emotion; register tracks arousal). Instrumental backing is a
planned next stage.

> **Caveat carried by design.** Music renders valence/arousal strongly (ecstasy,
> grief, terror, rage separate cleanly) but *admiration/trust*, *vigilance*,
> *amazement/surprise* and *loathing/disgust* have weaker musical signatures and tend
> to blur in the MuLan checks — hence each emotion also carries valence/arousal
> coordinates as an interpretable, continuous fallback.

> **KNOWN ISSUE — MuLan text-tower lyric alignment (open, project-wide, not just
> this arm).** In a dry-run validation, MuLan's text tower correctly matched only
> 4/8 choruses to their target Plutchik emotion anchor; every miss stayed within the
> right *valence* band (e.g. terror/grief → loathing), never crossed it. Root cause:
> MuLan's text tower is trained on music *descriptions/tags* (MusicCaps-style), not
> lyrics — raw lyric text is out-of-distribution for it, and valence is a known
> stronger signal than fine-grained emotion in this kind of embedding (also reported
> for lyrics-only models, e.g. MoodyLyrics / Çano & Morisio 2017). This affects BOTH
> arms of the project (any `mulan_*` column that touches lyric text in
> `results/master_results.csv`, and the generative arm's anchor scoring), so the fix
> should be made once, centrally, rather than patched locally in `lmcgen`. Candidate
> fixes to evaluate (roughly cheapest → most involved): (1) prompt-ensemble the
> anchors (many templates, averaged — a well-established CLIP-style gain) and score
> by margin instead of raw cosine; (2) mean-center text/audio embeddings separately
> before scoring (the "modality gap" is a known geometric offset between towers);
> (3) add convergent, text-native lyric-emotion validators independent of MuLan —
> NRC EmoLex / NRC-VAD lexicon scoring (already used for the lexical check) and/or a
> GoEmotions-tuned classifier — and treat cross-method agreement as the evidence
> rather than MuLan cosine alone; (4) a small human rating pass (the gold standard
> every MER dataset — MoodyLyrics, DEAM, MERGE — ultimately relies on). **Deferred**
> pending a broader pass over how MuLan is used across the whole repo.

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
