# validation — LMC metric spot-tests

A self-contained sandbox for **comparing different ways of computing Lyric–Music
Congruence (LMC)** on a small, hand-picked set of songs, driven entirely from
`lmc_validation.ipynb`. It does **not** touch the main pipeline (`src/lmc`), the
metrics package (`src/lmcmetrics`), or the generative arm (`src/lmcgen`) — it
*reuses* their code and adds two new models plus a prompt/segmentation sweep.

## What it does

For a Spotify playlist (default: **5 covers of "Mad World"** — same lyrics, very
different arrangements), it computes `LMC = cosine(audio, text)` for every
combination of:

- **4 models** — MuQ-MuLan, LAION-CLAP (music), Microsoft CLAP (msclap), CLaMP 3
- **3 text prompts** — raw lyrics · `"a {song|song segment} that contains the lyrics …"`
  · `"a {song|song segment} representing the idea of the following lyrics: …"`
- **3 segmentation levels** — song-wide, segment-wide (chorus/verse), line-by-line

and writes three CSVs to `validation/results/`.

The point is **ordering, not p-values**: which covers / segments / lines does each
metric consider more vs. less congruent? "Mad World" is a good probe because a
metric that captures affective congruence should rank the sparse, melancholic
arrangements above the brighter ones despite identical lyrics.

## Outputs (`validation/results/`)

| file | one row per | columns |
|---|---|---|
| `song_wide.csv` | cover | `artist` + 12 score columns (`<model>__<prompt>`) = **4×3+1** |
| `segment_wide.csv` | (cover, segment) | `artist, segment_label, position_pct, n_lines` + 12 scores |
| `line_by_line.csv` | (cover, line) | `artist, line_index, position_pct, is_chorus, line_text` + 12 scores |

Score columns are `mulan__raw`, `mulan__contains`, …, `clamp3__idea`. A model that
can't be loaded gets `NaN` columns (the others still run).

## Setup

Run the **notebook** in the **`lmc` conda env** (it already has MuQ-MuLan, LAION-CLAP,
MERT, spotipy, yt-dlp, librosa). Two of the four models need one-time setup:

**Microsoft CLAP** — one line in the `lmc` env:
```bash
pip install msclap
```

**CLaMP 3** — a *separate* env (it pins `transformers==4.40.0` / `numpy==1.26.4`, which
would fight the `lmc` env), driven by our wrapper via subprocess. If you skip this, the
`clamp3__*` columns are just NaN and the other three models still run.
```bash
# 1. repo (already cloned to ~/Desktop/clamp3) — otherwise:
#    git clone https://github.com/sanderwood/clamp3 ~/Desktop/clamp3
# 2. dedicated env + deps (requirements.txt has NO torch, so add it — the Mac wheel has MPS):
conda create -n clamp3 python=3.10 -y
conda run -n clamp3 pip install torch torchaudio
conda run -n clamp3 pip install torchcodec         # REQUIRED — see note below
conda run -n clamp3 pip install -r ~/Desktop/clamp3/requirements.txt
# 3. the 2.57 GB audio checkpoint goes in ~/Desktop/clamp3/code/  (filename:
#    weights_clamp3_saas_h_size_768_..._p_length_512.pth)   [download handled separately]
# 4. tell the notebook where the env's python is (find it with `conda info --envs`):
#    set os.environ['LMCVAL_CLAMP3_PYTHON'] in the notebook's first cell.
```
CLaMP 3 uses `accelerate`'s device abstraction (no hard-coded CUDA), so it runs on
Apple Silicon via MPS/CPU — slowly, but fine for this 5-song spot test. On first audio
run it also downloads **MERT-v1-95M** from HuggingFace (~0.4 GB).

> **`torchcodec` is required, not optional, on newer torch/torchaudio pairs.**
> CLaMP 3's `requirements.txt` doesn't pin torch/torchaudio, and on recent versions
> `torchaudio.load()` needs the separate `torchcodec` package to read audio at all.
> Without it, CLaMP 3 fails *silently* — the outer script exits 0 (its own extractors
> catch every per-file error and just log it), so you get all-NaN `clamp3__*` columns
> with no visible error. Our wrapper now surfaces a warning if this happens (checking
> `code/**/logs/error_log.txt` and a "Found 0 files" message), but installing
> `torchcodec` up front avoids it entirely.

Spotify credentials (`SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET`) must be set to
read the playlist. Open `lmc_validation.ipynb` and run top to bottom.

**Offline plumbing check** (no downloads, no models, no creds):

```bash
cd validation && python -m lmcval.selftest        # fabricates audio + mock models
```

## How it reuses the main pipeline

- **audio download** — `lmc.audio._search_best` / `_score_candidate` (official-audio
  ranking) via `lmcval.acquire.download_audio`
- **lyrics** — per-cover synced lyrics from the **LRCLIB REST API** (the main
  pipeline reads the bulk dump; the API is simpler for a few known tracks), parsed
  with `lmc.utils.parse_lrc`
- **chorus/verse** — `lmc.chorus.detect_chorus`
- **audio slicing** — `lmc.embeddings._slice`
- **MuLan & LAION-CLAP** — the `lmc.embeddings._MuLan` / `_CLAP` embedders directly

## Package layout

```
validation/
  lmc_validation.ipynb    the notebook you run
  lmcval/
    config.py    playlist, models, the 3 prompts (+ optional extras), paths, knobs
    acquire.py   Spotify playlist -> LRCLIB synced lyrics -> yt-dlp audio -> Track objects
    models.py    the 4 model wrappers (uniform embed_audio_batch / embed_text_batch) + MockModel
    units.py     Track -> song / segment / line units (audio slice + lyric text + position)
    run.py       score every model×prompt×unit -> 3 CSVs + legible ranked summaries
    selftest.py  offline mock test of the compute + CSV plumbing
  data/          cached audio + .lrc + manifest        (gitignored)
  results/       the three output CSVs                 (gitignored)
```

## Prompt notes / recommendations

The three requested prompts are the default. `config.OPTIONAL_PROMPTS` holds a few
extras you can enable (see the notebook's optional cell):

- `mood` — `"a {unit} whose mood matches the lyrics …"` — pushes the caption-trained
  text towers toward *affect* rather than surface content, which is closer to the
  congruence construct.
- `about` — `"a {unit} about …"` — topical framing.
- `label` — `"lyrics: …"` — minimal framing, a lighter-touch alternative to raw text.

For the **line** level the longer templates can swamp a 5–8 word line; if line-level
scores look noisy, prefer `raw` and `mood` there.
