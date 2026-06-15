"""
10_lyric_timeline.py — Line-level Lyric-Music Congruence Timeline

Strategy
--------
  For each song:

  1.  Run WhisperX on the audio to get word-level timestamps.
  2.  Take the lyrics line by line (stripping section headers).
  3.  For each lyric line, walk the WhisperX word list sequentially to
      find the best-matching span of words — recording the audio start
      and end time of that span.
  4.  Compute position_pct: midpoint of the matched span as a percentage
      of total song duration.  This is the x-axis for time-series plots.
  5.  Extract an audio window of ±5 s around the matched span (clamped
      to song boundaries), embed it with MuQ-MuLan.
  6.  Embed the lyric line text with MuQ-MuLan.
  7.  Compute cosine similarity (LMC) between the two embeddings.
  8.  Write one row per lyric line to the output CSV.

Output CSV columns
------------------
  song_id, artist_name, genre, orientation,
  line_idx          — 0-based line index within the song
  line_text         — the raw lyric line
  line_words        — word count of the line
  match_start_s     — WhisperX-matched audio start (seconds)
  match_end_s       — WhisperX-matched audio end (seconds)
  window_start_s    — clip start after ±5 s padding (seconds)
  window_end_s      — clip end after ±5 s padding (seconds)
  position_pct      — midpoint / total_duration × 100 (0–100)
  lmc               — cosine similarity (MuQ-MuLan)
  match_confidence  — fraction of line words matched in WhisperX output
  total_duration_s  — full song length (seconds)

Modeling notes (see docstring at bottom of file)
"""

from __future__ import annotations
import os
import sys
import csv
import json
import re
import logging
import argparse
import numpy as np
import torch
import librosa
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CATALOG, RESULTS_DIR, MODELS
from utils import (
    setup_logging, get_device, model_to_device,
    strip_section_headers, split_text_chunks, cosine_sim,
)

import importlib.util

def _load(alias, filename):
    if alias in sys.modules:
        return sys.modules[alias]
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

_lyrics_mod = _load("scrape_lyrics", "01_scrape_lyrics.py")
_audio_mod  = _load("scrape_audio",  "02_scrape_audio.py")

load_all_lyrics = _lyrics_mod.load_all_lyrics
get_audio_path  = _audio_mod.get_audio_path

setup_logging()
logger = logging.getLogger(__name__)

TIMELINE_DIR = Path(RESULTS_DIR) / "lyric_timeline"
WINDOW_PAD_S = 5.0      # seconds of audio padding either side of matched span
MIN_LINE_WORDS = 3      # skip lines shorter than this (e.g. "Mmm", "Yeah")
MIN_MATCH_CONF = 0.25   # minimum word-match confidence to keep a line


# ── Text helpers ───────────────────────────────────────────────────────────────

def clean_lyrics_to_lines(lyrics_raw: str) -> list[str]:
    """
    Strip section headers and blank lines; return non-trivial lyric lines.
    Preserves original line order.
    """
    no_headers = strip_section_headers(lyrics_raw)
    lines = []
    for line in no_headers.splitlines():
        line = line.strip()
        # Skip empty lines, pure punctuation, and very short filler lines
        if len(line.split()) >= MIN_LINE_WORDS:
            lines.append(line)
    return lines


def normalise_word(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


# ── WhisperX ──────────────────────────────────────────────────────────────────

def load_whisperx(device: str):
    try:
        import whisperx
    except ImportError:
        logger.error("whisperx not installed. Run: pip install whisperx")
        sys.exit(1)

    wx_device    = "cpu" if device == "mps" else device
    compute_type = "int8" if wx_device == "cpu" else "float16"

    logger.info(f"Loading WhisperX on {wx_device}...")
    model = whisperx.load_model("base", device=wx_device, compute_type=compute_type)
    return model, wx_device


def transcribe_and_align(wx_model, audio_path: str,
                          wx_device: str) -> list[dict] | None:
    """
    Returns list of {"word": str, "start": float, "end": float}
    with word-level timestamps, or None on failure.
    """
    try:
        import whisperx
        audio  = whisperx.load_audio(audio_path)
        result = wx_model.transcribe(audio, batch_size=8)
        if not result.get("segments"):
            return None

        align_model, align_meta = whisperx.load_align_model(
            language_code = result.get("language", "en"),
            device        = wx_device,
        )
        aligned = whisperx.align(
            result["segments"], align_model, align_meta,
            audio, wx_device, return_char_alignments=False,
        )
        words = []
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []):
                if "start" in w and "end" in w and "word" in w:
                    words.append({
                        "word":  w["word"].strip(),
                        "start": float(w["start"]),
                        "end":   float(w["end"]),
                    })
        return words if words else None
    except Exception as e:
        logger.warning(f"  WhisperX failed: {e}")
        return None


# ── Sequential line matching ───────────────────────────────────────────────────

def match_line_to_words(
    line: str,
    wx_words: list[dict],
    cursor: int,
    search_radius: int = 80,
) -> tuple[float, float, float, int] | None:
    """
    Find the best contiguous span in wx_words[cursor : cursor+search_radius]
    that matches the words in `line`.

    Returns (start_s, end_s, match_confidence, new_cursor) or None.

    match_confidence = fraction of line words found in the matched span.
    new_cursor advances past the matched span so the next call starts there.
    """
    line_words = [normalise_word(w) for w in line.split() if normalise_word(w)]
    n_line     = len(line_words)
    n_wx       = len(wx_words)

    if n_line == 0:
        return None

    search_end = min(n_wx - n_line + 1, cursor + search_radius)
    if search_end <= cursor:
        return None

    best_pos  = cursor
    best_hits = 0

    for i in range(cursor, search_end):
        window = [normalise_word(wx_words[j]["word"])
                  for j in range(i, min(i + n_line, n_wx))]
        hits   = sum(lw == ww for lw, ww in zip(line_words, window))
        if hits > best_hits:
            best_hits = hits
            best_pos  = i

    confidence = best_hits / n_line
    end_pos    = min(best_pos + n_line - 1, n_wx - 1)
    start_s    = wx_words[best_pos]["start"]
    end_s      = wx_words[end_pos]["end"]
    new_cursor = end_pos + 1

    return start_s, end_s, confidence, new_cursor


# ── MuLan helpers ──────────────────────────────────────────────────────────────

def load_mulan(device: str):
    from muq import MuQMuLan
    logger.info("Loading MuQ-MuLan...")
    model = MuQMuLan.from_pretrained(MODELS["mulan"]["hf_id"])
    return model_to_device(model, device)


def embed_audio_window(model, wav: np.ndarray, sr: int,
                        start_s: float, end_s: float,
                        total_dur: float, device: str) -> np.ndarray | None:
    """
    Embed the audio window [start_s - PAD, end_s + PAD], clamped to [0, total_dur].
    Returns a [512] numpy array or None on failure.
    """
    try:
        win_start = max(0.0, start_s - WINDOW_PAD_S)
        win_end   = min(total_dur, end_s + WINDOW_PAD_S)

        s_i = int(win_start * sr)
        e_i = int(win_end   * sr)
        chunk = wav[s_i:e_i]

        if len(chunk) < int(0.5 * sr):   # < 0.5 s — too short
            return None

        wav_t = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(wavs=wav_t)
        return emb.squeeze(0).cpu().numpy()
    except Exception as e:
        logger.debug(f"  Audio window embed error: {e}")
        return None


def embed_text_line(model, line: str, device: str) -> np.ndarray | None:
    """
    Embed a single lyric line with MuLan.
    Falls back to chunk-averaging for unusually long lines.
    """
    try:
        chunks = split_text_chunks(line, max_words=200)
        vecs   = []
        for chunk in chunks:
            with torch.no_grad():
                emb = model(texts=[chunk])
            vecs.append(emb.squeeze(0).cpu().numpy())
        return np.stack(vecs).mean(axis=0)
    except Exception as e:
        logger.debug(f"  Text embed error: {e}")
        return None


# ── Core per-song analysis ────────────────────────────────────────────────────

def analyse_song_timeline(
    mulan, wx_model, wx_device: str,
    song_id: str, audio_path: str,
    lyrics_raw: str, song_meta: dict,
    device: str,
) -> list[dict]:
    """
    Run the full line-level timeline for one song.
    Returns a list of row dicts (one per matched lyric line).
    """
    lines = clean_lyrics_to_lines(lyrics_raw)
    if not lines:
        logger.info(f"  [{song_id}] No usable lyric lines — skipping")
        return []

    # Load audio
    sr = MODELS["mulan"]["audio_sr"]
    try:
        wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        logger.warning(f"  [{song_id}] Audio load failed: {e}")
        return []

    total_dur = len(wav) / sr

    # Run WhisperX
    wx_words = transcribe_and_align(wx_model, audio_path, wx_device)
    if not wx_words or len(wx_words) < 5:
        logger.info(f"  [{song_id}] WhisperX returned no words — skipping")
        return []

    logger.info(f"  [{song_id}] {len(lines)} lines, {len(wx_words)} WhisperX words")

    rows    = []
    cursor  = 0

    for line_idx, line in enumerate(lines):
        match = match_line_to_words(line, wx_words, cursor, search_radius=100)
        if match is None:
            continue

        start_s, end_s, confidence, cursor = match

        if confidence < MIN_MATCH_CONF:
            logger.debug(f"  Line {line_idx}: low confidence ({confidence:.2f}) — skipping")
            continue

        # Audio window with ±5 s padding
        win_start = max(0.0, start_s - WINDOW_PAD_S)
        win_end   = min(total_dur, end_s + WINDOW_PAD_S)
        midpoint  = (start_s + end_s) / 2.0
        pos_pct   = (midpoint / total_dur) * 100.0

        # Embeddings
        a_emb = embed_audio_window(mulan, wav, sr, start_s, end_s, total_dur, device)
        t_emb = embed_text_line(mulan, line, device)

        if a_emb is None or t_emb is None:
            continue

        lmc = cosine_sim(a_emb, t_emb)

        rows.append({
            # Identifiers
            "song_id":        song_id,
            "artist_name":    song_meta["artist_name"],
            "genre":          song_meta["genre"],
            "orientation":    song_meta["orientation"],
            # Line info
            "line_idx":       line_idx,
            "line_text":      line,
            "line_words":     len(line.split()),
            # Timing
            "match_start_s":  round(start_s, 3),
            "match_end_s":    round(end_s, 3),
            "window_start_s": round(win_start, 3),
            "window_end_s":   round(win_end, 3),
            "position_pct":   round(pos_pct, 3),
            "total_duration_s": round(total_dur, 3),
            # LMC
            "lmc":            round(lmc, 6),
            "match_confidence": round(confidence, 3),
        })

    logger.info(f"  [{song_id}] {len(rows)} lines matched and embedded")
    return rows


# ── Main loop ─────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "song_id", "artist_name", "genre", "orientation",
    "line_idx", "line_text", "line_words",
    "match_start_s", "match_end_s",
    "window_start_s", "window_end_s",
    "position_pct", "total_duration_s",
    "lmc", "match_confidence",
]


def run_lyric_timeline(
    device: str | None = None,
    force: bool = False,
    artist_filter: list[str] | None = None,
):
    if device is None:
        device = get_device()

    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv  = TIMELINE_DIR / "lyric_timeline.csv"
    done_log = TIMELINE_DIR / "done_songs.json"

    # Resume: load list of already-completed song IDs
    done: set[str] = set()
    if not force and done_log.exists():
        with open(done_log) as f:
            done = set(json.load(f))
        logger.info(f"Resuming — {len(done)} songs already complete")

    all_lyrics = load_all_lyrics()

    # Collect songs to process
    to_process = []
    for artist_code, artist_data in CATALOG.items():
        if artist_filter and artist_code not in artist_filter:
            continue
        folder = artist_data["folder"]
        for song_id in artist_data["songs"]:
            if not force and song_id in done:
                continue
            ap  = get_audio_path(song_id, folder)
            raw = all_lyrics.get(song_id, {}).get("lyrics", "")
            if not ap or not raw:
                continue
            to_process.append((
                song_id, ap, raw,
                {
                    "artist_name": artist_data["name"],
                    "genre":       artist_data["genre"],
                    "orientation": artist_data["orientation"],
                }
            ))

    if not to_process:
        logger.info("Nothing to process.")
        return

    logger.info(f"Processing {len(to_process)} songs...")

    # Load models once
    mulan            = load_mulan(device)
    wx_model, wx_dev = load_whisperx(device)

    # Open CSV in append mode so checkpoints work
    file_exists = out_csv.exists() and not force
    with open(out_csv, "a" if file_exists else "w",
              newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for i, (song_id, audio_path, lyrics_raw, meta) in enumerate(to_process, 1):
            logger.info(f"\n[{i}/{len(to_process)}] {song_id} — {meta['artist_name']}")

            rows = analyse_song_timeline(
                mulan, wx_model, wx_dev,
                song_id, audio_path, lyrics_raw, meta, device,
            )

            if rows:
                writer.writerows(rows)
                f.flush()

            done.add(song_id)

            # Save checkpoint every 5 songs
            if i % 5 == 0 or i == len(to_process):
                with open(done_log, "w") as dl:
                    json.dump(list(done), dl)
                logger.info(f"  Checkpoint saved ({i}/{len(to_process)} songs done)")

    # Summary
    import pandas as pd
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        logger.info(f"\n{'='*60}")
        logger.info(f"Timeline complete.")
        logger.info(f"  Rows    : {len(df):,}")
        logger.info(f"  Songs   : {df['song_id'].nunique()}")
        logger.info(f"  LMC mean: {df['lmc'].mean():.4f}")
        logger.info(f"  LMC sd  : {df['lmc'].std():.4f}")
        logger.info(f"  Output  : {out_csv}")

    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line-level lyric timeline")
    parser.add_argument("--device",  default=None)
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--artists", nargs="*")
    args = parser.parse_args()

    run_lyric_timeline(
        device        = args.device,
        force         = args.force,
        artist_filter = args.artists or None,
    )


# =============================================================================
# MODELING NOTES
# =============================================================================
#
# The lyric_timeline.csv gives you one observation per lyric line, with
# position_pct as the continuous time axis and lmc as the outcome. The
# key challenge is that observations within a song are not independent —
# they share an artist, a genre, and the song-level compositional choices
# that set the baseline LMC. There are also variable numbers of lines per
# song, and the spacing between lines is irregular. Below are five modeling
# approaches roughly in order of analytical sophistication.
#
#
# 1. LOESS SMOOTHING PER SONG (visualisation, no inference)
# ----------------------------------------------------------
# The simplest starting point. For each song, fit a LOESS curve of
# lmc ~ position_pct. Overlay all songs by genre or orientation to see
# whether there are consistent arc shapes (e.g. does the chorus tend to
# produce an LMC spike near 30–40% and 70–80%?). In R:
#
#   ggplot(df, aes(x = position_pct, y = lmc, group = song_id)) +
#     geom_smooth(method = "loess", se = FALSE, alpha = 0.3) +
#     facet_wrap(~genre_cluster)
#
# This is purely descriptive but surprisingly informative. You can see
# whether the temporal pattern of congruence is systematic or noisy.
#
#
# 2. FUNCTIONAL DATA ANALYSIS (FDA)
# -----------------------------------
# Each song's lmc-over-time trajectory is a function, not a vector of
# scalars. FDA treats those functions as the unit of observation. The
# approach:
#   a. Interpolate each song's lmc(position_pct) onto a common grid
#      (e.g. 0, 1, 2, ..., 100 percent) using spline smoothing.
#   b. Run functional PCA (fpca) to find the dominant "shapes" of
#      LMC-over-time variation — e.g. PC1 might be overall level,
#      PC2 might be "rises in the second half".
#   c. Regress the fPCA scores on popularity, genre, orientation.
#
# In R, the `fda` and `fdapace` packages handle this. This is the most
# principled approach for the time-series question but requires a minimum
# of ~15–20 data points per song to get a stable smooth, which should be
# achievable with line-level data.
#
#
# 3. HIERARCHICAL GAM (recommended middle ground)
# ------------------------------------------------
# A Generalised Additive Mixed Model fits a smooth nonlinear function of
# position_pct while respecting the nested data structure (lines within
# songs, songs within artists). In R with `mgcv`:
#
#   library(mgcv)
#   m <- bam(
#     lmc ~ s(position_pct, k=10) +
#           s(position_pct, by = orientation, k=10) +
#           s(song_id, bs="re") +
#           s(artist_code, bs="re"),
#     data   = df,
#     method = "fREML"
#   )
#
# The `s(position_pct)` term captures the average LMC arc over the course
# of a song. The `by=orientation` interaction tests whether narrative vs.
# production songs have different temporal profiles. The random effects
# partial out song- and artist-level baseline differences.
#
# To test whether the *shape* of the arc predicts popularity, extract
# per-song fitted values from the model and correlate them with the
# master_results popularity score. Alternatively, include popularity as a
# moderator of the smooth:
#
#   m2 <- bam(
#     lmc ~ s(position_pct, k=10) +
#           s(position_pct, by = popularity_z, k=10) +
#           s(song_id, bs="re"),
#     data = df %>% left_join(master_results, by="song_id")
#   )
#
# A significant `s(position_pct, by=popularity_z)` term means that more
# popular songs have a systematically different temporal LMC profile.
#
#
# 4. CHORUS DETECTION VIA LMC PEAKS
# ------------------------------------
# An indirect test of the chorus-LMC hypothesis without requiring section
# labels: identify local maxima in each song's smoothed lmc curve and
# test whether those peaks cluster at positions consistent with typical
# chorus placement (roughly 25–35% and 60–70% for a standard
# verse-chorus-verse-chorus structure). In R:
#
#   library(pracma)
#   peaks <- df %>%
#     group_by(song_id) %>%
#     arrange(position_pct) %>%
#     mutate(smooth_lmc = predict(loess(lmc ~ position_pct, span=0.3))) %>%
#     summarise(peak_positions = list(findpeaks(smooth_lmc)[,"position"]))
#
# Then test whether mean peak position differs by genre or correlates
# with popularity. This is speculative but produces a clean, interpretable
# result for a paper figure.
#
#
# 5. CHANGE-POINT MODEL
# ----------------------
# If you hypothesise that LMC doesn't change gradually but shifts
# abruptly at structural boundaries (verse → chorus), a change-point
# model is appropriate. The `mcp` package in R fits Bayesian multiple
# change-point models with mixed effects:
#
#   library(mcp)
#   model <- list(
#     lmc ~ 1,                    # segment 1: intercept
#     ~ 1,                        # segment 2: new intercept after change
#     ~ 1                         # segment 3: another change
#   )
#   fit <- mcp(model, data = one_song_df,
#              par_x = "position_pct")
#
# Run this per song and extract the inferred change-point locations.
# Then test whether those locations predict section boundaries when
# section labels are available, and whether change-point magnitude
# (size of LMC jump) correlates with popularity.
#
# =============================================================================
