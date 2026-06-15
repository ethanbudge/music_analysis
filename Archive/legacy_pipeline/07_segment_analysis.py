"""
07_segment_analysis.py — Segment-level LMC using WhisperX forced alignment.

Strategy
--------
  1. Parse Genius lyrics into sections ([Verse], [Chorus], etc.)
  2. Run WhisperX on the audio to get word-level timestamps via
     wav2vec2 forced alignment.
  3. Sequentially match each lyric section's words to the WhisperX
     word timestamps to find precise start/end times per section.
  4. Slice the audio at those boundaries (not equal chunks).
  5. Compute MuQ-MuLan embeddings for each audio slice + lyric text.

Why sequential matching?
-------------------------
  Choruses repeat, so naive word search would always find the first
  occurrence. Walking through sections in order and consuming matched
  words prevents this.

Fallback
--------
  If WhisperX fails (e.g. no speech detected, instrumental) or fewer
  than min_sections are found, the song is skipped gracefully.

Outputs (results/segment_analysis/)
-------
  segment_details.json   — per song, per section boundaries + LMC
  segment_summary.csv    — one row per song with summary stats
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
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent))

from config import CATALOG, AUDIO_BASE_DIR, RESULTS_DIR, MODELS, SEGMENT_CONFIG
from utils import (
    setup_logging, get_device, model_to_device,
    parse_lyric_sections, split_text_chunks, cosine_sim,
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

_lyrics = _load("scrape_lyrics", "01_scrape_lyrics.py")
_audio  = _load("scrape_audio",  "02_scrape_audio.py")

load_all_lyrics = _lyrics.load_all_lyrics
get_audio_path  = _audio.get_audio_path

setup_logging()
logger = logging.getLogger(__name__)

SEG_DIR = Path(RESULTS_DIR) / "segment_analysis"


# ── WhisperX alignment ────────────────────────────────────────────────────────

def load_whisperx(device: str):
    """Load WhisperX model. Uses float32/int8 on CPU for MPS stability."""
    try:
        import whisperx
    except ImportError:
        logger.error("whisperx not installed. Run: pip install whisperx")
        sys.exit(1)

    # WhisperX alignment model works most reliably on CPU for MPS machines
    wx_device       = "cpu" if device == "mps" else device
    compute_type    = "int8" if wx_device == "cpu" else "float16"

    logger.info(f"Loading WhisperX (whisper=base, align on {wx_device})...")
    model = whisperx.load_model(
        "base",
        device       = wx_device,
        compute_type = compute_type,
    )
    return model, wx_device


def transcribe_and_align(wx_model, audio_path: str,
                          wx_device: str) -> list[dict] | None:
    """
    Transcribe audio with WhisperX and return word-level timestamps.

    Returns list of {"word": str, "start": float, "end": float}
    or None on failure.
    """
    try:
        import whisperx

        # Load audio at 16kHz (WhisperX requirement)
        audio = whisperx.load_audio(audio_path)

        # Transcribe
        result = wx_model.transcribe(audio, batch_size=8)
        if not result.get("segments"):
            logger.warning("  WhisperX: no segments found (instrumental?)")
            return None

        # Align to get word-level timestamps
        align_model, align_meta = whisperx.load_align_model(
            language_code = result.get("language", "en"),
            device        = wx_device,
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            align_meta,
            audio,
            wx_device,
            return_char_alignments=False,
        )

        # Flatten all word entries
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
        logger.warning(f"  WhisperX alignment failed: {e}")
        return None


# ── Word matching ─────────────────────────────────────────────────────────────

def _normalise_word(w: str) -> str:
    """Lowercase and strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9']", "", w.lower())


def match_sections_to_timestamps(
    sections: list[dict],
    wx_words: list[dict],
    min_match_ratio: float = 0.35,
) -> list[tuple[float, float]] | None:
    """
    Sequentially match lyric sections to WhisperX word timestamps.

    For each section, walks forward through the remaining WhisperX word
    list looking for the best contiguous span of matching words.

    Returns list of (start_s, end_s) tuples, one per section, or None
    if matching fails badly across the whole song.
    """
    wx_norm   = [_normalise_word(w["word"]) for w in wx_words]
    cursor    = 0   # current position in wx_words
    boundaries = []

    for section in sections:
        section_words = [_normalise_word(w)
                         for w in section["text"].split()
                         if _normalise_word(w)]
        if not section_words:
            boundaries.append(None)
            continue

        n_sec    = len(section_words)
        n_wx     = len(wx_words)
        best_pos = cursor
        best_hits = 0

        # Sliding window search from cursor onward
        search_limit = min(n_wx, cursor + max(n_sec * 6, 60))
        for i in range(cursor, search_limit - min(n_sec, 5) + 1):
            window = wx_norm[i : i + n_sec]
            hits   = sum(sw == ww for sw, ww in zip(section_words, window))
            if hits > best_hits:
                best_hits = hits
                best_pos  = i

        match_ratio = best_hits / max(n_sec, 1)

        if match_ratio < min_match_ratio:
            # Poor match — estimate position from song progress
            logger.debug(f"  Low match ({match_ratio:.2f}) for '{section['header']}'"
                         f" — using position estimate")
            # Use a reasonable position estimate rather than failing
            end_pos  = min(cursor + n_sec, n_wx - 1)
            start_s  = wx_words[cursor]["start"] if cursor < n_wx else wx_words[-1]["end"]
            end_s    = wx_words[end_pos]["end"]  if end_pos < n_wx else wx_words[-1]["end"]
            boundaries.append((start_s, end_s))
            cursor = end_pos
        else:
            end_pos = min(best_pos + n_sec - 1, n_wx - 1)
            start_s = wx_words[best_pos]["start"]
            end_s   = wx_words[end_pos]["end"]
            boundaries.append((start_s, end_s))
            cursor  = end_pos + 1

    if all(b is None for b in boundaries):
        return None

    # Fill None entries with interpolated times
    total_dur = wx_words[-1]["end"] if wx_words else 0
    n = len(boundaries)
    filled = []
    for i, b in enumerate(boundaries):
        if b is not None:
            filled.append(b)
        else:
            prev_end = filled[-1][1] if filled else 0.0
            next_start = next(
                (boundaries[j][0] for j in range(i+1, n) if boundaries[j] is not None),
                total_dur
            )
            mid = (prev_end + next_start) / 2
            filled.append((prev_end, mid))

    return filled


# ── MuLan helpers ─────────────────────────────────────────────────────────────

def load_mulan(device: str):
    from muq import MuQMuLan
    model = MuQMuLan.from_pretrained(MODELS["mulan"]["hf_id"])
    return model_to_device(model, device)


def embed_audio_slice(model, wav: np.ndarray, sr: int,
                      start_s: float, end_s: float,
                      device: str) -> np.ndarray | None:
    """Extract and embed a slice of audio between start_s and end_s."""
    try:
        start_i = int(start_s * sr)
        end_i   = min(int(end_s * sr), len(wav))
        chunk   = wav[start_i:end_i]

        if len(chunk) < int(SEGMENT_CONFIG["min_segment_secs"] * sr):
            return None

        wav_t = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(wavs=wav_t)
        return emb.squeeze(0).cpu().numpy()
    except Exception as e:
        logger.debug(f"  Audio slice embed error: {e}")
        return None


def embed_text(model, text: str, device: str,
               max_words: int = 200) -> np.ndarray | None:
    chunks = split_text_chunks(text, max_words)
    vecs = []
    for chunk in chunks:
        try:
            with torch.no_grad():
                emb = model(texts=[chunk])
            vecs.append(emb.squeeze(0).cpu().numpy())
        except Exception as e:
            logger.debug(f"  Text chunk embed error: {e}")
    if not vecs:
        return None
    return np.stack(vecs).mean(axis=0)


# ── Core per-song analysis ────────────────────────────────────────────────────

class SegmentResult(NamedTuple):
    song_id:      str
    section_idx:  int
    header:       str
    section_type: str
    start_s:      float
    end_s:        float
    duration_s:   float
    lmc:          float
    text_len:     int
    match_method: str   # "whisperx" or "fallback_equal"


def analyse_song(mulan_model, wx_model, wx_device: str,
                 song_id: str, audio_path: str, lyrics_raw: str,
                 device: str) -> list[SegmentResult] | None:
    """
    Analyse one song at the segment level using WhisperX alignment.
    Falls back to equal-duration splits if alignment fails.
    """
    sections = parse_lyric_sections(lyrics_raw, min_words=5)
    n_sections = len(sections)

    if n_sections < SEGMENT_CONFIG["min_sections"]:
        logger.info(f"  [{song_id}] Only {n_sections} section(s) — skipping")
        return None

    # Load audio at MuLan sample rate
    sr = MODELS["mulan"]["audio_sr"]
    try:
        wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        logger.warning(f"  [{song_id}] Audio load failed: {e}")
        return None

    total_dur = len(wav) / sr

    # ── Try WhisperX alignment ────────────────────────────────────────────
    method    = "whisperx"
    wx_words  = transcribe_and_align(wx_model, audio_path, wx_device)
    boundaries = None

    if wx_words and len(wx_words) >= 5:
        boundaries = match_sections_to_timestamps(sections, wx_words)
        if boundaries is None:
            logger.info(f"  [{song_id}] WhisperX matching failed — using equal splits")
    else:
        logger.info(f"  [{song_id}] WhisperX returned no words — using equal splits")

    # ── Fallback: equal-duration splits ──────────────────────────────────
    if boundaries is None:
        method    = "fallback_equal"
        chunk_dur = total_dur / n_sections
        boundaries = [
            (i * chunk_dur, (i + 1) * chunk_dur)
            for i in range(n_sections)
        ]

    # ── Embed each section ────────────────────────────────────────────────
    results = []
    for idx, (section, (start_s, end_s)) in enumerate(zip(sections, boundaries)):
        a_emb = embed_audio_slice(mulan_model, wav, sr, start_s, end_s, device)
        t_emb = embed_text(mulan_model, section["text"], device)

        if a_emb is None or t_emb is None:
            logger.debug(f"  [{song_id}] Section {idx} embed failed — skipping")
            continue

        lmc = cosine_sim(a_emb, t_emb)
        results.append(SegmentResult(
            song_id      = song_id,
            section_idx  = idx,
            header       = section["header"],
            section_type = section["section_type"],
            start_s      = round(start_s, 3),
            end_s        = round(end_s, 3),
            duration_s   = round(end_s - start_s, 3),
            lmc          = lmc,
            text_len     = len(section["text"].split()),
            match_method = method,
        ))

    return results if results else None


# ── Summary builder ───────────────────────────────────────────────────────────

def build_summary(all_results: dict[str, list[SegmentResult]]) -> list[dict]:
    rows = []
    for song_id, segs in all_results.items():
        lmcs   = [s.lmc for s in segs]
        by_type = {}
        for s in segs:
            by_type.setdefault(s.section_type, []).append(s.lmc)

        def mean_type(t):
            vals = by_type.get(t, [])
            return float(np.mean(vals)) if vals else None

        wx_count  = sum(1 for s in segs if s.match_method == "whisperx")
        method    = "whisperx" if wx_count > len(segs) // 2 else "fallback_equal"

        rows.append({
            "song_id":          song_id,
            "n_segments":       len(segs),
            "alignment_method": method,
            "mean_lmc_all":     float(np.mean(lmcs)),
            "sd_lmc":           float(np.std(lmcs)),
            "max_lmc":          float(np.max(lmcs)),
            "min_lmc":          float(np.min(lmcs)),
            "mean_lmc_verse":   mean_type("verse"),
            "mean_lmc_chorus":  mean_type("chorus"),
            "mean_lmc_bridge":  mean_type("bridge"),
            "mean_lmc_intro":   mean_type("intro"),
            "mean_lmc_outro":   mean_type("outro"),
        })
    return rows


# ── Main entry point ──────────────────────────────────────────────────────────

def run_segment_analysis(device: str | None = None, force: bool = False,
                          artist_filter: list[str] | None = None):
    if not SEGMENT_CONFIG.get("enabled", True):
        logger.info("Segment analysis disabled in config.")
        return {}, []

    if device is None:
        device = get_device()

    SEG_DIR.mkdir(parents=True, exist_ok=True)

    detail_path  = SEG_DIR / "segment_details.json"
    summary_path = SEG_DIR / "segment_summary.csv"

    all_details: dict[str, list] = {}
    if not force and detail_path.exists():
        with open(detail_path) as f:
            all_details = json.load(f)
        logger.info(f"Resuming — {len(all_details)} songs already done.")

    all_lyrics = load_all_lyrics()

    to_process = []
    for artist_code, artist_data in CATALOG.items():
        if artist_filter and artist_code not in artist_filter:
            continue
        folder = artist_data["folder"]
        for song_id in artist_data["songs"]:
            if not force and song_id in all_details:
                continue
            ap  = get_audio_path(song_id, folder)
            raw = all_lyrics.get(song_id, {}).get("lyrics", "")
            if not ap or not raw:
                continue
            to_process.append((song_id, ap, raw))

    if not to_process:
        logger.info("All songs already analysed or missing data.")
    else:
        logger.info(f"Loading models for {len(to_process)} songs...")
        mulan_model      = load_mulan(device)
        wx_model, wx_dev = load_whisperx(device)

        for i, (song_id, audio_path, lyrics_raw) in enumerate(to_process, 1):
            logger.info(f"[{i}/{len(to_process)}] {song_id}")
            segs = analyse_song(
                mulan_model, wx_model, wx_dev,
                song_id, audio_path, lyrics_raw, device,
            )

            if segs:
                all_details[song_id] = [s._asdict() for s in segs]
                mean_lmc = np.mean([s.lmc for s in segs])
                methods  = set(s.match_method for s in segs)
                logger.info(f"  ✓ {len(segs)} sections | "
                            f"mean LMC={mean_lmc:.4f} | method={methods}")
            else:
                all_details[song_id] = []

            if i % 5 == 0 or i == len(to_process):
                with open(detail_path, "w") as f:
                    json.dump(all_details, f, indent=2)
                logger.info("  Checkpoint saved.")

    with open(detail_path, "w") as f:
        json.dump(all_details, f, indent=2)

    valid = {k: [SegmentResult(**s) for s in v]
             for k, v in all_details.items() if v}
    summary_rows = build_summary(valid)

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        wx_count  = sum(1 for r in summary_rows if r["alignment_method"] == "whisperx")
        fb_count  = len(summary_rows) - wx_count
        logger.info(f"\nSegment summary: {len(summary_rows)} songs "
                    f"({wx_count} WhisperX-aligned, {fb_count} equal-split fallback)")
        logger.info(f"→ {summary_path}")

    return all_details, summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment-level LMC (WhisperX)")
    parser.add_argument("--device",  default=None)
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--artists", nargs="*")
    args = parser.parse_args()

    run_segment_analysis(
        device        = args.device,
        force         = args.force,
        artist_filter = args.artists or None,
    )