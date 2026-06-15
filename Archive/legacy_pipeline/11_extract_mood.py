"""
11_extract_mood.py — Extract audio features via librosa (no TensorFlow).

Replaces the Essentia TF approach which fails on Apple Silicon / Python 3.13.
Uses librosa-only signal processing to compute proxy scores for each mood
dimension. All output columns match the Stan model inputs exactly.

Features
--------
  mood_happy       : tempo × spectral brightness × major-mode proxy
  mood_sad         : slow tempo + minor mode + low energy
  mood_relaxed     : low ZCR + low energy + regular rhythm
  mood_aggressive  : high ZCR + loud + high spectral contrast
  mood_party       : high tempo + high energy + strong beats
  danceability     : beat clarity / rhythmic regularity
  voice_instrumental: harmonic-to-percussive ratio (low = vocal-heavy)

All values normalised to [0, 1].
"""

from __future__ import annotations
import os
import sys
import csv
import logging
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CATALOG, RESULTS_DIR
from utils import setup_logging

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

_audio_mod = _load("scrape_audio", "02_scrape_audio.py")
get_audio_path = _audio_mod.get_audio_path

setup_logging()
logger = logging.getLogger(__name__)

OUT_CSV = Path(RESULTS_DIR) / "essentia_mood.csv"

CSV_FIELDS = [
    "song_id", "artist_name", "genre", "orientation",
    "mood_happy", "mood_sad", "mood_relaxed",
    "mood_aggressive", "mood_party",
    "danceability", "voice_instrumental",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalise(x, lo, hi):
    """Clip and scale x from [lo, hi] to [0, 1]."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def extract_features(audio_path: str, sr: int = 22050) -> dict | None:
    """
    Compute acoustic proxy scores using librosa only.
    Returns dict with keys matching CSV_FIELDS mood columns.
    """
    try:
        import librosa
    except ImportError:
        logger.error("librosa not installed. Run: pip install librosa")
        sys.exit(1)

    try:
        # ── Load audio ──────────────────────────────────────────────────
        y, _ = librosa.load(audio_path, sr=sr, mono=True,
                             duration=120.0)   # first 2 min is enough

        if len(y) < sr * 5:
            logger.warning(f"  Audio too short: {len(y)/sr:.1f}s")
            return None

        # ── Raw acoustic measurements ────────────────────────────────────

        # 1. Tempo and beat strength
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        # Beat regularity: std dev of inter-beat intervals (lower = more regular)
        if len(beats) > 2:
            ibi      = np.diff(librosa.frames_to_time(beats, sr=sr))
            beat_reg = float(1.0 / (1.0 + np.std(ibi)))   # 0–1, higher = more regular
        else:
            beat_reg = 0.3

        # 2. RMS energy (loudness)
        rms      = float(np.mean(librosa.feature.rms(y=y)))
        rms_norm = normalise(rms, 0.01, 0.25)

        # 3. Spectral centroid (brightness — higher = brighter/happier)
        centroid      = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        centroid_norm = normalise(centroid, 500, 4000)

        # 4. Zero crossing rate (roughness / aggressiveness)
        zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        zcr_norm = normalise(zcr, 0.01, 0.20)

        # 5. Spectral contrast (variation between peaks and valleys)
        contrast      = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = float(np.mean(contrast))
        contrast_norm = normalise(contrast_mean, 10, 60)

        # 6. Harmonic vs percussive separation
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy  = float(np.mean(librosa.feature.rms(y=y_harm)))
        perc_energy  = float(np.mean(librosa.feature.rms(y=y_perc)))
        denom        = harm_energy + perc_energy + 1e-9
        # High harmonic ratio → more melodic/vocal content
        harm_ratio   = float(harm_energy / denom)
        perc_ratio   = float(perc_energy / denom)

        # 7. Chroma (key / mode estimation)
        chroma   = librosa.feature.chroma_cqt(y=y, sr=sr)
        # Rough major-vs-minor: major thirds (indices 4) vs minor thirds (indices 3)
        # This is a very rough proxy — positive = major-leaning
        major_score = float(np.mean(chroma[[0, 4, 7], :]))   # C, E, G
        minor_score = float(np.mean(chroma[[0, 3, 7], :]))   # C, Eb, G
        mode_proxy  = normalise(major_score - minor_score, -0.1, 0.1)

        # 8. Spectral rolloff (upper frequency content)
        rolloff      = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        rolloff_norm = normalise(rolloff, 1000, 8000)

        # 9. Tempo normalised (80–180 BPM typical range)
        tempo_norm = normalise(tempo, 60, 180)

        # 10. Dynamic complexity (variation in loudness)
        rms_frames    = librosa.feature.rms(y=y)[0]
        dynamic_range = float(np.std(rms_frames))
        dyn_norm      = normalise(dynamic_range, 0.01, 0.15)

        # ── Compose mood proxy scores ─────────────────────────────────────
        # Each score is a weighted linear combination, clipped to [0.05, 0.95]
        # to avoid boundary degeneration in Beta regression.

        mood_happy = np.clip(
            0.35 * centroid_norm +
            0.25 * mode_proxy    +
            0.20 * tempo_norm    +
            0.20 * rms_norm,
            0.05, 0.95
        )

        mood_sad = np.clip(
            0.35 * (1 - centroid_norm) +
            0.30 * (1 - mode_proxy)    +
            0.20 * (1 - tempo_norm)    +
            0.15 * (1 - rms_norm),
            0.05, 0.95
        )

        mood_relaxed = np.clip(
            0.35 * (1 - tempo_norm)   +
            0.30 * (1 - zcr_norm)     +
            0.20 * (1 - rms_norm)     +
            0.15 * beat_reg,
            0.05, 0.95
        )

        mood_aggressive = np.clip(
            0.35 * zcr_norm          +
            0.30 * rms_norm          +
            0.20 * (1 - mode_proxy)  +
            0.15 * contrast_norm,
            0.05, 0.95
        )

        mood_party = np.clip(
            0.35 * tempo_norm   +
            0.30 * rms_norm     +
            0.20 * beat_reg     +
            0.15 * perc_ratio,
            0.05, 0.95
        )

        # Danceability: beat regularity × tempo × percussive energy
        danceability = np.clip(
            0.50 * beat_reg    +
            0.30 * tempo_norm  +
            0.20 * perc_ratio,
            0.05, 0.95
        )

        # Voice/instrumental: high harmonic ratio + low percussive ratio
        # → more likely to be voice-heavy
        voice_instrumental = np.clip(
            0.60 * harm_ratio       +
            0.40 * (1 - perc_ratio),
            0.05, 0.95
        )

        return {
            "mood_happy":        round(float(mood_happy),        5),
            "mood_sad":          round(float(mood_sad),          5),
            "mood_relaxed":      round(float(mood_relaxed),      5),
            "mood_aggressive":   round(float(mood_aggressive),   5),
            "mood_party":        round(float(mood_party),        5),
            "danceability":      round(float(danceability),      5),
            "voice_instrumental": round(float(voice_instrumental), 5),
        }

    except Exception as e:
        logger.warning(f"  Feature extraction failed: {e}")
        return None


def extract_all_mood(force: bool = False,
                      artist_filter: list[str] | None = None) -> str:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if not force and OUT_CSV.exists():
        import pandas as pd
        existing = set(pd.read_csv(OUT_CSV)["song_id"].tolist())
        logger.info(f"Resuming — {len(existing)} songs already extracted.")

    to_process = []
    for artist_code, artist_data in CATALOG.items():
        if artist_filter and artist_code not in artist_filter:
            continue
        folder = artist_data["folder"]
        for song_id in artist_data["songs"]:
            if not force and song_id in existing:
                continue
            ap = get_audio_path(song_id, folder)
            if not ap:
                continue
            to_process.append((
                song_id, ap,
                artist_data["name"],
                artist_data["genre"],
                artist_data["orientation"],
            ))

    if not to_process:
        logger.info("Nothing to process.")
        return str(OUT_CSV)

    logger.info(f"Extracting features for {len(to_process)} songs (librosa, no TF)...")

    file_exists = OUT_CSV.exists() and not force
    with open(OUT_CSV, "a" if file_exists else "w",
              newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for i, (song_id, audio_path, artist, genre, orient) in enumerate(to_process, 1):
            logger.info(f"[{i}/{len(to_process)}] {song_id}")

            feats = extract_features(audio_path)

            if feats is None:
                logger.warning(f"  [{song_id}] Skipping")
                continue

            row = {
                "song_id":     song_id,
                "artist_name": artist,
                "genre":       genre,
                "orientation": orient,
                **feats,
            }
            writer.writerow(row)
            f.flush()

            if i % 10 == 0:
                logger.info(f"  Checkpoint ({i}/{len(to_process)} done)")

    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    mood_cols = [c for c in CSV_FIELDS
                 if c not in ("song_id", "artist_name", "genre", "orientation")]
    logger.info(f"\nDone: {len(df)} songs")
    for col in mood_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals):
            logger.info(f"  {col:25s}  mean={vals.mean():.3f}  sd={vals.std():.3f}")

    return str(OUT_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--artists", nargs="*")
    args = parser.parse_args()
    extract_all_mood(force=args.force, artist_filter=args.artists or None)