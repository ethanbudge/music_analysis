"""
mood.py — librosa-based acoustic mood proxies (kept from the original pipeline).

These are the audio "mood calculators": cheap, TensorFlow-free signal-processing
proxies for the standard mood dimensions, all normalised to [0.05, 0.95] so they
behave well as Beta-regression controls in the Stan models. Values are cached per
song in the project DB; already-computed songs are skipped.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone

import numpy as np

from . import db as projdb

logger = logging.getLogger(__name__)


def _normalise(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def extract_features(audio_path: str, sr: int = 22_050) -> dict | None:
    """Compute librosa mood/acoustic proxies for one audio file (first 2 min)."""
    import librosa
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=120.0)
        if len(y) < sr * 5:
            logger.warning("  audio too short: %.1fs", len(y) / sr)
            return None

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        if len(beats) > 2:
            ibi = np.diff(librosa.frames_to_time(beats, sr=sr))
            beat_reg = float(1.0 / (1.0 + np.std(ibi)))
        else:
            beat_reg = 0.3

        rms      = float(np.mean(librosa.feature.rms(y=y)));                 rms_n = _normalise(rms, 0.01, 0.25)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))); cen_n = _normalise(centroid, 500, 4000)
        zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y=y)));   zcr_n = _normalise(zcr, 0.01, 0.20)
        contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))); con_n = _normalise(contrast, 10, 60)

        y_h, y_p = librosa.effects.hpss(y)
        h_e = float(np.mean(librosa.feature.rms(y=y_h)))
        p_e = float(np.mean(librosa.feature.rms(y=y_p)))
        denom = h_e + p_e + 1e-9
        harm_ratio, perc_ratio = h_e / denom, p_e / denom

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        major  = float(np.mean(chroma[[0, 4, 7], :]))
        minor  = float(np.mean(chroma[[0, 3, 7], :]))
        mode_p = _normalise(major - minor, -0.1, 0.1)
        tempo_n = _normalise(tempo, 60, 180)

        def clip(v): return round(float(np.clip(v, 0.05, 0.95)), 5)
        return {
            "mood_happy":      clip(0.35*cen_n + 0.25*mode_p + 0.20*tempo_n + 0.20*rms_n),
            "mood_sad":        clip(0.35*(1-cen_n) + 0.30*(1-mode_p) + 0.20*(1-tempo_n) + 0.15*(1-rms_n)),
            "mood_relaxed":    clip(0.35*(1-tempo_n) + 0.30*(1-zcr_n) + 0.20*(1-rms_n) + 0.15*beat_reg),
            "mood_aggressive": clip(0.35*zcr_n + 0.30*rms_n + 0.20*(1-mode_p) + 0.15*con_n),
            "mood_party":      clip(0.35*tempo_n + 0.30*rms_n + 0.20*beat_reg + 0.15*perc_ratio),
            "danceability":    clip(0.50*beat_reg + 0.30*tempo_n + 0.20*perc_ratio),
            "voice_instrumental": clip(0.60*harm_ratio + 0.40*(1-perc_ratio)),
        }
    except Exception as e:                                     # noqa: BLE001
        logger.warning("  mood extraction failed: %s", e)
        return None


def extract_pending(limit: int | None = None, force: bool = False) -> dict:
    """Extract mood features for songs that have audio but no mood row yet."""
    with projdb.connect() as conn:
        have_audio = projdb.songs_with_audio(conn)
        done = set() if force else {r["track_id"] for r in conn.execute("SELECT track_id FROM mood")}
    todo = [dict(r) for r in have_audio if force or r["track_id"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        logger.info("Mood: nothing pending.")
        return {"attempted": 0, "done": 0}

    logger.info("Mood: %d songs to process.", len(todo))
    ok = 0
    for i, song in enumerate(todo, 1):
        logger.info("[%d/%d] %s", i, len(todo), song["track_id"])
        feats = extract_features(song["file_path"])
        if feats is None:
            continue
        feats["track_id"] = song["track_id"]
        feats["fetched_at"] = datetime.now(timezone.utc).isoformat()
        with projdb.connect() as conn:
            projdb.upsert(conn, "mood", feats)
        ok += 1
    logger.info("Mood done: %d songs.", ok)
    return {"attempted": len(todo), "done": ok}
