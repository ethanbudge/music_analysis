"""
gather.py — one call to advance stages 1-4 of notebooks/pipeline.ipynb
(setup/sample → audio → popularity → mood/MERT/chorus) for a fixed-size batch
of songs, so the whole gathering pipeline can be driven by a single command
instead of stepping through the notebook.

Every underlying stage is already idempotent and resumable (see db.py), so
`run_batch` is just `lrclib.sample` followed by the three enrichment stages,
each capped to `batch_size` pending items -- calling it repeatedly grows the
corpus without redoing finished work.
"""

from __future__ import annotations
import logging

from . import config, lrclib, audio, popularity, mood, mert, chorus, db

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


def run_batch(batch_size: int = DEFAULT_BATCH_SIZE, seed: int | None = None) -> dict:
    """
    Sample up to `batch_size` new songs, then run audio download, popularity
    enrichment, and the mood/MERT/chorus acoustic controls -- each capped to
    `batch_size` pending items so one call does one bounded unit of work.
    """
    config.ensure_dirs()
    before = lrclib.setup()
    new_ids = lrclib.sample(batch_size, seed=seed)

    audio_result = audio.download_pending(limit=batch_size)
    popularity_result = popularity.fetch_pending(limit=batch_size)
    mood_result = mood.extract_pending(limit=batch_size)
    mert_result = mert.extract_pending(limit=batch_size)
    chorus_result = chorus.compute_pending()

    progress = db.progress_report()
    logger.info(
        "Batch complete — %d new songs sampled (corpus now %d). Progress: %s",
        len(new_ids), progress["sampled"], progress,
    )
    return {
        "new_songs":  len(new_ids),
        "before":     before,
        "audio":      audio_result,
        "popularity": popularity_result,
        "mood":       mood_result,
        "mert":       mert_result,
        "chorus":     chorus_result,
        "progress":   progress,
    }
