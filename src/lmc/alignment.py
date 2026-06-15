"""
alignment.py — Compute Lyric-Music Congruence (LMC) from cached embeddings.

LMC is the cosine similarity between an audio embedding and the corresponding
lyric-text embedding in a joint space. From each cached bundle we derive:

  Song-wide          whole-song audio vs whole-lyrics text.
  Line-level         per synced line, audio-window vs line text, for four
                     context windows — exact (0 s), ±1 s, ±5 s, ±10 s —
                     summarised per song as the mean over lines.
  Segment-wide       chorus vs non-chorus: each group's concatenated audio vs
                     its joined text.

Two tables are written:
  lmc        one row per (track, model, method) — song-level scalars used by the
             master table and the track/segment Stan models.
  lmc_lines  one row per (track, model, line, window) — the line-level series
             (with position_pct and chorus flag) used by the timeline analysis
             and the line Stan model.

Resumable: a (track, model) pair is skipped if it already has lmc rows, unless
force=True.
"""

from __future__ import annotations
import json
import logging

import numpy as np

from .config import MODELS, EMBEDDINGS_DIR, CONTEXT_WINDOWS
from .utils import cosine_sim, parse_lrc, load_song_embeddings, embedding_path
from . import db as projdb
from . import chorus as chorus_mod

logger = logging.getLogger(__name__)


def _cos_or_none(a, b):
    if a is None or b is None or not np.any(a) or not np.any(b):
        return None
    return cosine_sim(a, b)


def _song_methods(bundle, lines, flags, duration) -> tuple[dict, list[dict]]:
    """Return (song-level method→value, list of line-level rows) for one bundle."""
    methods: dict[str, float] = {}

    # Song-wide.
    s = _cos_or_none(bundle.get("audio_full"), bundle.get("text_full"))
    if s is not None:
        methods["song"] = s

    # Segment-wide.
    for label in ("chorus", "nonchorus"):
        v = _cos_or_none(bundle.get(f"{label}_audio"), bundle.get(f"{label}_text"))
        if v is not None:
            methods[f"seg_{label}"] = v

    # Line-level windows.
    line_text = bundle.get("line_text")
    line_rows: list[dict] = []
    L = len(lines)
    dur = duration or (lines[-1]["start"] + 1 if lines else 1)
    for w in CONTEXT_WINDOWS:
        audio_w = bundle.get(f"audio_{w}")
        if line_text is None or audio_w is None:
            continue
        vals = []
        for i in range(min(L, len(line_text), len(audio_w))):
            c = _cos_or_none(audio_w[i], line_text[i])
            if c is None:
                continue
            vals.append(c)
            ln = lines[i]
            mid = (ln["start"] + (ln["end"] if ln["end"] is not None else ln["start"])) / 2.0
            line_rows.append({
                "line_idx": i, "window": w,
                "position_pct": round(100.0 * mid / dur, 3) if dur else None,
                "is_chorus": int(bool(flags[i])) if i < len(flags) else 0,
                "lmc": round(c, 6),
            })
        if vals:
            methods[f"line_{w}"] = float(np.mean(vals))
    return methods, line_rows


def compute_pending(models: list[str] | None = None, force: bool = False,
                    limit: int | None = None) -> dict:
    """Compute LMC for every (song, model) bundle that hasn't been scored yet."""
    models = models or list(MODELS.keys())

    with projdb.connect() as conn:
        songs = {r["track_id"]: dict(r) for r in conn.execute(
            "SELECT track_id, duration, synced_lyrics FROM songs")}
        scored = {(r["track_id"], r["model"]) for r in conn.execute(
            "SELECT DISTINCT track_id, model FROM lmc")}

    total_done = 0
    for model_key in models:
        todo = []
        for tid in songs:
            if not force and (tid, model_key) in scored:
                continue
            if embedding_path(EMBEDDINGS_DIR, model_key, tid).exists():
                todo.append(tid)
        if limit:
            todo = todo[:limit]
        if not todo:
            logger.info("LMC[%s]: nothing pending.", model_key)
            continue

        logger.info("LMC[%s]: scoring %d songs.", model_key, len(todo))
        for n, tid in enumerate(todo, 1):
            bundle = load_song_embeddings(embedding_path(EMBEDDINGS_DIR, model_key, tid))
            if bundle is None:
                continue
            song = songs[tid]
            lines = parse_lrc(song["synced_lyrics"] or "")
            flags = chorus_mod.get_flags(tid) or [False] * len(lines)
            methods, line_rows = _song_methods(bundle, lines, flags, song["duration"])

            with projdb.connect() as conn:
                for method, value in methods.items():
                    projdb.upsert(conn, "lmc", {
                        "track_id": tid, "model": model_key,
                        "method": method, "value": round(float(value), 6)})
                for lr in line_rows:
                    projdb.upsert(conn, "lmc_lines", {
                        "track_id": tid, "model": model_key, **lr})
            total_done += 1
            if n % 25 == 0:
                logger.info("  [%s] %d/%d", model_key, n, len(todo))

        logger.info("LMC[%s] done.", model_key)

    return {"scored": total_done}
