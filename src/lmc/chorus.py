"""
chorus.py — Algorithmic chorus detection from time-synced lyric lines.

The segment-level LMC reduces each song to two groups — chorus vs. non-chorus —
so we need to label each synced line. We use a lyrics-repetition method, which
is well suited to LRCLIB data (no audio required, aligned to the synced
timestamps):

  1. Normalise each line (lowercase, strip punctuation).
  2. A line is "recurrent" if its normalised text appears ≥ 2 times.
  3. Find contiguous runs of recurrent lines (length ≥ MIN_BLOCK). The run text
     that recurs most often (and at least twice) is the chorus signature.
  4. Flag every line belonging to any occurrence of that signature as chorus.
  5. Fallback: if no multi-line block recurs, flag the single most-repeated line
     (the hook), provided it repeats at least twice.

This is intentionally simple and transparent (documented as a limitation). An
audio self-similarity matrix is a natural future alternative; see README.
"""

from __future__ import annotations
import json
import logging
from collections import Counter
from datetime import datetime, timezone

from .utils import parse_lrc, normalise_line
from . import db as projdb

logger = logging.getLogger(__name__)

MIN_BLOCK = 2   # minimum number of consecutive recurrent lines to count as a chorus block


def detect_chorus(lines: list[dict]) -> list[bool]:
    """Return a list[bool] (chorus flag) aligned to `lines` from parse_lrc()."""
    norm = [normalise_line(ln["text"]) for ln in lines]
    counts = Counter(n for n in norm if n)
    recurrent = [bool(n) and counts[n] >= 2 for n in norm]
    flags = [False] * len(lines)
    if not any(recurrent):
        return flags

    # Contiguous runs of recurrent lines.
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(norm):
        if recurrent[i]:
            j = i
            while j < len(norm) and recurrent[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    blocks = [(s, e) for s, e in runs if e - s >= MIN_BLOCK]
    if blocks:
        sigs = Counter(" / ".join(norm[s:e]) for s, e in blocks)
        chorus_sig, freq = sigs.most_common(1)[0]
        if freq >= 2:
            for s, e in blocks:
                if " / ".join(norm[s:e]) == chorus_sig:
                    for k in range(s, e):
                        flags[k] = True
            return flags

    # Fallback: most-repeated single line (the hook).
    hook, freq = counts.most_common(1)[0]
    if freq >= 2:
        for k, n in enumerate(norm):
            if n == hook:
                flags[k] = True
    return flags


def compute_pending(force: bool = False) -> dict:
    """Detect and store chorus flags for every corpus song (lyrics-only stage)."""
    with projdb.connect() as conn:
        done = set() if force else {r["track_id"] for r in conn.execute("SELECT track_id FROM chorus")}
        songs = [dict(r) for r in conn.execute("SELECT track_id, synced_lyrics FROM songs")
                 if force or r["track_id"] not in done]
    if not songs:
        logger.info("Chorus: nothing pending.")
        return {"attempted": 0, "done": 0}

    ok = 0
    for song in songs:
        lines = parse_lrc(song["synced_lyrics"] or "")
        flags = detect_chorus(lines)
        with projdb.connect() as conn:
            projdb.upsert(conn, "chorus", {
                "track_id":   song["track_id"],
                "line_flags": json.dumps(flags),
                "n_chorus":   int(sum(flags)),
                "method":     "lyrics_repetition",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        ok += 1
    logger.info("Chorus: labelled %d songs.", ok)
    return {"attempted": len(songs), "done": ok}


def get_flags(track_id: int) -> list[bool] | None:
    with projdb.connect() as conn:
        row = conn.execute("SELECT line_flags FROM chorus WHERE track_id = ?", (track_id,)).fetchone()
    return json.loads(row["line_flags"]) if row else None
