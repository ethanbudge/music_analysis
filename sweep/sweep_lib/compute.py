"""
compute.py — compute the LMC grid over the existing corpus, resumably.

For every corpus song with audio, and every model, we build the audio units
(whole song, chorus, non-chorus, and each line under ±1/±5/±10 s windows) and the
lyric texts (whole lyrics, chorus, non-chorus, each line), embed them with the
model, format the text three ways (raw / contains / idea), and record

    LMC = cosine(audio_embedding, text_embedding)

as one scalar per (track_id, model, prompt, method) in the `lmc_sweep` table. Line
methods are the MEAN cosine over a song's lines (a single per-song scalar — no
curvature). A song is marked done per model in `sweep_progress`, so runs are fully
resumable and you can process one model at a time.

Reuses: lmc.db (corpus + audio), lmc.chorus (chorus flags), lmc.utils.parse_lrc,
lmc.embeddings._slice (audio slicing), lmcval.models (the 4 model wrappers).
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timezone

import numpy as np


def _now() -> float:
    return time.monotonic()


def _secs(t0: float) -> float:
    return time.monotonic() - t0

from . import config
from lmc import db as projdb
from lmc import chorus as chorus_mod
from lmc.config import EMBEDDINGS_DIR
from lmc.utils import parse_lrc, load_song_embeddings, embedding_path
from lmc.embeddings import _slice
from lmcval.models import load_models

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lmc_sweep (
    track_id INTEGER,
    model    TEXT,      -- mulan | clap | msclap | clamp3
    prompt   TEXT,      -- raw | contains | idea
    method   TEXT,      -- song | seg_chorus | seg_nonchorus | line_buf1 | line_buf5 | line_buf10
    value    REAL,
    PRIMARY KEY (track_id, model, prompt, method)
);
CREATE TABLE IF NOT EXISTS sweep_progress (
    track_id INTEGER,
    model    TEXT,
    done_at  TEXT,
    PRIMARY KEY (track_id, model)
);
"""


def ensure_tables() -> None:
    with projdb.connect() as conn:
        conn.executescript(_SCHEMA)


# ─── helpers ──────────────────────────────────────────────────────────────────────
def _cosine(a, b):
    """Cosine of two vectors, or None if either is missing / degenerate / non-finite."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size != a.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(a @ b / (na * nb))


def _text_level(tkey) -> str:
    """'song' at song level; everything else is a 'song segment' (segment/line)."""
    if tkey == "song":
        return "song"
    return "segment" if tkey in ("chorus", "nonchorus") else "line"


def _flags_for(tid: int, lines: list) -> list:
    """Chorus flags from the DB, falling back to on-the-fly detection (fault-tolerant)."""
    try:
        f = chorus_mod.get_flags(tid)
    except Exception:                                          # noqa: BLE001  (e.g. no chorus table)
        f = None
    if not f or len(f) != len(lines):
        f = chorus_mod.detect_chorus(lines)
    return f


def _build_units(wav: np.ndarray, lines: list, flags: list, dur: float) -> dict:
    """Audio slices + raw texts for one song (present segments only)."""
    sr = config.BASE_SR
    audio, text = {}, {}
    audio["song"] = wav
    text["song"] = "\n".join(ln["text"] for ln in lines)

    chorus_idx    = [i for i, f in enumerate(flags) if f and i < len(lines)]
    nonchorus_idx = [i for i, f in enumerate(flags) if (not f) and i < len(lines)]
    for label, idxs in (("chorus", chorus_idx), ("nonchorus", nonchorus_idx)):
        if not idxs:
            continue
        seg = np.concatenate([_slice(wav, sr, lines[i]["start"], lines[i]["end"], dur) for i in idxs])
        if len(seg) >= int(0.1 * sr):
            audio[label] = seg
            text[label] = " ".join(lines[i]["text"] for i in idxs)

    for i, ln in enumerate(lines):
        text[("line", i)] = ln["text"]
        for method, pad in config.LINE_WINDOWS.items():
            seg = _slice(wav, sr, ln["start"], ln["end"], dur, pad)
            if len(seg) >= int(0.1 * sr):
                audio[("line", method, i)] = seg
    return {"audio": audio, "text": text, "n_lines": len(lines)}


def _load_song_job(song: dict):
    """Load one song's audio + build its units; None if unusable."""
    import librosa
    try:
        wav, _ = librosa.load(song["file_path"], sr=config.BASE_SR, mono=True)
    except Exception as e:                                     # noqa: BLE001
        logger.warning("  [%s] audio load failed: %s", song["track_id"], e)
        return None
    if wav is None or len(wav) < int(0.2 * config.BASE_SR):
        return None
    lines = parse_lrc(song["synced_lyrics"] or "")
    if not lines:
        return None
    flags = _flags_for(song["track_id"], lines)
    units = _build_units(wav, lines, flags, len(wav) / config.BASE_SR)
    units["tid"] = song["track_id"]
    return units


# ─── pending / progress ───────────────────────────────────────────────────────────
def _pending(model_key: str, limit: int | None = None) -> list[dict]:
    ensure_tables()
    with projdb.connect() as conn:
        songs = [dict(r) for r in projdb.songs_with_audio(conn)]
        done = {r["track_id"] for r in conn.execute(
            "SELECT track_id FROM sweep_progress WHERE model = ?", (model_key,))}
    todo = [s for s in songs if s["track_id"] not in done]
    return todo[:limit] if limit else todo


def coverage() -> dict:
    """How many songs are done per model (for the notebook's progress view)."""
    ensure_tables()
    with projdb.connect() as conn:
        total = projdb.count(conn, "audio", "status = 'done'")
        done = {m: 0 for m in config.MODELS}
        for r in conn.execute("SELECT model, COUNT(*) AS n FROM sweep_progress GROUP BY model"):
            done[r["model"]] = r["n"]
    return {"songs_with_audio": total, "done": done}


# ─── the driver ───────────────────────────────────────────────────────────────────
def _score_chunk(model, model_key: str, jobs: list) -> tuple[list, list]:
    """Embed a chunk of songs in batched calls and return (lmc rows, progress rows)."""
    # flatten audio units across the chunk → one embed call
    akeys, awavs = [], []
    for ji, job in enumerate(jobs):
        for akey, wav in job["audio"].items():
            akeys.append((ji, akey)); awavs.append(wav)
    A = model.embed_audio_batch(awavs, config.BASE_SR) if awavs else np.zeros((0, 1))
    avec = {k: A[i] for i, k in enumerate(akeys)}

    # flatten text per prompt across the chunk → one embed call per prompt
    tvec = {}
    for prompt in config.PROMPTS:
        tkeys, ttexts = [], []
        for ji, job in enumerate(jobs):
            for tkey, raw in job["text"].items():
                ttexts.append(config.format_prompt(prompt, raw, _text_level(tkey)))
                tkeys.append((ji, tkey))
        T = model.embed_text_batch(ttexts) if ttexts else np.zeros((0, 1))
        tvec[prompt] = {k: T[i] for i, k in enumerate(tkeys)}

    now = datetime.now(timezone.utc).isoformat()
    rows, prog = [], []
    for ji, job in enumerate(jobs):
        tid = job["tid"]
        for prompt in config.PROMPTS:
            tv = tvec[prompt]
            # song + segments (single-cosine methods)
            for akey, method in (("song", "song"), ("chorus", "seg_chorus"),
                                 ("nonchorus", "seg_nonchorus")):
                v = _cosine(avec.get((ji, akey)), tv.get((ji, akey)))
                if v is not None:
                    rows.append((tid, model_key, prompt, method, round(v, 6)))
            # line windows: mean cosine over the song's lines
            for method in config.LINE_METHODS:
                vals = []
                for i in range(job["n_lines"]):
                    c = _cosine(avec.get((ji, ("line", method, i))), tv.get((ji, ("line", i))))
                    if c is not None:
                        vals.append(c)
                if vals:
                    rows.append((tid, model_key, prompt, method, round(float(np.mean(vals)), 6)))
        prog.append((tid, model_key, now))
    return rows, prog


# ─── reuse of the existing MuLan/CLAP embedding cache ─────────────────────────────
# The observational pipeline already cached, per song, a bundle with all the AUDIO
# embeddings (prompt-independent) + the RAW-text embeddings. We reuse those instead of
# recomputing: raw-prompt LMC needs zero model calls, and contains/idea only re-embed
# the (short) wrapped text — the expensive audio embedding is never redone.
_CACHE_MODELS = ("mulan", "clap")
_BUNDLE_KEYS = ("audio_full", "chorus_audio", "nonchorus_audio",
                "audio_buf1", "audio_buf5", "audio_buf10",
                "text_full", "chorus_text", "nonchorus_text", "line_text")


def _load_bundle(model_key: str, tid: int):
    """Return a usable cached bundle for (model, song), or None to compute fresh."""
    if model_key not in _CACHE_MODELS:
        return None
    b = load_song_embeddings(embedding_path(EMBEDDINGS_DIR, model_key, tid))
    if b is None or not all(k in b for k in _BUNDLE_KEYS):
        return None
    return b


def _score_song_from_cache(model_key: str, tid: int, bundle: dict, song: dict, model):
    """Score one song from the cached bundle (audio + raw text); re-embed only the
    wrapped contains/idea text. Returns (rows, progress) or None to fall back."""
    lines = parse_lrc(song["synced_lyrics"] or "")
    L = len(lines)
    line_text_raw = bundle["line_text"]
    if L == 0 or getattr(line_text_raw, "shape", (0,))[0] != L:
        return None                                    # bundle/lyrics out of sync → recompute

    flags = _flags_for(tid, lines)

    A = {"song": bundle["audio_full"], "seg_chorus": bundle["chorus_audio"],
         "seg_nonchorus": bundle["nonchorus_audio"]}
    A_line = {m: bundle["audio_" + m[len("line_"):]] for m in config.LINE_METHODS}  # line_buf1→audio_buf1
    T_raw = {"song": bundle["text_full"], "seg_chorus": bundle["chorus_text"],
             "seg_nonchorus": bundle["nonchorus_text"]}

    # raw text strings needed to build the contains/idea prompts
    song_text = "\n".join(ln["text"] for ln in lines)
    seg_text = {}
    for sk, want in (("seg_chorus", True), ("seg_nonchorus", False)):
        idxs = [i for i, f in enumerate(flags) if bool(f) == want]
        seg_text[sk] = " ".join(lines[i]["text"] for i in idxs) if idxs else None
    line_texts = [ln["text"] for ln in lines]

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for prompt in config.PROMPTS:
        if prompt == "raw":
            T_song, T_seg, T_line = T_raw["song"], T_raw, line_text_raw
        else:
            keys, texts = ["song"], [config.format_prompt(prompt, song_text, "song")]
            for sk in ("seg_chorus", "seg_nonchorus"):
                if seg_text[sk]:
                    keys.append(sk); texts.append(config.format_prompt(prompt, seg_text[sk], "segment"))
            n_head = len(texts)
            texts += [config.format_prompt(prompt, t, "line") for t in line_texts]
            E = model.embed_text_batch(texts)
            emap = {k: E[j] for j, k in enumerate(keys)}
            T_song = emap.get("song")
            T_seg = {"seg_chorus": emap.get("seg_chorus"), "seg_nonchorus": emap.get("seg_nonchorus")}
            T_line = E[n_head:]

        v = _cosine(A["song"], T_song)
        if v is not None:
            rows.append((tid, model_key, prompt, "song", round(v, 6)))
        for sk in ("seg_chorus", "seg_nonchorus"):
            v = _cosine(A[sk], T_seg[sk])
            if v is not None:
                rows.append((tid, model_key, prompt, sk, round(v, 6)))
        for method in config.LINE_METHODS:
            Aw = A_line[method]
            vals = [c for i in range(min(L, len(Aw), len(T_line)))
                    if (c := _cosine(Aw[i], T_line[i])) is not None]
            if vals:
                rows.append((tid, model_key, prompt, method, round(float(np.mean(vals)), 6)))
    return rows, [(tid, model_key, now)]


def _write(rows: list, prog: list) -> None:
    with projdb.connect() as conn:
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO lmc_sweep (track_id, model, prompt, method, value) "
                "VALUES (?,?,?,?,?)", rows)
        if prog:
            conn.executemany(
                "INSERT OR REPLACE INTO sweep_progress (track_id, model, done_at) "
                "VALUES (?,?,?)", prog)


def compute_model(model_key: str, chunk_size: int = 16, limit: int | None = None,
                  device: str | None = None) -> dict:
    """Compute the whole grid for ONE model over all pending songs (resumable).

    Progress is logged after EVERY song (fast models) or every chunk (CLaMP 3). Make
    sure logging is on — `from lmc.utils import setup_logging; setup_logging()` — or
    you'll see no output while it works. The fast models are processed one song at a
    time (bounded memory + a per-song heartbeat); CLaMP 3 is batched `chunk_size`
    songs per subprocess call so it doesn't reload its model per song.
    """
    assert model_key in config.MODELS, f"unknown model {model_key}"
    ensure_tables()
    pending = _pending(model_key, limit)
    n = len(pending)
    if not pending:
        logger.info("[%s] nothing pending.", model_key)
        return {"model": model_key, "processed": 0}

    logger.info("[%s] %d songs pending — loading model…", model_key, n)
    models = load_models([model_key], device=device)
    if model_key not in models:
        logger.error("[%s] failed to load — see warning above. Skipping.", model_key)
        return {"model": model_key, "processed": 0, "error": "load_failed"}
    model = models[model_key]
    logger.info("[%s] model loaded; starting.", model_key)

    processed = 0
    t0 = _now()
    if model_key == "clamp3":
        # Batch a chunk of songs per subprocess call (amortises CLaMP 3's model reload).
        for start in range(0, n, chunk_size):
            chunk = pending[start:start + chunk_size]
            jobs = [j for j in (_load_song_job(s) for s in chunk) if j is not None]
            if not jobs:
                continue
            hi = min(start + chunk_size, n)
            logger.info("  [clamp3] songs %d–%d/%d: embedding %d songs (slow: subprocess + MERT-95M)…",
                        start + 1, hi, n, len(jobs))
            rows, prog = _score_chunk(model, model_key, jobs)
            _write(rows, prog)
            processed += len(jobs)
            logger.info("  [clamp3] %d/%d done (%d rows, %.0fs elapsed).", hi, n, len(rows),
                        _secs(t0))
    else:
        # One song at a time: REUSE the cached MuLan/CLAP bundle where possible
        # (audio never re-embedded; only contains/idea wrapped text is computed).
        n_cached = 0
        for i, song in enumerate(pending, 1):
            tid = song["track_id"]
            src, result = "cache", None
            bundle = _load_bundle(model_key, tid)
            if bundle is not None:
                result = _score_song_from_cache(model_key, tid, bundle, song, model)
            if result is None:                          # no / stale cache → compute fresh
                src = "fresh"
                job = _load_song_job(song)
                if job is None:
                    logger.info("  [%s] %d/%d  track %s — skipped (no usable audio/lyrics).",
                                model_key, i, n, tid)
                    continue
                result = _score_chunk(model, model_key, [job])
            rows, prog = result
            _write(rows, prog)
            processed += 1
            n_cached += (src == "cache")
            logger.info("  [%s] %d/%d  track %-9s  %-5s → %2d rows  (%.1fs/song avg)",
                        model_key, i, n, tid, src, len(rows), _secs(t0) / max(processed, 1))
        if model_key in _CACHE_MODELS:
            logger.info("  [%s] reused cached audio for %d/%d songs.", model_key, n_cached, processed)

    logger.info("[%s] done: %d songs processed in %.0fs.", model_key, processed, _secs(t0))
    return {"model": model_key, "processed": processed}


def compute_all(models: list[str] | None = None, chunk_size: int = 16,
                limit: int | None = None) -> dict:
    """Compute every model in turn (resumable). Run one at a time for CLaMP 3."""
    out = {}
    for mk in (models or config.MODELS):
        out[mk] = compute_model(mk, chunk_size=chunk_size, limit=limit)
    return out
