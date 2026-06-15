"""
embeddings.py — Compute and cache joint audio-text embeddings (MuQ-MuLan and
LAION-CLAP) for every level of analysis, once per song.

For each song + model we cache a single compressed .npz "bundle" containing
everything the alignment stage needs, so embeddings are never recomputed:

  audio_full              [D]      whole-song audio
  text_full               [D]      whole-lyrics text
  line_text               [L, D]   per synced line, text
  audio_exact/buf1/5/10   [L, D]   per line, audio under each context window
  chorus_audio/text       [D]      concatenated chorus audio / joined chorus text
  nonchorus_audio/text    [D]      everything not flagged chorus

Resumable + iterative: a song is skipped if its bundle file already exists
(unless force=True), and bundles are written atomically per song.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone

import numpy as np

from .config import MODELS, EMBEDDINGS_DIR, CONTEXT_WINDOWS
from .utils import (get_device, model_to_device, split_text_chunks,
                    parse_lrc, lrc_to_plaintext, embedding_path,
                    save_song_embeddings)
from . import db as projdb
from . import chorus as chorus_mod

logger = logging.getLogger(__name__)


# ─── Model wrappers: each exposes embed_audio(wav) and embed_text(text) → [D] ────
class _MuLan:
    sr = MODELS["mulan"]["audio_sr"]
    dim = MODELS["mulan"]["dim"]

    def __init__(self, device):
        import torch
        from muq import MuQMuLan
        self.torch = torch
        self.device = device
        logger.info("Loading MuQ-MuLan on %s…", device)
        self.model = model_to_device(MuQMuLan.from_pretrained(MODELS["mulan"]["hf_id"]), device)

    def embed_audio(self, wav: np.ndarray) -> np.ndarray | None:
        if wav is None or len(wav) < int(0.1 * self.sr):
            return None
        try:
            t = self.torch.tensor(wav, dtype=self.torch.float32).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                return self.model(wavs=t).squeeze(0).cpu().numpy()
        except Exception as e:                                 # noqa: BLE001
            logger.debug("  mulan audio embed failed: %s", e)
            return None

    def embed_text(self, text: str) -> np.ndarray | None:
        chunks = split_text_chunks(text, 200)
        if not chunks:
            return None
        try:
            with self.torch.no_grad():
                vecs = self.model(texts=chunks).cpu().numpy()
            return vecs.mean(axis=0)
        except Exception as e:                                 # noqa: BLE001
            logger.debug("  mulan text embed failed: %s", e)
            return None


class _CLAP:
    sr = MODELS["clap"]["audio_sr"]
    dim = MODELS["clap"]["dim"]
    chunk_s = MODELS["clap"]["chunk_s"]

    def __init__(self, device):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device
        logger.info("Loading LAION-CLAP on %s…", device)
        self.processor = ClapProcessor.from_pretrained(MODELS["clap"]["hf_id"])
        self.model = model_to_device(ClapModel.from_pretrained(MODELS["clap"]["hf_id"]), device)

    def _chunks(self, wav):
        n = int(self.chunk_s * self.sr)
        if len(wav) <= n:
            return [wav]
        return [wav[i:i + n] for i in range(0, len(wav), n) if len(wav[i:i + n]) >= int(0.2 * self.sr)]

    def embed_audio(self, wav: np.ndarray) -> np.ndarray | None:
        if wav is None or len(wav) < int(0.1 * self.sr):
            return None
        try:
            vecs = []
            for ch in self._chunks(wav):
                inp = self.processor(audios=ch, sampling_rate=self.sr, return_tensors="pt").to(self.device)
                with self.torch.no_grad():
                    vecs.append(self.model.get_audio_features(**inp).squeeze(0).cpu().numpy())
            return np.stack(vecs).mean(axis=0) if vecs else None
        except Exception as e:                                 # noqa: BLE001
            logger.debug("  clap audio embed failed: %s", e)
            return None

    def embed_text(self, text: str) -> np.ndarray | None:
        chunks = split_text_chunks(text, 60)
        if not chunks:
            return None
        try:
            inp = self.processor(text=chunks, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                vecs = self.model.get_text_features(**inp).cpu().numpy()
            return vecs.mean(axis=0)
        except Exception as e:                                 # noqa: BLE001
            logger.debug("  clap text embed failed: %s", e)
            return None


def _load_embedder(model_key: str, device: str):
    return {"mulan": _MuLan, "clap": _CLAP}[model_key](device)


# ─── Bundle construction ─────────────────────────────────────────────────────────
def _slice(wav, sr, start, end, total_dur, pad=0.0):
    a = max(0.0, start - pad)
    b = min(total_dur, (end if end is not None else total_dur) + pad)
    b = max(b, a + 0.1)
    return wav[int(a * sr):int(b * sr)]


def _build_bundle(emb, wav, sr, total_dur, lines, flags) -> dict:
    D = emb.dim
    zero = np.zeros(D, dtype=np.float32)

    bundle = {
        "audio_full": (emb.embed_audio(wav) if len(wav) else zero),
        "text_full":  emb.embed_text(lrc_to_plaintext_from_lines(lines)),
    }
    bundle["audio_full"] = _safe(bundle["audio_full"], D)
    bundle["text_full"]  = _safe(bundle["text_full"], D)

    L = len(lines)
    line_text = np.zeros((L, D), dtype=np.float32)
    win_audio = {w: np.zeros((L, D), dtype=np.float32) for w in CONTEXT_WINDOWS}
    for i, ln in enumerate(lines):
        line_text[i] = _safe(emb.embed_text(ln["text"]), D)
        for w, pad in CONTEXT_WINDOWS.items():
            seg = _slice(wav, sr, ln["start"], ln["end"], total_dur, pad)
            win_audio[w][i] = _safe(emb.embed_audio(seg), D)
    bundle["line_text"] = line_text
    for w in CONTEXT_WINDOWS:
        bundle[f"audio_{w}"] = win_audio[w]

    # Segment: chorus vs non-chorus.
    for label, want in (("chorus", True), ("nonchorus", False)):
        idxs = [i for i, f in enumerate(flags) if bool(f) == want]
        if idxs:
            seg_wav = np.concatenate([_slice(wav, sr, lines[i]["start"], lines[i]["end"], total_dur)
                                      for i in idxs]) if len(wav) else np.array([])
            seg_text = " ".join(lines[i]["text"] for i in idxs)
            bundle[f"{label}_audio"] = _safe(emb.embed_audio(seg_wav), D)
            bundle[f"{label}_text"]  = _safe(emb.embed_text(seg_text), D)
        else:
            bundle[f"{label}_audio"] = zero
            bundle[f"{label}_text"]  = zero
    return bundle


def lrc_to_plaintext_from_lines(lines) -> str:
    return "\n".join(ln["text"] for ln in lines)


def _safe(v, D):
    return v.astype(np.float32) if v is not None else np.zeros(D, dtype=np.float32)


# ─── Driver ──────────────────────────────────────────────────────────────────────
def embed_pending(model_key: str, limit: int | None = None,
                  force: bool = False, device: str | None = None) -> dict:
    """
    Compute embedding bundles for songs that have audio but no cached bundle for
    `model_key` yet. Loads the model once and processes songs incrementally.
    """
    assert model_key in MODELS, f"unknown model {model_key}"
    import librosa
    sr = MODELS[model_key]["audio_sr"]
    device = device or get_device()

    with projdb.connect() as conn:
        have_audio = [dict(r) for r in projdb.songs_with_audio(conn)]
        cached = set() if force else {
            r["track_id"] for r in conn.execute(
                "SELECT track_id FROM embeddings WHERE model = ?", (model_key,))}

    todo = [s for s in have_audio if force or s["track_id"] not in cached]
    # Belt-and-braces: also honour an existing bundle file on disk.
    todo = [s for s in todo if force or not embedding_path(EMBEDDINGS_DIR, model_key, s["track_id"]).exists()]
    if limit:
        todo = todo[:limit]
    if not todo:
        logger.info("Embeddings[%s]: nothing pending.", model_key)
        return {"attempted": 0, "done": 0}

    logger.info("Embeddings[%s]: %d songs (device=%s).", model_key, len(todo), device)
    emb = _load_embedder(model_key, device)

    ok = 0
    for i, song in enumerate(todo, 1):
        tid = song["track_id"]
        logger.info("[%d/%d] %s", i, len(todo), tid)
        try:
            wav, _ = librosa.load(song["file_path"], sr=sr, mono=True)
        except Exception as e:                                 # noqa: BLE001
            logger.warning("  audio load failed: %s", e)
            continue
        total_dur = len(wav) / sr
        lines = parse_lrc(song["synced_lyrics"] or "")
        flags = chorus_mod.get_flags(tid) or [False] * len(lines)
        if len(flags) != len(lines):                           # stale chorus flags
            flags = chorus_mod.detect_chorus(lines)

        bundle = _build_bundle(emb, wav, sr, total_dur, lines, flags)
        path = embedding_path(EMBEDDINGS_DIR, model_key, tid)
        save_song_embeddings(path, bundle)
        with projdb.connect() as conn:
            projdb.upsert(conn, "embeddings", {
                "track_id": tid, "model": model_key, "path": str(path),
                "n_lines": len(lines),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        ok += 1

    logger.info("Embeddings[%s] done: %d songs.", model_key, ok)
    return {"attempted": len(todo), "done": ok}
