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

from .config import MODELS, MERT, EMBEDDINGS_DIR, CONTEXT_WINDOWS
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
    """LAION-CLAP via the official `laion_clap` package.

    We use laion_clap rather than transformers' ClapModel because that path
    silently failed to load this checkpoint's projection / logit-scale weights
    (matched audio-text cosine ≈ 0). laion_clap loads the original checkpoints
    directly. Point config MODELS['clap']['ckpt'] (env LMC_CLAP_CKPT) at the
    music checkpoint; otherwise the package's default checkpoint is downloaded.
    """
    sr = MODELS["clap"]["audio_sr"]
    dim = MODELS["clap"]["dim"]
    chunk_s = MODELS["clap"]["chunk_s"]

    def __init__(self, device):
        import torch
        import laion_clap
        self.torch = torch
        self.device = device
        cfg = MODELS["clap"]
        # The audio backbone MUST match the checkpoint: the music checkpoint is
        # HTSAT-base (1024-d), but laion_clap's default download is HTSAT-tiny
        # (768-d). Using the wrong one => a size-mismatch crash on load. So only
        # request HTSAT-base when a (music) checkpoint is actually provided.
        amodel = cfg["amodel"] if cfg["ckpt"] else "HTSAT-tiny"
        logger.info("Loading LAION-CLAP (%s) on %s…", amodel, device)
        self.model = laion_clap.CLAP_Module(
            enable_fusion=cfg["enable_fusion"], amodel=amodel, device=device)
        if cfg["ckpt"]:
            self.model.load_ckpt(ckpt=cfg["ckpt"])           # music checkpoint
        else:
            logger.warning("LMC_CLAP_CKPT unset — loading laion_clap's default GENERAL "
                           "(non-music) HTSAT-tiny checkpoint. For the music model, download "
                           "music_audioset_epoch_15_esc_90.14.pt and set LMC_CLAP_CKPT.")
            self.model.load_ckpt()
        self.model.eval()

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
                x = np.ascontiguousarray(ch, dtype=np.float32)[None, :]   # [1, samples]
                with self.torch.no_grad():
                    v = self.model.get_audio_embedding_from_data(x=x, use_tensor=False)
                vecs.append(np.asarray(v).squeeze(0))
            return np.stack(vecs).mean(axis=0) if vecs else None
        except Exception as e:                                 # noqa: BLE001
            logger.warning("  clap audio embed failed: %s", e)
            return None

    def embed_text(self, text: str) -> np.ndarray | None:
        chunks = split_text_chunks(text, 60)
        if not chunks:
            return None
        try:
            with self.torch.no_grad():
                # laion_clap expects >= 2 texts; pad with a copy if needed.
                q = chunks if len(chunks) >= 2 else chunks * 2
                v = self.model.get_text_embedding(q, use_tensor=False)
            return np.asarray(v)[:len(chunks)].mean(axis=0)
        except Exception as e:                                 # noqa: BLE001
            logger.warning("  clap text embed failed: %s", e)
            return None


class _MERT:
    """MERT-v1-330M audio-only encoder → one 1024-d vector per song.

    Mean-pools the hidden states over time and over layers. Used only for control
    features (no text tower), so it exposes embed_audio() and is driven by mert.py.
    """
    sr = MERT["audio_sr"]
    dim = MERT["dim"]
    chunk_s = MERT["chunk_s"]

    def __init__(self, device):
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.torch = torch
        self.device = device
        logger.info("Loading %s on %s…", MERT["name"], device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT["hf_id"], trust_remote_code=True)
        self.model = model_to_device(
            AutoModel.from_pretrained(MERT["hf_id"], trust_remote_code=True), device).eval()

    def _chunks(self, wav):
        # MERT operates on short clips; a whole song in one pass overflows the MPS
        # buffer cap. Split into ~chunk_s windows, drop sub-0.5 s tails.
        n = int(self.chunk_s * self.sr)
        if len(wav) <= n:
            return [wav]
        return [wav[i:i + n] for i in range(0, len(wav), n) if len(wav[i:i + n]) >= int(0.5 * self.sr)]

    def _embed_chunk(self, ch):
        inp = self.processor(ch, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp, output_hidden_states=True)
        # hidden_states: tuple(L+1) of [1, T, 1024] → mean over time then layers.
        hs = self.torch.stack(out.hidden_states, dim=0).squeeze(1)       # [L+1, T, 1024]
        return hs.mean(dim=1).mean(dim=0).cpu().numpy()                  # [1024]

    def embed_audio(self, wav: np.ndarray) -> np.ndarray | None:
        if wav is None or len(wav) < int(0.1 * self.sr):
            return None
        try:
            vecs = [self._embed_chunk(ch) for ch in self._chunks(wav)]   # average over chunks
            return np.stack(vecs).mean(axis=0) if vecs else None
        except Exception as e:                                 # noqa: BLE001
            logger.warning("  mert embed failed: %s", e)
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
        # Guard against silently caching a dead bundle (e.g. an embedder API break):
        # if the song-level audio AND text vectors are both all-zero, the embedder
        # failed for this song — skip rather than poison the cache.
        if not (np.any(bundle.get("audio_full")) and np.any(bundle.get("text_full"))):
            logger.warning("  [%s] all-zero embedding for %s — embedder failed, skipping.",
                           model_key, tid)
            continue
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
