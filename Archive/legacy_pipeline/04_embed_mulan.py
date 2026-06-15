"""
04_embed_mulan.py — Compute MuQ-MuLan audio + text embeddings for all songs.

Model: OpenMuQ/MuQ-MuLan-large
  • Jointly trained audio-text embedding (shared latent space)
  • Audio: 24 kHz waveform → 512-D embedding
  • Text:  XLM-RoBERTa → 512-D embedding
  • Similarity: cosine in shared space (equivalent to dot product after L2-norm)

Outputs (in results/embeddings/mulan/)
-------
  audio_embeddings.npz   — {song_id: np.ndarray [512]}
  text_embeddings.npz    — {song_id: np.ndarray [512]}
  similarities.json      — {song_id: float}  (diagonal cosine similarities)
  checkpoint.json        — resume state (which songs already processed)

MPS notes
---------
  MuQ-MuLan runs on MPS. Set PYTORCH_ENABLE_MPS_FALLBACK=1 (done in utils)
  so any unsupported ops silently fall back to CPU without crashing.
"""

from __future__ import annotations
import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import librosa
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CATALOG, AUDIO_BASE_DIR, EMBEDDINGS_DIR, MODELS
from utils import (
    setup_logging, get_device, model_to_device,
    save_embeddings_npz, load_embeddings_npz,
    load_checkpoint, save_checkpoint, diagonal_similarities,
    split_text_chunks,
)

import importlib.util, sys, os

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

MODEL_KEY = "mulan"
MODEL_CFG = MODELS[MODEL_KEY]
OUT_DIR   = Path(EMBEDDINGS_DIR) / MODEL_KEY


def load_mulan(device: str):
    """Load MuQ-MuLan model."""
    from muq import MuQMuLan
    logger.info(f"Loading {MODEL_CFG['name']} on {device}…")
    model = MuQMuLan.from_pretrained(MODEL_CFG["hf_id"])
    model = model_to_device(model, device)
    return model


def get_audio_embedding(model, wav_path: str, device: str,
                         sr: int = 24_000) -> np.ndarray | None:
    """Load audio → MuLan audio embedding [512]."""
    try:
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        wav_t  = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(wavs=wav_t)
        return emb.squeeze(0).cpu().numpy()
    except Exception as e:
        logger.warning(f"  Audio embedding failed: {e}")
        return None


def get_text_embedding(model, text: str, device: str,
                        max_words: int = 200) -> np.ndarray | None:
    """
    Compute MuLan text embedding.  If text exceeds token limit, chunks and averages.
    Returns [512] numpy array.
    """
    def _embed(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            emb = model(texts=texts)
        return emb.cpu().numpy()

    try:
        chunks = split_text_chunks(text, max_words)
        if len(chunks) == 1:
            return _embed([text])[0]
        # Average over chunks
        vecs = _embed(chunks)   # [C, 512]
        return vecs.mean(axis=0)
    except Exception as e:
        logger.warning(f"  Text embedding failed ({e}) — trying with halved chunk size")
        try:
            chunks = split_text_chunks(text, max_words // 2)
            vecs   = []
            for chunk in chunks:
                try:
                    vecs.append(_embed([chunk])[0])
                except Exception:
                    pass
            if vecs:
                return np.stack(vecs).mean(axis=0)
        except Exception as e2:
            logger.error(f"  Text embedding failed entirely: {e2}")
        return None


def embed_all(device: str | None = None, force: bool = False) -> dict:
    """
    Compute and save embeddings for all songs with available audio + lyrics.

    Returns {song_id: {"audio": np.ndarray, "text": np.ndarray, "sim": float}}
    """
    if device is None:
        device = get_device()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = str(OUT_DIR / "checkpoint.json")
    done            = set(load_checkpoint(checkpoint_path).get("done", []))

    all_lyrics = load_all_lyrics()

    audio_embs : dict[str, np.ndarray] = {}
    text_embs  : dict[str, np.ndarray] = {}

    # Pre-load any already-computed embeddings
    audio_npz_path = str(OUT_DIR / "audio_embeddings.npz")
    text_npz_path  = str(OUT_DIR / "text_embeddings.npz")
    if not force:
        if Path(audio_npz_path).exists():
            audio_embs = load_embeddings_npz(audio_npz_path)
        if Path(text_npz_path).exists():
            text_embs  = load_embeddings_npz(text_npz_path)

    # Identify songs to process
    to_process = []
    for artist_code, artist_data in CATALOG.items():
        folder = artist_data["folder"]
        for song_id in artist_data["songs"]:
            if not force and song_id in done:
                continue
            audio_path = get_audio_path(song_id, folder)
            if not audio_path:
                logger.debug(f"[{song_id}] No audio file — skipping")
                continue
            if song_id not in all_lyrics:
                logger.debug(f"[{song_id}] No lyrics — skipping")
                continue
            to_process.append((song_id, audio_path, all_lyrics[song_id]["lyrics_clean"]))

    if not to_process:
        logger.info("Nothing to process — all songs already embedded or missing data.")
        return {}

    logger.info(f"Loading model for {len(to_process)} songs…")
    model = load_mulan(device)

    n_ok = 0
    for i, (song_id, audio_path, lyrics) in enumerate(to_process, 1):
        logger.info(f"[{i}/{len(to_process)}] {song_id}")

        a_emb = get_audio_embedding(model, audio_path, device, sr=MODEL_CFG["audio_sr"])
        t_emb = get_text_embedding(model, lyrics, device)

        if a_emb is None or t_emb is None:
            logger.warning(f"  [{song_id}] Skipping — embedding failed")
            continue

        audio_embs[song_id] = a_emb
        text_embs[song_id]  = t_emb
        done.add(song_id)
        n_ok += 1

        # Save incrementally every 5 songs
        if n_ok % 5 == 0 or i == len(to_process):
            save_embeddings_npz(audio_embs, audio_npz_path)
            save_embeddings_npz(text_embs,  text_npz_path)
            save_checkpoint({"done": list(done)}, checkpoint_path)
            logger.info(f"  Checkpoint saved ({n_ok} songs processed)")

    # ── Compute diagonal similarities ─────────────────────────────────────
    sims = diagonal_similarities(audio_embs, text_embs)
    with open(OUT_DIR / "similarities.json", "w") as f:
        json.dump(sims, f, indent=2)

    logger.info(f"\n{'═'*60}")
    logger.info(f"MuLan embedding done. {len(sims)} similarities saved → {OUT_DIR}")

    # Summary stats
    vals = list(sims.values())
    logger.info(f"  Mean LMC: {np.mean(vals):.4f}  |  Std: {np.std(vals):.4f}  |  "
                f"Min: {np.min(vals):.4f}  |  Max: {np.max(vals):.4f}")

    return {"audio": audio_embs, "text": text_embs, "sims": sims}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MuQ-MuLan embeddings")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu")
    parser.add_argument("--force",  action="store_true", help="Recompute all (ignore checkpoint)")
    args = parser.parse_args()

    embed_all(device=args.device, force=args.force)
