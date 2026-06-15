"""
05_embed_clap.py — Compute LAION-CLAP audio + text embeddings.

Model: laion/larger_clap_music
  • CLAP = Contrastive Language-Audio Pretraining (Wu et al., 2023)
  • Trained on music + speech + Audioset + LAION-Audio-630K
  • Audio: 48 kHz, processed in 10-second chunks (attentional feature fusion)
  • Text:  RoBERTa → projected into shared 512-D space
  • This variant specifically fine-tuned for music (highest GTZAN accuracy)

Key difference from MuLan
-------------------------
  CLAP was trained on general audio-caption pairs (not music-specific playlist text).
  This makes it a useful comparison — do music-specific training data yield
  better LMC predictive validity for streaming popularity?

Audio handling
--------------
  Full songs are chunked into 10-second overlapping windows (1s overlap).
  Each chunk is embedded independently and the mean is used as the song embedding.
  This is identical to the "feature fusion" approach in the CLAP paper.

MPS notes
---------
  ClapAudioModel uses some windowed-attention ops that occasionally fall back
  to CPU on MPS. PYTORCH_ENABLE_MPS_FALLBACK=1 is set automatically.
  If you see very slow processing, set --device cpu for this model.

Outputs (in results/embeddings/clap/)
-------
  audio_embeddings.npz
  text_embeddings.npz
  similarities.json
  checkpoint.json
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
    chunk_audio, split_text_chunks,
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

MODEL_KEY = "clap"
MODEL_CFG = MODELS[MODEL_KEY]
OUT_DIR   = Path(EMBEDDINGS_DIR) / MODEL_KEY


def load_clap(device: str):
    try:
        import laion_clap
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("Run: pip install laion-clap huggingface_hub")
        sys.exit(1)

    logger.info(f"Loading LAION-CLAP (music model) on {device}...")

    # Download checkpoint to local cache (~600 MB, only happens once)
    ckpt_path = hf_hub_download(
        repo_id  = "lukewys/laion_clap",
        filename = "music_audioset_epoch_15_esc_90.14.pt",
    )
    logger.info(f"Checkpoint at: {ckpt_path}")

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)
    model.load_ckpt(ckpt_path)
    model.eval()
    return model


def get_audio_embedding_clap(model, wav_path: str,
                              device: str, sr: int = 48_000,
                              **kwargs) -> np.ndarray | None:
    """
    Load audio and compute CLAP audio embedding.
    laion_clap handles chunking and fusion internally.
    Audio is quantized to int16 as required by the laion_clap API.
    """
    try:
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)

        # laion_clap expects int16-range float32
        wav = np.clip(wav, -1.0, 1.0)
        wav_int16 = (wav * 32767.0).astype(np.float32)

        with torch.no_grad():
            emb = model.get_audio_embedding_from_data(
                x          = [wav_int16],
                use_tensor = False,
            )
        return emb[0]   # shape [512]

    except Exception as e:
        logger.warning(f"  CLAP audio embedding failed: {e}")
        return None


def get_text_embedding_clap(model, text: str,
                             device: str, max_words: int = 200,
                             **kwargs) -> np.ndarray | None:
    """
    Compute CLAP text embedding with chunking for long lyrics.
    laion_clap accepts a plain list of strings.
    """
    try:
        chunks = split_text_chunks(text, max_words)
        with torch.no_grad():
            embs = model.get_text_embedding(
                x          = chunks,
                use_tensor = False,
            )
        return embs.mean(axis=0)   # shape [512]

    except Exception as e:
        logger.warning(f"  CLAP text embedding failed: {e}")
        return None


def embed_all(device: str | None = None, force: bool = False) -> dict:
    """Compute CLAP embeddings for all available songs."""
    if device is None:
        device = get_device()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = str(OUT_DIR / "checkpoint.json")
    done            = set(load_checkpoint(checkpoint_path).get("done", []))

    all_lyrics = load_all_lyrics()

    audio_embs: dict[str, np.ndarray] = {}
    text_embs : dict[str, np.ndarray] = {}

    if not force:
        audio_npz = str(OUT_DIR / "audio_embeddings.npz")
        text_npz  = str(OUT_DIR / "text_embeddings.npz")
        if Path(audio_npz).exists():
            audio_embs = load_embeddings_npz(audio_npz)
        if Path(text_npz).exists():
            text_embs  = load_embeddings_npz(text_npz)

    to_process = []
    for artist_code, artist_data in CATALOG.items():
        folder = artist_data["folder"]
        for song_id in artist_data["songs"]:
            if not force and song_id in done:
                continue
            audio_path = get_audio_path(song_id, folder)
            if not audio_path:
                continue
            if song_id not in all_lyrics:
                continue
            to_process.append((song_id, audio_path,
                                all_lyrics[song_id]["lyrics_clean"]))

    if not to_process:
        logger.info("Nothing to process.")
        return {}

    logger.info(f"Loading CLAP model for {len(to_process)} songs…")
    model = load_clap(device)

    sr      = MODEL_CFG["audio_sr"]
    chunk_s = MODEL_CFG["chunk_s"]
    n_ok    = 0

    for i, (song_id, audio_path, lyrics) in enumerate(to_process, 1):
        logger.info(f"[{i}/{len(to_process)}] {song_id}")

        a_emb = get_audio_embedding_clap(model, audio_path, device, sr=sr)
        t_emb = get_text_embedding_clap(model, lyrics, device)

        if a_emb is None or t_emb is None:
            logger.warning(f"  [{song_id}] Skipping — embedding failed")
            continue

        audio_embs[song_id] = a_emb
        text_embs[song_id]  = t_emb
        done.add(song_id)
        n_ok += 1

        if n_ok % 5 == 0 or i == len(to_process):
            save_embeddings_npz(audio_embs, str(OUT_DIR / "audio_embeddings.npz"))
            save_embeddings_npz(text_embs,  str(OUT_DIR / "text_embeddings.npz"))
            save_checkpoint({"done": list(done)}, checkpoint_path)
            logger.info(f"  Checkpoint ({n_ok} done)")

    sims = diagonal_similarities(audio_embs, text_embs)
    with open(OUT_DIR / "similarities.json", "w") as f:
        json.dump(sims, f, indent=2)

    vals = list(sims.values())
    logger.info(f"\nCLAP done. {len(sims)} songs. "
                f"Mean LMC={np.mean(vals):.4f}  Std={np.std(vals):.4f}")

    return {"audio": audio_embs, "text": text_embs, "sims": sims}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CLAP embeddings")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    embed_all(device=args.device, force=args.force)
