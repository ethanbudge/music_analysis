"""
06_embed_mert_sbert.py — Late-fusion baseline: MERT (audio) + Sentence-BERT (text).

Rationale
---------
  MuLan and CLAP are jointly-trained (the audio and text towers share a
  contrastive learning objective, so their embedding spaces are explicitly
  aligned).  This script provides a critical BASELINE using:

    Audio:  MERT-v1-95M  (Music undERstanding Transformer, Ma et al. 2023)
            Trained via masked audio modelling on music; outputs rich
            acoustic representations but NOT aligned to text.

    Text:   all-mpnet-base-v2  (Sentence-BERT)
            Strong general-purpose sentence encoder.

  If the late-fusion baseline yields similar LMC predictive validity as
  MuLan/CLAP, it suggests the joint training is not adding critical
  information — and vice versa.  This comparison is the key methodological
  test in Section 4 of the paper.

Audio embedding
---------------
  MERT expects 24 kHz mono audio.  We extract the final hidden state
  (layer 12 of 12) and apply mean pooling over time → [768]-D vector.

Text embedding
--------------
  SentenceTransformer('all-mpnet-base-v2') → mean-pooled [768]-D vector.
  Long lyrics are chunked at 200 words and averaged.

Similarity
----------
  Cosine similarity between L2-normalised audio and text vectors.
  NOTE: these are in DIFFERENT embedding spaces — so the absolute value
  is not directly interpretable as "semantic similarity" the way it is for
  jointly trained models.  It is nonetheless a valid cross-modal distance
  measure for ranking purposes.

Outputs (results/embeddings/mert_sbert/)
-------
  audio_embeddings.npz    — {song_id: [768]}
  text_embeddings.npz     — {song_id: [768]}
  similarities.json       — {song_id: float}
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
    split_text_chunks, chunk_audio,
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

MODEL_KEY = "mert_sbert"
MODEL_CFG = MODELS[MODEL_KEY]
OUT_DIR   = Path(EMBEDDINGS_DIR) / MODEL_KEY


# ── MERT audio encoder ────────────────────────────────────────────────────────

def load_mert(device: str):
    """Load MERT-v1-95M model + processor."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    hf_id = MODEL_CFG["audio_hf_id"]
    logger.info(f"Loading MERT ({hf_id}) on {device}…")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        hf_id, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
    model = model_to_device(model, device)
    return model, processor


def get_mert_embedding(model, processor, wav_path: str, device: str,
                        sr: int = 24_000,
                        max_duration_s: float = 120.0) -> np.ndarray | None:
    """
    Compute MERT audio embedding.
    Truncates audio to max_duration_s to avoid OOM on long tracks.
    Returns mean-pooled last hidden state [768].
    """
    try:
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)

        # Truncate to avoid OOM
        max_samples = int(max_duration_s * sr)
        if len(wav) > max_samples:
            # Take the first 60s + the middle 60s as representative
            mid_start = max(0, (len(wav) // 2) - (max_samples // 4))
            wav = np.concatenate([
                wav[:max_samples // 2],
                wav[mid_start : mid_start + max_samples // 2],
            ])

        inputs = processor(
            wav,
            sampling_rate  = sr,
            return_tensors = "pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Use the last hidden state, mean-pool over time
        last_hidden = outputs.last_hidden_state          # [1, T, 768]
        emb = last_hidden.squeeze(0).mean(dim=0)         # [768]
        return emb.cpu().float().numpy()

    except Exception as e:
        logger.warning(f"  MERT embedding failed: {e}")
        return None


# ── Sentence-BERT text encoder ────────────────────────────────────────────────

def load_sbert(device: str):
    """Load Sentence-BERT all-mpnet-base-v2."""
    from sentence_transformers import SentenceTransformer
    hf_id = MODEL_CFG["text_hf_id"]
    logger.info(f"Loading SBERT ({hf_id}) on {device}…")
    model = SentenceTransformer(hf_id, device=device)
    return model


def get_sbert_embedding(sbert_model, text: str,
                         max_words: int = 200) -> np.ndarray | None:
    """
    Compute Sentence-BERT embedding for potentially long text.
    Chunks if needed; returns mean [768].
    """
    try:
        chunks = split_text_chunks(text, max_words)
        if len(chunks) == 1:
            return sbert_model.encode(text, convert_to_numpy=True,
                                       show_progress_bar=False)
        vecs = sbert_model.encode(chunks, convert_to_numpy=True,
                                   show_progress_bar=False)
        return vecs.mean(axis=0)
    except Exception as e:
        logger.warning(f"  SBERT embedding failed: {e}")
        return None


# ── Main embedding loop ───────────────────────────────────────────────────────

def embed_all(device: str | None = None, force: bool = False) -> dict:
    """Compute MERT+SBERT embeddings for all available songs."""
    if device is None:
        device = get_device()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = str(OUT_DIR / "checkpoint.json")
    done            = set(load_checkpoint(checkpoint_path).get("done", []))

    all_lyrics = load_all_lyrics()

    audio_embs: dict[str, np.ndarray] = {}
    text_embs : dict[str, np.ndarray] = {}

    if not force:
        a_npz = str(OUT_DIR / "audio_embeddings.npz")
        t_npz = str(OUT_DIR / "text_embeddings.npz")
        if Path(a_npz).exists():
            audio_embs = load_embeddings_npz(a_npz)
        if Path(t_npz).exists():
            text_embs  = load_embeddings_npz(t_npz)

    to_process = []
    for artist_code, artist_data in CATALOG.items():
        folder = artist_data["folder"]
        for song_id in artist_data["songs"]:
            if not force and song_id in done:
                continue
            ap = get_audio_path(song_id, folder)
            if not ap:
                continue
            if song_id not in all_lyrics:
                continue
            to_process.append((song_id, ap, all_lyrics[song_id]["lyrics_clean"]))

    if not to_process:
        logger.info("Nothing to process.")
        return {}

    logger.info(f"Loading models for {len(to_process)} songs…")
    mert_model, mert_proc = load_mert(device)
    sbert_model           = load_sbert(device)

    sr   = MODEL_CFG["audio_sr"]
    n_ok = 0

    for i, (song_id, audio_path, lyrics) in enumerate(to_process, 1):
        logger.info(f"[{i}/{len(to_process)}] {song_id}")

        a_emb = get_mert_embedding(mert_model, mert_proc, audio_path, device, sr=sr)
        t_emb = get_sbert_embedding(sbert_model, lyrics)

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
    logger.info(f"\nMERT+SBERT done. {len(sims)} songs. "
                f"Mean={np.mean(vals):.4f}  Std={np.std(vals):.4f}")

    return {"audio": audio_embs, "text": text_embs, "sims": sims}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MERT+SBERT embeddings")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    embed_all(device=args.device, force=args.force)
