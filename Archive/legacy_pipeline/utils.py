"""
utils.py — Shared utilities for the Musical Congruence pipeline.

Handles:
  • MPS / CUDA / CPU device selection with op-level fallback warnings
  • Safe text embedding with automatic chunking for token-limit errors
  • Cosine similarity helpers
  • Lyrics section parsing (for segment analysis)
  • Checkpoint / resume helpers so expensive embedding runs can be interrupted
"""

from __future__ import annotations
import os
import re
import json
import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Callable, Any

logger = logging.getLogger(__name__)

# ─── Device detection ─────────────────────────────────────────────────────────

def get_device(prefer_mps: bool = True) -> str:
    """
    Returns 'cuda', 'mps', or 'cpu'.

    MPS NOTE: Apple Silicon supports float32 ops but not all float16 ops.
    Always load models in float32 on MPS. Some ops may silently fall back
    to CPU; set PYTORCH_ENABLE_MPS_FALLBACK=1 to allow this.
    """
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        # Enable fallback for unsupported ops
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def to_device_safe(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Move tensor to device, falling back to CPU if MPS raises."""
    try:
        return tensor.to(device)
    except Exception as e:
        if device == "mps":
            logger.warning(f"MPS move failed ({e}), falling back to CPU.")
            return tensor.to("cpu")
        raise


def model_to_device(model, device: str):
    """Move model to device; cast to float32 on MPS to avoid fp16 issues."""
    if device == "mps":
        model = model.float()   # MPS does not fully support fp16
    return model.to(device).eval()


# ─── Cosine similarity ────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_matrix(audio_embs: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
    """
    Compute N×N cosine similarity matrix.
    audio_embs: [N, D], text_embs: [N, D]
    Returns: [N, N] similarity matrix
    """
    # L2-normalise
    audio_norm = audio_embs / (np.linalg.norm(audio_embs, axis=1, keepdims=True) + 1e-9)
    text_norm  = text_embs  / (np.linalg.norm(text_embs,  axis=1, keepdims=True) + 1e-9)
    return audio_norm @ text_norm.T


# ─── Text splitting (for token-limit-aware embedding) ─────────────────────────

def split_text_chunks(text: str, max_words: int = 200) -> list[str]:
    """
    Split text into overlapping word chunks of at most max_words.
    Overlap is 10% of max_words to preserve context at boundaries.
    """
    words  = text.split()
    if len(words) <= max_words:
        return [text]
    step    = max(1, int(max_words * 0.9))
    chunks  = []
    start   = 0
    while start < len(words):
        chunk = words[start : start + max_words]
        chunks.append(" ".join(chunk))
        start += step
    return chunks


def embed_text_safe(
    embed_fn: Callable[[list[str]], np.ndarray],
    text: str,
    max_words: int = 200,
) -> np.ndarray:
    """
    Embed text with automatic chunking if it exceeds max_words.
    embed_fn should accept a list of strings and return [N, D] numpy array.
    Returns a single [D] vector (mean of chunk embeddings).
    """
    chunks = split_text_chunks(text, max_words)
    if len(chunks) == 1:
        return embed_fn([text])[0]
    embeddings = embed_fn(chunks)   # [C, D]
    return embeddings.mean(axis=0)  # [D]


# ─── Lyrics section parser ────────────────────────────────────────────────────

SECTION_HEADER_RE = re.compile(r'\[([^\]]+)\]')

def parse_lyric_sections(lyrics: str, min_words: int = 5) -> list[dict]:
    """
    Parse Genius-style lyrics with [Section Header] markers into a list of
    {'header': str, 'text': str, 'section_type': str} dicts.

    section_type is normalised to one of:
      'intro' | 'verse' | 'pre-chorus' | 'chorus' | 'bridge' | 'outro' | 'other'

    Sections with fewer than min_words of content are skipped.
    """
    KNOWN_TYPES = ["intro", "verse", "pre-chorus", "pre chorus", "chorus",
                   "bridge", "hook", "refrain", "outro", "break", "interlude"]

    def normalise(header: str) -> str:
        h = header.lower().strip()
        for t in KNOWN_TYPES:
            if t in h:
                return t.replace(" ", "-")
        return "other"

    parts    = SECTION_HEADER_RE.split(lyrics)
    sections = []

    # parts alternates: [pre-header-text, header1, content1, header2, content2, ...]
    i = 0
    # handle any preamble before the first header
    if parts and not SECTION_HEADER_RE.search("[" + parts[0] + "]"):
        preamble = parts[0].strip()
        if len(preamble.split()) >= min_words:
            sections.append({"header": "Intro", "text": preamble, "section_type": "intro"})
        i = 1

    while i + 1 < len(parts):
        header  = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if len(content.split()) >= min_words:
            sections.append({
                "header":       header,
                "text":         content,
                "section_type": normalise(header),
            })
        i += 2

    # If no headers found at all, return the whole lyrics as one block
    if not sections and len(lyrics.split()) >= min_words:
        sections = [{"header": "Full", "text": lyrics.strip(), "section_type": "other"}]

    return sections


def strip_section_headers(lyrics: str) -> str:
    """Remove [Header] markers but keep lyric text — for track-level embedding."""
    return SECTION_HEADER_RE.sub("", lyrics).strip()


# ─── Audio chunking for models with a max window (e.g. CLAP: 10s) ─────────────

def chunk_audio(wav: np.ndarray, sr: int, chunk_s: float = 10.0,
                overlap_s: float = 1.0) -> list[np.ndarray]:
    """
    Split a 1-D audio array into overlapping chunks of chunk_s seconds.
    Returns list of arrays; any chunk shorter than 0.5 s is discarded.
    """
    chunk_samples   = int(chunk_s   * sr)
    overlap_samples = int(overlap_s * sr)
    step            = chunk_samples - overlap_samples
    min_len         = int(0.5 * sr)

    chunks = []
    start  = 0
    while start < len(wav):
        chunk = wav[start : start + chunk_samples]
        if len(chunk) >= min_len:
            # Pad short final chunk to full length with zeros
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)
        start += step
    return chunks


# ─── Checkpoint / resume helpers ──────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    """Load a JSON checkpoint, return {} if it doesn't exist."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict, path: str) -> None:
    """Save a JSON checkpoint atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(p))


# ─── Embedding I/O ────────────────────────────────────────────────────────────

def save_embeddings_npz(embeddings: dict[str, np.ndarray], path: str) -> None:
    """Save {song_id: embedding_array} to .npz."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **embeddings)
    logger.info(f"Saved embeddings → {path}")


def load_embeddings_npz(path: str) -> dict[str, np.ndarray]:
    """Load .npz into {song_id: embedding_array} dict."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ─── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ─── Song-level similarity extraction ────────────────────────────────────────

def diagonal_similarities(
    audio_embs: dict[str, np.ndarray],
    text_embs:  dict[str, np.ndarray],
) -> dict[str, float]:
    """
    For each song_id present in both dicts, return the cosine similarity
    of its own audio embedding vs. its own text embedding.
    (i.e. the diagonal of the N×N similarity matrix.)
    """
    ids  = [sid for sid in audio_embs if sid in text_embs]
    sims = {}
    for sid in ids:
        sims[sid] = cosine_sim(audio_embs[sid], text_embs[sid])
    return sims
