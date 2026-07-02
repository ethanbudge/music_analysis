"""
utils.py — Shared helpers: device selection, embedding I/O, cosine similarity,
LRC parsing, and text chunking.

Carried over (and trimmed) from the original pipeline. The MERT+SBERT helpers
and the Genius section-header parser are gone; chorus detection now lives in
chorus.py and works from LRCLIB synced timestamps instead of bracketed headers.
"""

from __future__ import annotations
import os
import re
import logging
import numpy as np
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ─── Logging ─────────────────────────────────────────────────────────────────────
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ─── Torch device selection ──────────────────────────────────────────────────────
def get_device(prefer_mps: bool = True) -> str:
    """Return 'cuda', 'mps', or 'cpu'. Enables MPS CPU-fallback for unsupported ops."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def model_to_device(model, device: str):
    """Move model to device; force float32 on MPS (fp16 is not fully supported)."""
    if device == "mps":
        model = model.float()
    return model.to(device).eval()


# ─── Cosine similarity ───────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (the LMC measure)."""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


# ─── Text chunking (token-limit-aware embedding) ─────────────────────────────────
def split_text_chunks(text: str, max_words: int = 200) -> list[str]:
    """Split text into overlapping word chunks of at most max_words (10% overlap)."""
    words = text.split()
    if len(words) <= max_words:
        return [text] if text.strip() else []
    step, chunks, start = max(1, int(max_words * 0.9)), [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_words]))
        start += step
    return chunks


def embed_text_safe(embed_fn: Callable[[list[str]], np.ndarray], text: str,
                    max_words: int = 200) -> np.ndarray | None:
    """Embed text, chunk-averaging if it exceeds max_words. Returns [D] or None."""
    chunks = split_text_chunks(text, max_words)
    if not chunks:
        return None
    try:
        if len(chunks) == 1:
            return embed_fn([chunks[0]])[0]
        return embed_fn(chunks).mean(axis=0)
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"  text embed failed ({e}); retrying with smaller chunks")
        vecs = []
        for c in split_text_chunks(text, max_words // 2):
            try:
                vecs.append(embed_fn([c])[0])
            except Exception:                                  # noqa: BLE001
                pass
        return np.stack(vecs).mean(axis=0) if vecs else None


def ascii_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are ASCII (crude Latin-script filter)."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(c.isascii() for c in alpha) / len(alpha)


# ─── LRC (time-synced lyrics) parsing ────────────────────────────────────────────
_LRC_TS_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")

def parse_lrc(synced: str) -> list[dict]:
    """
    Parse an LRC-format synced-lyrics string into time-ordered lines.

    Returns a list of {"idx", "start", "end", "text"} dicts (blank lines dropped).
    `end` is the next line's start time; the final line's end is left as None and
    should be clamped to the audio duration by the caller.
    A single LRC line may carry multiple timestamps (repeats) — each is emitted.
    """
    events: list[tuple[float, str]] = []
    for raw in synced.splitlines():
        stamps = _LRC_TS_RE.findall(raw)
        if not stamps:
            continue
        text = _LRC_TS_RE.sub("", raw).strip()
        if not text:
            continue
        for mm, ss in stamps:
            events.append((int(mm) * 60 + float(ss), text))

    events.sort(key=lambda e: e[0])
    lines = []
    for i, (start, text) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else None
        lines.append({"idx": i, "start": start, "end": end, "text": text})
    return lines


def lrc_to_plaintext(synced: str) -> str:
    """Strip timestamps from an LRC string to recover plain running lyrics."""
    return "\n".join(ln["text"] for ln in parse_lrc(synced))


def normalise_line(text: str) -> str:
    """Lowercase, strip punctuation/whitespace — for repeated-line (chorus) matching."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# ─── Embedding I/O (one .npz per song per model) ─────────────────────────────────
def embedding_path(embeddings_dir, model_key: str, track_id: int) -> Path:
    return Path(embeddings_dir) / model_key / f"{track_id}.npz"


def save_song_embeddings(path, arrays: dict[str, np.ndarray]) -> None:
    """Atomically save a song's embedding bundle to .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends ".npz" unless the name already ends in it, so the
    # temp name must end in ".npz" or the file numpy writes won't match `tmp`.
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_song_embeddings(path) -> dict[str, np.ndarray] | None:
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
