"""
models.py — the four embedding models under test, behind one interface.

Every model exposes:
    embed_audio_batch(wavs: list[np.ndarray], sr: int) -> np.ndarray [N, D]
    embed_text_batch(texts: list[str])                  -> np.ndarray [N, D]

LMC is then cosine(audio_vec, text_vec) within a single model's space (each model
lives in its own space; we never compare across models).

  mulan       reuses lmc.embeddings._MuLan   (MuQ-MuLan)          — reuse
  laion_clap  reuses lmc.embeddings._CLAP    (LAION-CLAP music)   — reuse
  ms_clap     Microsoft CLAP via the `msclap` package             — new
  clamp3      CLaMP 3 via its clamp3_embd.py script (subprocess)   — new

A model that can't load (e.g. CLaMP 3 not configured) is simply absent from the
returned registry; the compute layer fills its columns with NaN and carries on.
`MockModel` gives deterministic vectors for offline plumbing tests.
"""

from __future__ import annotations
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def _resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return wav
    import librosa
    return librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=target)


def _nan(dim: int) -> np.ndarray:
    return np.full(dim, np.nan, dtype=np.float32)


# ── reused pipeline models ────────────────────────────────────────────────────────
class MuLanModel:
    name = "mulan"
    target_sr = 24_000

    def __init__(self, device: str):
        from lmc.embeddings import _MuLan
        self.m = _MuLan(device)
        self.dim = self.m.dim

    def embed_audio_batch(self, wavs, sr):
        out = []
        for w in wavs:
            v = self.m.embed_audio(_resample(w, sr, self.target_sr))
            out.append(v if v is not None else _nan(self.dim))
        return np.stack(out)

    def embed_text_batch(self, texts):
        out = [self.m.embed_text(t) for t in texts]
        return np.stack([v if v is not None else _nan(self.dim) for v in out])


class LaionClapModel:
    name = "laion_clap"
    target_sr = 48_000

    def __init__(self, device: str):
        from lmc.embeddings import _CLAP
        self.m = _CLAP(device)                 # loads the music checkpoint via lmc.config
        self.dim = self.m.dim

    def embed_audio_batch(self, wavs, sr):
        out = []
        for w in wavs:
            v = self.m.embed_audio(_resample(w, sr, self.target_sr))   # _CLAP chunks internally
            out.append(v if v is not None else _nan(self.dim))
        return np.stack(out)

    def embed_text_batch(self, texts):
        out = [self.m.embed_text(t) for t in texts]
        return np.stack([v if v is not None else _nan(self.dim) for v in out])


# ── Microsoft CLAP (msclap) ───────────────────────────────────────────────────────
class MsClapModel:
    name = "ms_clap"
    target_sr = 44_100
    chunk_s = 7.0        # msclap uses a fixed ~7 s window; chunk long audio + average

    def __init__(self, device: str):
        from msclap import CLAP
        use_cuda = device == "cuda"
        logger.info("Loading Microsoft CLAP (version %s)…", config.MSCLAP_VERSION)
        self.m = CLAP(version=config.MSCLAP_VERSION, use_cuda=use_cuda)
        self._tmp = Path(tempfile.mkdtemp(prefix="msclap_", dir=config.CACHE_DIR))
        # probe embedding dim
        self.dim = int(self.embed_text_batch(["probe"]).shape[1])

    def _chunks(self, wav, sr):
        n = int(self.chunk_s * sr)
        if len(wav) <= n:
            return [wav]
        return [wav[i:i + n] for i in range(0, len(wav), n) if len(wav[i:i + n]) >= int(0.5 * sr)]

    def embed_audio_batch(self, wavs, sr):
        import soundfile as sf
        paths, groups = [], []          # groups[i] = list of temp-file indices for wav i
        for wav in wavs:
            idx = []
            for ch in self._chunks(_resample(wav, sr, self.target_sr), self.target_sr):
                p = self._tmp / f"a{len(paths):05d}.wav"
                sf.write(p, np.asarray(ch, dtype=np.float32), self.target_sr)
                idx.append(len(paths)); paths.append(str(p))
            groups.append(idx)
        emb = self.m.get_audio_embeddings(paths)            # [sum_chunks, D] tensor
        emb = np.asarray(emb.detach().cpu().numpy() if hasattr(emb, "detach") else emb)
        return np.stack([emb[idx].mean(axis=0) for idx in groups])

    def _embed_one_text(self, text: str) -> np.ndarray:
        # msclap is caption-oriented; embed EACH text on its own (never batch texts of
        # different token lengths — that triggers a torch.stack size mismatch), and
        # chunk long lyrics into <=200-word pieces + average, like the MuLan/CLAP paths.
        from lmc.utils import split_text_chunks
        chunks = split_text_chunks(text, 200) or [text or " "]
        vecs = []
        for ch in chunks:
            emb = self.m.get_text_embeddings([ch])
            vecs.append(np.asarray(emb.detach().cpu().numpy() if hasattr(emb, "detach") else emb)[0])
        return np.mean(vecs, axis=0)

    def embed_text_batch(self, texts):
        out = []
        for t in texts:
            try:
                out.append(self._embed_one_text(t))
            except Exception as e:                             # noqa: BLE001
                logger.warning("  ms_clap text embed failed (%s) — NaN.", e)
                out.append(_nan(self.dim))
        return np.stack(out)


# ── CLaMP 3 (subprocess to clamp3_embd.py) ────────────────────────────────────────
class Clamp3Model:
    name = "clamp3"
    target_sr = 24_000       # written to disk; CLaMP 3 extracts MERT features itself
    dim = 768

    def __init__(self, device: str):
        if not config.CLAMP3_REPO:
            raise RuntimeError("CLAMP3_REPO not set (clone sanderwood/clamp3 and set "
                               "LMCVAL_CLAMP3_REPO).")
        self.repo = Path(config.CLAMP3_REPO).expanduser()
        if not (self.repo / config.CLAMP3_SCRIPT).exists():
            raise RuntimeError(f"{config.CLAMP3_SCRIPT} not found in {self.repo}.")
        self.python = self._resolve_python(config.CLAMP3_PYTHON)

    @staticmethod
    def _resolve_python(p: str) -> str:
        """Accept either the env's python binary or the env dir; validate it exists."""
        pp = Path(p).expanduser()
        if pp.is_dir():                       # pointed at the env FOLDER → use its python
            pp = pp / "bin" / "python"
        if not pp.exists():
            raise RuntimeError(
                f"CLaMP 3 python not found at '{pp}'. Set LMCVAL_CLAMP3_PYTHON to the "
                f"env's python BINARY, e.g. /opt/anaconda3/envs/clamp3/bin/python "
                f"(find the base with `conda info --envs`).")
        return str(pp)

    def _run(self, in_dir: Path, out_dir: Path):
        # extract_mert.py's error_log.txt is append-only across runs, so clear it first
        # — otherwise a warning below could just be stale noise from a PAST failed run.
        for log_path in self.repo.glob("**/logs/error_log.txt"):
            log_path.unlink(missing_ok=True)

        cmd = [self.python, config.CLAMP3_SCRIPT, str(in_dir), str(out_dir), "--get_global"]
        env = os.environ.copy()
        # clamp3_embd.py internally calls a bare `python extract_mert.py`, so put the
        # CLaMP 3 env's bin dir first on PATH; and let MPS fall back to CPU for any
        # unsupported ops on Apple Silicon.
        env["PATH"] = str(Path(self.python).resolve().parent) + os.pathsep + env.get("PATH", "")
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        result = subprocess.run(cmd, cwd=str(self.repo), check=True, env=env,
                                capture_output=True, text=True, timeout=3600)
        # clamp3_embd.py's own extractors (extract_mert.py / extract_clamp3.py) catch
        # per-file errors internally and just log them, so the outer process can exit
        # 0 even when EVERY file failed (e.g. a missing torchaudio/torchcodec backend).
        # Surface that here rather than silently returning NaN with no explanation.
        out = (result.stdout or "") + (result.stderr or "")
        if "Found 0 files in total" in out:
            logger.warning("  CLaMP 3 reported 0 input files — check %s is reachable "
                           "from its own process.", in_dir)
        for log_name in ("error_log.txt",):
            for log_path in self.repo.glob(f"**/logs/{log_name}"):
                try:
                    tail = log_path.read_text().strip().splitlines()[-3:]
                except Exception:                              # noqa: BLE001
                    continue
                if tail:
                    logger.warning("  CLaMP 3 %s (last lines): %s", log_path, " | ".join(tail))

    def _collect(self, stems: list[str], out_dir: Path) -> np.ndarray:
        vecs, missing = [], []
        for stem in stems:
            hits = list(out_dir.glob(f"{stem}.npy"))
            if hits:
                vecs.append(np.load(hits[0]).reshape(-1).astype(np.float32))
            else:
                vecs.append(_nan(self.dim))
                missing.append(stem)
        if missing:
            logger.warning("  CLaMP 3 produced no output for %d/%d item(s) (e.g. %s) — "
                           "those get NaN. clamp3_embd.py can exit 0 even when every file "
                           "failed internally (see the warnings above, if any, or check "
                           "%s/**/logs/*.txt).", len(missing), len(stems), missing[:3], self.repo)
        return np.stack(vecs)

    def _embed_files(self, write_fn, items) -> np.ndarray:
        work = Path(tempfile.mkdtemp(prefix="clamp3_", dir=config.CACHE_DIR))
        in_dir, out_dir = work / "in", work / "out"
        # NB: clamp3_embd.py SKIPS extraction if the output dir already exists, so we
        # must NOT pre-create out_dir — only the input dir.
        in_dir.mkdir()
        stems = [f"{i:05d}" for i in range(len(items))]
        for stem, item in zip(stems, items):
            write_fn(in_dir, stem, item)
        try:
            self._run(in_dir, out_dir)
        except Exception as e:                                 # noqa: BLE001
            logger.warning("  CLaMP 3 run failed (%s) — filling NaN.",
                           getattr(e, "stderr", None) or e)
            return np.stack([_nan(self.dim) for _ in items])
        return self._collect(stems, out_dir)

    def embed_audio_batch(self, wavs, sr):
        import soundfile as sf

        def _w(in_dir, stem, wav):
            sf.write(in_dir / f"{stem}.wav", _resample(wav, sr, self.target_sr), self.target_sr)
        return self._embed_files(_w, list(wavs))

    def embed_text_batch(self, texts):
        def _w(in_dir, stem, text):
            (in_dir / f"{stem}.txt").write_text(text or " ", encoding="utf-8")
        return self._embed_files(_w, list(texts))


# ── mock (offline plumbing tests) ─────────────────────────────────────────────────
class MockModel:
    """Deterministic vectors from simple audio stats / text hashes (no downloads)."""
    def __init__(self, name="mock", dim=32, seed=0):
        self.name, self.dim, self.seed = name, dim, seed

    def _vec(self, key: int) -> np.ndarray:
        return np.random.default_rng(self.seed + key).standard_normal(self.dim).astype(np.float32)

    def embed_audio_batch(self, wavs, sr):
        out = []
        for w in wavs:
            k = int(abs(np.nan_to_num(np.mean(w)) * 1e4) + len(w)) % 100_000
            out.append(self._vec(k))
        return np.stack(out)

    def embed_text_batch(self, texts):
        return np.stack([self._vec(hash(t) % 100_000) for t in texts])


_REGISTRY = {"mulan": MuLanModel, "laion_clap": LaionClapModel,
             "ms_clap": MsClapModel, "clamp3": Clamp3Model}


def load_models(names: list[str] | None = None, device: str | None = None,
                mock: bool = False) -> dict:
    """Instantiate the requested models; skip (with a warning) any that fail to load."""
    from lmc.utils import get_device
    names = names or config.MODEL_KEYS
    device = device or get_device()
    if mock:
        return {n: MockModel(name=n, seed=i) for i, n in enumerate(names)}
    out = {}
    for n in names:
        try:
            out[n] = _REGISTRY[n](device)
            logger.info("  loaded %s.", config.MODEL_DISPLAY.get(n, n))
        except Exception as e:                                 # noqa: BLE001
            logger.warning("  could NOT load %s (%s) — its columns will be NaN.",
                           config.MODEL_DISPLAY.get(n, n), e)
    return out
