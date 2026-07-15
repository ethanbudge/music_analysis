"""
lyriclmc.py — Step 3: a small bespoke contrastive lyric<->audio model.

WHAT THIS IS (in plain terms)
-----------------------------
We build our own tiny joint embedding space where "close" means "these lyrics and
this music go together". Unlike MuLan, the text side sees real lyrics through a
sentence encoder, so lyrics are in-distribution.

We do NOT retrain the big encoders. We take:
  * a FROZEN audio vector per song  (MERT, already cached)         -> 1024-d
  * a FROZEN lyric vector per song  (sentence encoder, cached)     ->  384-d
and learn only two small "projection heads" (a couple of matrix layers) that map
each into a shared d-dimensional space. That is the entire trainable model — a few
hundred thousand numbers. It trains in minutes on CPU/MPS and, importantly, touches
only cached EMBEDDINGS, never raw audio (a compute win and a legal win).

HOW IT LEARNS (contrastive / "CLIP" loss)
-----------------------------------------
Each training batch is B songs, each a (audio, lyrics) pair. We compute all B x B
similarities between projected audio and projected lyrics. The model is rewarded for
making each song's OWN pair the most similar in its row and column, and pushing the
other B-1 (mismatched) pairs down. Those B-1 mismatches are the "negatives". Over
many batches the space arranges itself so true pairs sit close and mismatches far —
which is exactly congruence.

WHY THIS IS THE RIGHT SHAPE FOR YOUR THESIS
-------------------------------------------
The output is a full high-dimensional embedding geometry (not a 2-number readout),
so it preserves the "holistic representation" idea — but with a text tower that
finally understands lyrics. And the training objective IS the matched-vs-mismatched
test from evaluate.py, so held-out retrieval AUC is an honest quality measure.

Everything about a run (weights, config, the train/val split, the learning curve) is
saved to its own timestamped folder under data/lmcmetrics/lyriclmc/, so you can keep
and compare iterations.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from . import evaluate

logger = logging.getLogger(__name__)


# ── the model ─────────────────────────────────────────────────────────────────────
def _build_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class ProjectionHead:
    """Factory for a small MLP head (defined lazily so torch import stays optional)."""
    @staticmethod
    def make(in_dim: int, hidden: int, out_dim: int, dropout: float):
        torch, nn = _build_torch()

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, out_dim),
                )

            def forward(self, x):
                z = self.net(x)
                return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)   # L2-normalise

        return _Head()


def _make_model(audio_dim: int, text_dim: int, cfg: dict):
    torch, nn = _build_torch()

    class LyricLMCNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_head = ProjectionHead.make(audio_dim, cfg["hidden_dim"],
                                                  cfg["proj_dim"], cfg["dropout"])
            self.text_head = ProjectionHead.make(text_dim, cfg["hidden_dim"],
                                                 cfg["proj_dim"], cfg["dropout"])
            # Learnable temperature (as in CLIP): stored in log-space, clamped so it
            # can't collapse to a degenerate value.
            self.logit_scale = nn.Parameter(torch.tensor(float(cfg["init_logit_scale"])))

        def forward(self, a, t):
            return self.audio_head(a), self.text_head(t)

        def clamped_scale(self):
            return self.logit_scale.clamp(max=float(cfg["max_logit_scale"])).exp()

    return LyricLMCNet()


def info_nce(za, zt, scale):
    """Symmetric contrastive loss over a batch of aligned rows (the CLIP loss)."""
    torch, nn = _build_torch()
    logits = scale * za @ zt.t()                 # [B, B] similarities
    labels = torch.arange(za.shape[0], device=za.device)
    return 0.5 * (nn.functional.cross_entropy(logits, labels) +
                  nn.functional.cross_entropy(logits.t(), labels))


# ── trained-run container (weights live on disk; this is the load/score handle) ──
@dataclass
class LyricLMC:
    run_dir: Path
    cfg: dict
    audio_dim: int
    text_dim: int
    _net: object = None
    _device: str = "cpu"

    # -- projection / scoring (numpy in, numpy out) --
    def _ensure_net(self):
        if self._net is None:
            self.load_weights()

    def project(self, audio: np.ndarray, text: np.ndarray):
        """Map cached (audio, lyric) vectors into the learned shared space."""
        torch, _ = _build_torch()
        self._ensure_net()
        A = torch.tensor(np.atleast_2d(audio), dtype=torch.float32, device=self._device)
        T = torch.tensor(np.atleast_2d(text), dtype=torch.float32, device=self._device)
        with torch.no_grad():
            za, zt = self._net(A, T)
        return za.cpu().numpy(), zt.cpu().numpy()

    def score_pairs(self, audio: np.ndarray, text: np.ndarray) -> np.ndarray:
        """Congruence for aligned rows: cosine of projected audio vs its own lyrics."""
        za, zt = self.project(audio, text)
        return np.sum(za * zt, axis=1)

    def score_matrix_from_reps(self, audio: np.ndarray, text: np.ndarray) -> np.ndarray:
        """Full [N,N] matrix S[i,j] = cos(project_audio_i, project_text_j)."""
        za, zt = self.project(audio, text)
        return za @ zt.T

    # -- persistence --
    def load_weights(self):
        torch, _ = _build_torch()
        from lmc.utils import get_device
        self._device = get_device()
        self._net = _make_model(self.audio_dim, self.text_dim, self.cfg).to(self._device).eval()
        state = torch.load(self.run_dir / "model.pt", map_location=self._device)
        self._net.load_state_dict(state)
        return self

    @classmethod
    def load_run(cls, run_dir: str | Path) -> "LyricLMC":
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "config.json").read_text())
        obj = cls(run_dir=run_dir, cfg=meta["cfg"],
                  audio_dim=meta["audio_dim"], text_dim=meta["text_dim"])
        return obj.load_weights()


# ── training ──────────────────────────────────────────────────────────────────────
def _val_auc(net, A_val, T_val, device):
    """Held-out matched-vs-mismatched AUC using the Step-2 harness."""
    torch, _ = _build_torch()
    net.eval()
    with torch.no_grad():
        za, zt = net(torch.tensor(A_val, dtype=torch.float32, device=device),
                     torch.tensor(T_val, dtype=torch.float32, device=device))
    S = za.cpu().numpy() @ zt.cpu().numpy().T
    return evaluate.retrieval_metrics(S)["auc_mean"]


def train_lyriclmc(audio: np.ndarray, lyric: np.ndarray, track_ids: list[int],
                   cfg: dict | None = None, run_name: str | None = None,
                   save: bool = True) -> tuple[LyricLMC, dict]:
    """Train the projection heads on cached (audio, lyric) pairs.

    Parameters
    ----------
    audio : [N, audio_dim]  cached audio vectors (e.g. MERT), row-aligned to lyric.
    lyric : [N, text_dim]   cached lyric sentence vectors, same row order.
    track_ids : the N track ids, so the saved split is reproducible.

    Returns (LyricLMC handle, history dict). If save=True, writes a run folder.
    """
    torch, _ = _build_torch()
    from lmc.utils import get_device
    cfg = {**config.LYRICLMC, **(cfg or {})}
    device = get_device()
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    A = np.asarray(audio, dtype=np.float32)
    T = np.asarray(lyric, dtype=np.float32)
    N, audio_dim, text_dim = A.shape[0], A.shape[1], T.shape[1]
    assert T.shape[0] == N == len(track_ids), "audio, lyric, track_ids must align."

    # Train / validation split (held out for early stopping and honest AUC).
    perm = rng.permutation(N)
    n_val = max(8, int(round(cfg["val_frac"] * N)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    A_tr, T_tr = A[tr_idx], T[tr_idx]
    A_val, T_val = A[val_idx], T[val_idx]

    net = _make_model(audio_dim, text_dim, cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history = {"epoch": [], "train_loss": [], "val_auc": []}
    best_auc, best_state, bad = -1.0, None, 0
    bs = min(cfg["batch_size"], len(tr_idx))

    logger.info("LyricLMC: N=%d (train=%d, val=%d), audio_dim=%d, text_dim=%d, d=%d, device=%s",
                N, len(tr_idx), len(val_idx), audio_dim, text_dim, cfg["proj_dim"], device)

    for epoch in range(cfg["epochs"]):
        net.train()
        order = rng.permutation(len(tr_idx))
        losses = []
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            if len(b) < 2:                    # need >=2 for in-batch negatives
                continue
            a = torch.tensor(A_tr[b], dtype=torch.float32, device=device)
            t = torch.tensor(T_tr[b], dtype=torch.float32, device=device)
            za, zt = net(a, t)
            loss = info_nce(za, zt, net.clamped_scale())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        vauc = _val_auc(net, A_val, T_val, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["val_auc"].append(vauc)
        if vauc > best_auc + 1e-4:
            best_auc, bad = vauc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if epoch % 5 == 0 or bad == 0:
            logger.info("  epoch %3d | loss %.4f | val_auc %.4f%s",
                        epoch, history["train_loss"][-1], vauc, "  *" if bad == 0 else "")
        if bad >= cfg["patience"]:
            logger.info("  early stop at epoch %d (best val_auc=%.4f)", epoch, best_auc)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    history["best_val_auc"] = best_auc

    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = config.LYRICLMC_RUNS_DIR / run_name
    if save:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), run_dir / "model.pt")
        (run_dir / "config.json").write_text(json.dumps({
            "cfg": cfg, "audio_dim": audio_dim, "text_dim": text_dim,
            "audio_rep": config.AUDIO_REP, "text_encoder": config.TEXT_ENCODER,
            "best_val_auc": best_auc, "created": datetime.now().isoformat(),
        }, indent=2))
        (run_dir / "split.json").write_text(json.dumps({
            "train_track_ids": [int(track_ids[i]) for i in tr_idx],
            "val_track_ids":   [int(track_ids[i]) for i in val_idx],
        }, indent=2))
        import csv
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_auc"])
            for e, l, v in zip(history["epoch"], history["train_loss"], history["val_auc"]):
                w.writerow([e, l, v])
        logger.info("LyricLMC saved to %s (best val_auc=%.4f)", run_dir, best_auc)

    handle = LyricLMC(run_dir=run_dir, cfg=cfg, audio_dim=audio_dim, text_dim=text_dim)
    handle._net, handle._device = net, device
    return handle, history
