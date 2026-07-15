"""
run.py — end-to-end driver for the alternative congruence metrics on the REAL corpus.

This is the script you run (in the `lmc` conda env, after the observational pipeline
has produced MERT vectors and MuLan/CLAP bundles). It:

  1. caches a lyric sentence-vector per song            (the new text tower)
  2. trains LyricLMC on cached (MERT, lyric) pairs       (Step 3)
  3. evaluates EVERY metric on the SAME held-out songs   (Step 2 yardstick)
       - mulan_raw      : your current LMC (cosine, no fix)   <- the baseline to beat
       - mulan_centered : + Step-1 modality-gap fix
       - clap_raw / clap_centered
       - lyriclmc       : the learned lyric<->audio space
  4. computes global CKA/RSA + per-song local RSA        (Step 4 geometry)
  5. exports per-song congruence columns to results/lmcmetrics/congruence_scores.csv
     (merge into master_results.csv by track_id for the Stan models / survey analysis)

Everything is resumable and reads only cached artifacts. Typical use:

    import sys; sys.path.insert(0, "src")
    from lmcmetrics import run
    run.run_all()                      # does all of the above, writes CSVs

Or step by step (see the functions below). Nothing here modifies the `lmc` pipeline.
"""

from __future__ import annotations
import json
import logging

import numpy as np
import pandas as pd

from . import config, data, evaluate, geometry, lyric_encoder, lyriclmc, scorers
from .centering import Centerer

logger = logging.getLogger(__name__)


# ── rep loading (build aligned maps over the corpus) ─────────────────────────────
def _load_reps(limit: int | None = None) -> dict:
    """Gather every cached representation keyed by track_id, plus genre labels."""
    from lmc import db as projdb, mert as mert_mod
    from lmc.utils import load_song_embeddings, embedding_path

    with projdb.connect() as conn:
        tids = [r["track_id"] for r in conn.execute("SELECT track_id FROM songs ORDER BY track_id")]
    if limit:
        tids = tids[:limit]

    reps = {"mulan_a": {}, "mulan_t": {}, "clap_a": {}, "clap_t": {},
            "mert": {}, "lyric": {}, "genre": {}}
    reps["mert"] = mert_mod.load_vectors(tids)
    reps["lyric"] = lyric_encoder.load_vectors(tids)
    for model in ("mulan", "clap"):
        for tid in tids:
            b = load_song_embeddings(embedding_path(config.EMBEDDINGS_DIR, model, tid))
            if b is None:
                continue
            a, t = b.get("audio_full"), b.get("text_full")
            if a is not None and t is not None and np.any(a) and np.any(t):
                reps[f"{model}_a"][tid] = a
                reps[f"{model}_t"][tid] = t
    reps["genre"] = data._genre_map()
    reps["ids"] = tids
    return reps


def _mat(repmap: dict, ids: list[int]) -> np.ndarray:
    return np.stack([repmap[t] for t in ids])


# ── Step 2: build every metric's score matrix on a common id set ─────────────────
def _score_matrices(reps: dict, ids: list[int], lmc_model=None) -> dict:
    """Return {metric_name: S[N,N]} for all metrics that have reps for every id."""
    out = {}

    def _have(key):
        return all(t in reps[key] for t in ids)

    if _have("mulan_a") and _have("mulan_t"):
        A, T = _mat(reps["mulan_a"], ids), _mat(reps["mulan_t"], ids)
        out["mulan_raw"] = evaluate.cosine_score_matrix(A, T)
        c = Centerer(mode="mean").fit(A, T)
        out["mulan_centered"] = evaluate.cosine_score_matrix(A, T, centerer=c)
    if _have("clap_a") and _have("clap_t"):
        A, T = _mat(reps["clap_a"], ids), _mat(reps["clap_t"], ids)
        out["clap_raw"] = evaluate.cosine_score_matrix(A, T)
        c = Centerer(mode="mean").fit(A, T)
        out["clap_centered"] = evaluate.cosine_score_matrix(A, T, centerer=c)
    if lmc_model is not None and _have("mert") and _have("lyric"):
        A, T = _mat(reps["mert"], ids), _mat(reps["lyric"], ids)
        out["lyriclmc"] = lmc_model.score_matrix_from_reps(A, T)
    return out


def evaluate_on_ids(reps: dict, ids: list[int], lmc_model=None,
                    restrict: str | None = None) -> pd.DataFrame:
    """Tabulate matched-vs-mismatched metrics for every scorer on the same songs."""
    groups = np.asarray([reps["genre"].get(t, "unknown") for t in ids])
    mats = _score_matrices(reps, ids, lmc_model=lmc_model)
    rows = []
    for name, S in mats.items():
        m = evaluate.summarize(evaluate.retrieval_metrics(S, groups=groups, restrict=restrict))
        m["metric"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("metric").sort_values("auc_mean", ascending=False)
    front = ["auc_mean", "auc", "auc_t2a", "recall@1", "recall@5", "median_rank", "mrr", "n"]
    return df[[c for c in front if c in df.columns]]


# ── Step 3: train LyricLMC on all songs that have (audio, lyric) ─────────────────
def train(reps: dict, cfg: dict | None = None, run_name: str | None = None):
    ids = [t for t in reps["ids"] if t in reps["mert"] and t in reps["lyric"]]
    if len(ids) < 30:
        raise RuntimeError(f"Only {len(ids)} songs have both MERT and lyric vectors — "
                           "run lyric_encoder.embed_pending() and MERT extraction first.")
    A = _mat(reps["mert"], ids)
    T = _mat(reps["lyric"], ids)
    model, hist = lyriclmc.train_lyriclmc(A, T, ids, cfg=cfg, run_name=run_name, save=True)
    return model, hist


# ── Step 4: geometry (global + per-song) on MERT vs lyric ────────────────────────
def geometry_scores(reps: dict) -> tuple[dict, pd.DataFrame]:
    ids = [t for t in reps["ids"] if t in reps["mert"] and t in reps["lyric"]]
    A, T = _mat(reps["mert"], ids), _mat(reps["lyric"], ids)
    rep = geometry.geometry_report(A, T, metric=config.GEOMETRY["rdm_metric"],
                                   k=config.GEOMETRY["local_k"])
    per_song = pd.DataFrame({"track_id": ids, "geom_local_rsa": rep["local_rsa"]})
    glob = {k: v for k, v in rep.items() if k not in ("local_rsa",)}
    return glob, per_song


# ── per-song export (merge into master_results.csv by track_id) ───────────────────
def export_congruence_scores(reps: dict, lmc_model) -> pd.DataFrame:
    """Build results/lmcmetrics/congruence_scores.csv with per-song columns."""
    config.ensure_dirs()
    ids = [t for t in reps["ids"] if t in reps["mert"] and t in reps["lyric"]]
    A, T = _mat(reps["mert"], ids), _mat(reps["lyric"], ids)

    df = pd.DataFrame({"track_id": ids})
    # LyricLMC per-song congruence (cosine of own audio vs own lyrics, learned space).
    df["lyriclmc_song"] = lmc_model.score_pairs(A, T)
    # Calibrated margin from the harness (z of true score vs its impostor row).
    S = lmc_model.score_matrix_from_reps(A, T)
    df["lyriclmc_margin"] = evaluate.retrieval_metrics(S)["per_song_margin"]
    # Geometry per-song congruence.
    _, geo = geometry_scores(reps)
    df = df.merge(geo, on="track_id", how="left")
    # Centered MuLan/CLAP song score, where available (Step 1 as a per-song column).
    for model in ("mulan", "clap"):
        mids = [t for t in ids if t in reps[f"{model}_a"] and t in reps[f"{model}_t"]]
        if len(mids) < 5:
            continue
        Am, Tm = _mat(reps[f"{model}_a"], mids), _mat(reps[f"{model}_t"], mids)
        c = Centerer(mode="mean").fit(Am, Tm)
        Ac, Tc = c.transform_audio(Am), c.transform_text(Tm)
        Ac = Ac / np.clip(np.linalg.norm(Ac, axis=1, keepdims=True), 1e-12, None)
        Tc = Tc / np.clip(np.linalg.norm(Tc, axis=1, keepdims=True), 1e-12, None)
        col = pd.DataFrame({"track_id": mids,
                            f"{model}_song_centered": np.sum(Ac * Tc, axis=1)})
        df = df.merge(col, on="track_id", how="left")

    out = config.METRICS_RESULTS_DIR / "congruence_scores.csv"
    df.sort_values("track_id").to_csv(out, index=False)
    logger.info("Wrote %s (%d songs, %d columns)", out, len(df), df.shape[1])
    return df


# ── the whole thing ───────────────────────────────────────────────────────────────
def run_all(limit: int | None = None, cfg: dict | None = None,
            embed_lyrics: bool = True) -> dict:
    """Cache lyrics -> train LyricLMC -> compare all metrics (held-out) -> export."""
    from lmc.utils import setup_logging
    setup_logging()
    config.ensure_dirs()
    print(config.summary())

    if embed_lyrics:
        lyric_encoder.embed_pending(limit=limit)

    reps = _load_reps(limit=limit)
    n_both = sum(1 for t in reps["ids"] if t in reps["mert"] and t in reps["lyric"])
    logger.info("Corpus: %d songs; %d have both MERT + lyric vectors.", len(reps["ids"]), n_both)

    model, hist = train(reps, cfg=cfg)

    # Honest, leakage-free comparison: evaluate ALL metrics on LyricLMC's val split.
    val_ids = json.loads((model.run_dir / "split.json").read_text())["val_track_ids"]
    val_ids = [t for t in val_ids if all(t in reps[k] for k in ("mert", "lyric"))]
    logger.info("Held-out comparison on %d validation songs.", len(val_ids))

    tbl_global = evaluate_on_ids(reps, val_ids, lmc_model=model, restrict=None)
    tbl_hard   = evaluate_on_ids(reps, val_ids, lmc_model=model, restrict="same")

    config.METRICS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tbl_global.to_csv(config.METRICS_RESULTS_DIR / "retrieval_heldout.csv")
    tbl_hard.to_csv(config.METRICS_RESULTS_DIR / "retrieval_heldout_within_genre.csv")

    glob, _ = geometry_scores(reps)
    (config.METRICS_RESULTS_DIR / "geometry_global.json").write_text(json.dumps(
        {k: (v if np.isscalar(v) else float(v)) for k, v in glob.items()}, indent=2))

    scores = export_congruence_scores(reps, model)

    print("\n=== Held-out matched-vs-mismatched (all impostors) ===")
    print(tbl_global.round(4).to_string())
    print("\n=== Held-out, WITHIN-GENRE impostors (harder) ===")
    print(tbl_hard.round(4).to_string())
    print("\n=== Global geometry (MERT vs lyric) ===")
    print(f"  linear CKA   : {glob['linear_cka']:.4f}")
    print(f"  RSA spearman : {glob['rsa_spearman']:.4f}")
    print(f"  local-RSA avg: {glob['local_rsa_mean']:.4f}")
    print(f"\nWrote per-song columns -> {config.METRICS_RESULTS_DIR / 'congruence_scores.csv'}")

    return {"run_dir": str(model.run_dir), "best_val_auc": hist["best_val_auc"],
            "retrieval_heldout": tbl_global, "retrieval_within_genre": tbl_hard,
            "geometry_global": glob, "n_scored": len(scores)}


if __name__ == "__main__":
    run_all()
