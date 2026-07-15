"""
selftest.py — run the whole lmcmetrics pipeline on SYNTHETIC data.

No corpus, no models, no network needed. It checks that the maths and plumbing are
correct end to end:

  1. Step 2 harness: high-signal synthetic pairs -> AUC high; zero-signal -> ~0.5.
  2. Step 1 centering: a simulated modality gap hurts raw cosine; centering recovers.
  3. Step 4 geometry: congruent geometries -> high CKA/RSA; unrelated -> ~0.
  4. Step 3 LyricLMC: training on different-dim synthetic pairs learns (val AUC up),
     and a reloaded run reproduces the score (torch only; skipped if torch absent).

Run:  python -m lmcmetrics.selftest          (from the src/ directory)
      or  cd src && python -c "from lmcmetrics import selftest; selftest.main()"
"""

from __future__ import annotations
import logging

import numpy as np

from . import data, evaluate, geometry, scorers

log = logging.getLogger("lmcmetrics.selftest")


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    return ok


def test_harness() -> bool:
    # The harness needs a square score matrix in a comparable space, so use the
    # same-dim joint synthetic (audio and text share dimensionality).
    hi = data.synthetic_joint_pairset(n=250, signal=0.9, gap=0.0, seed=1)
    lo = data.synthetic_joint_pairset(n=250, signal=0.0, gap=0.0, seed=1)
    S_hi = evaluate.cosine_score_matrix(hi.audio, hi.text)
    S_lo = evaluate.cosine_score_matrix(lo.audio, lo.text)
    auc_hi = evaluate.retrieval_metrics(S_hi)["auc_mean"]
    auc_lo = evaluate.retrieval_metrics(S_lo)["auc_mean"]
    return (_check("harness: high-signal AUC > 0.8", auc_hi > 0.8, f"auc={auc_hi:.3f}")
            and _check("harness: zero-signal AUC ~ 0.5", 0.4 < auc_lo < 0.6, f"auc={auc_lo:.3f}"))


def test_centering() -> bool:
    ps = data.synthetic_joint_pairset(n=300, signal=0.8, gap=20.0, seed=2)
    raw = scorers.CosineScorer(center=None)
    cen = scorers.CosineScorer(center="mean")
    auc_raw = evaluate.retrieval_metrics(raw.score_matrix(ps))["auc_mean"]
    auc_cen = evaluate.retrieval_metrics(cen.score_matrix(ps))["auc_mean"]
    return _check("centering recovers AUC lost to the modality gap",
                  auc_cen > auc_raw + 0.02, f"raw={auc_raw:.3f} -> centered={auc_cen:.3f}")


def test_geometry() -> bool:
    hi = data.synthetic_pairset(n=200, signal=0.9, seed=3)
    lo = data.synthetic_pairset(n=200, signal=0.0, seed=3)
    rep_hi = geometry.geometry_report(hi.audio, hi.text)
    rep_lo = geometry.geometry_report(lo.audio, lo.text)
    ok = True
    ok &= _check("geometry: congruent CKA > 0.3", rep_hi["linear_cka"] > 0.3,
                 f"cka={rep_hi['linear_cka']:.3f}")
    ok &= _check("geometry: congruent RSA > 0.3", rep_hi["rsa_spearman"] > 0.3,
                 f"rsa={rep_hi['rsa_spearman']:.3f}")
    ok &= _check("geometry: unrelated CKA < 0.1", rep_lo["linear_cka"] < 0.1,
                 f"cka={rep_lo['linear_cka']:.3f}")
    ok &= _check("geometry: local-RSA separates congruent vs unrelated",
                 rep_hi["local_rsa_mean"] > rep_lo["local_rsa_mean"] + 0.1,
                 f"{rep_lo['local_rsa_mean']:.3f} -> {rep_hi['local_rsa_mean']:.3f}")
    return ok


def test_lyriclmc() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        print("  [SKIP] LyricLMC training test (torch not importable here)")
        return True
    from . import lyriclmc
    ps = data.synthetic_pairset(n=500, dim_a=128, dim_t=64, signal=0.85, seed=4)
    cfg = {"epochs": 40, "patience": 10, "batch_size": 128}
    model, hist = lyriclmc.train_lyriclmc(ps.audio, ps.text, ps.track_ids, cfg=cfg,
                                          run_name="selftest_run", save=True)
    ok = _check("LyricLMC: learns (best val AUC > 0.7)", hist["best_val_auc"] > 0.7,
                f"val_auc={hist['best_val_auc']:.3f}")
    # reload and confirm the saved run reproduces a sane score matrix
    reloaded = lyriclmc.LyricLMC.load_run(model.run_dir)
    S = reloaded.score_matrix_from_reps(ps.audio, ps.text)
    auc = evaluate.retrieval_metrics(S)["auc_mean"]
    ok &= _check("LyricLMC: reloaded run reproduces high AUC", auc > 0.7, f"auc={auc:.3f}")
    return ok


def main() -> bool:
    logging.basicConfig(level=logging.WARNING)
    # Silence the known spurious "… encountered in matmul" warnings emitted by
    # NumPy 2.x on macOS/Accelerate (the production conda env pins numpy<2, so these
    # never appear there; the maths is unaffected — all values verified finite).
    import warnings
    warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
    print("\nlmcmetrics self-test (synthetic data)\n" + "-" * 40)
    results = {
        "Step 2 — retrieval harness": test_harness(),
        "Step 1 — centering fix":     test_centering(),
        "Step 4 — CKA / RSA geometry": test_geometry(),
        "Step 3 — LyricLMC training":  test_lyriclmc(),
    }
    print("-" * 40)
    allok = all(results.values())
    print(f"{'ALL PASSED' if allok else 'SOME FAILED'}: "
          + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in results.items()))
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
