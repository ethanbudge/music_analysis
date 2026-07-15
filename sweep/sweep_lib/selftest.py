"""
selftest.py — validate the sweep compute + master-build plumbing with MOCK models,
a TEMP database, and synthetic in-memory audio. No corpus, no real models, no writes
to your real project.db.

Run:  cd sweep && python -m sweep_lib.selftest
"""

from __future__ import annotations
import os
import tempfile

# Redirect the pipeline's data/results roots to a temp dir BEFORE importing lmc.config
# (so nothing here can touch your real project.db / results).
_TMP = tempfile.mkdtemp(prefix="sweep_selftest_")
os.environ["LMC_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["LMC_RESULTS_DIR"] = os.path.join(_TMP, "results")

import logging
import numpy as np
import pandas as pd

from . import config, compute, build_master
from lmcval.models import MockModel
from lmc.utils import parse_lrc
from lmc import chorus as chorus_mod

_LRC = ("[00:01.00] all around me are familiar faces\n"
        "[00:03.00] mad world hold the line\n"
        "[00:05.00] worn out places worn out faces\n"
        "[00:07.00] mad world hold the line\n"
        "[00:09.00] bright and early for the daily races\n"
        "[00:11.00] mad world hold the line\n")


def _fake_job(tid: int) -> dict:
    rng = np.random.default_rng(tid)
    wav = (0.1 * rng.standard_normal(int(13 * config.BASE_SR))).astype(np.float32)
    lines = parse_lrc(_LRC)
    flags = chorus_mod.detect_chorus(lines)
    job = compute._build_units(wav, lines, flags, len(wav) / config.BASE_SR)
    job["tid"] = tid
    return job


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    return ok


def main() -> bool:
    logging.basicConfig(level=logging.WARNING)
    print("\nsweep_lib self-test (mock models, temp DB, synthetic audio)\n" + "-" * 58)
    ok = True

    # 1. naming + grid size
    ok &= _check("grid is 72 LMC columns (4×3×6)", len(config.all_lmc_columns()) == 72,
                 f"n={len(config.all_lmc_columns())}")
    ok &= _check("column naming", config.lmc_col("mulan", "raw", "song") == "mulan_raw_song"
                 and config.embedding("clamp3", "idea") == "clamp3_idea")
    ok &= _check("prompt wording: song vs song segment",
                 config.format_prompt("contains", "X", "song") == "a song that contains the lyrics X"
                 and config.format_prompt("contains", "X", "line") == "a song segment that contains the lyrics X")

    # 2. unit construction
    job = _fake_job(1)
    ok &= _check("units include song + chorus + nonchorus + line windows",
                 "song" in job["audio"] and "chorus" in job["audio"] and "nonchorus" in job["audio"]
                 and ("line", "line_buf5", 0) in job["audio"], f"n_audio={len(job['audio'])}")

    # 3. scoring one chunk with a mock model
    jobs = [_fake_job(1), _fake_job(2)]
    rows, prog = compute._score_chunk(MockModel(name="mulan", seed=0), "mulan", jobs)
    methods = {r[3] for r in rows}
    ok &= _check("rows cover all 6 methods", set(config.ALL_METHODS) <= methods,
                 f"methods={sorted(methods)}")
    ok &= _check("rows cover all 3 prompts", {r[2] for r in rows} == set(config.PROMPTS))
    ok &= _check("cosine values finite in [-1,1]",
                 all(np.isfinite(r[4]) and -1.0001 <= r[4] <= 1.0001 for r in rows))
    ok &= _check("progress marks both songs", len(prog) == 2)

    # 3b. cache reuse: score a song from a synthetic cached bundle (no audio recompute)
    L = len(parse_lrc(_LRC)); D = 16; rng = np.random.default_rng(7)
    bundle = {k: rng.standard_normal(D).astype(np.float32) for k in
              ("audio_full", "chorus_audio", "nonchorus_audio",
               "text_full", "chorus_text", "nonchorus_text")}
    bundle["line_text"] = rng.standard_normal((L, D)).astype(np.float32)
    for w in ("buf1", "buf5", "buf10"):
        bundle["audio_" + w] = rng.standard_normal((L, D)).astype(np.float32)
    res = compute._score_song_from_cache("mulan", 1, bundle,
                                         {"track_id": 1, "synced_lyrics": _LRC},
                                         MockModel(name="mulan", dim=D))
    ok &= _check("cache path returns rows", res is not None and len(res[0]) > 0)
    if res:
        crows = res[0]
        ok &= _check("cache rows cover all 6 methods", set(config.ALL_METHODS) <= {r[3] for r in crows})
        ok &= _check("cache rows cover all 3 prompts", {r[2] for r in crows} == set(config.PROMPTS))
        ok &= _check("cache values finite", all(np.isfinite(r[4]) for r in crows))
    ok &= _check("_load_bundle None for non-cache model", compute._load_bundle("msclap", 1) is None)

    # 4. end-to-end compute_model on a temp DB (mock everything DB/audio/model)
    compute.ensure_tables()
    fake_songs = [{"track_id": t} for t in (1, 2, 3)]
    compute._pending = lambda mk, limit=None: [] if mk != "mulan" else fake_songs   # noqa
    compute._load_song_job = lambda s: _fake_job(s["track_id"])                      # noqa
    compute.load_models = lambda names, device=None: {"mulan": MockModel(name="mulan")}  # noqa
    res = compute.compute_model("mulan", chunk_size=2)
    from lmc import db as projdb
    with projdb.connect() as conn:
        n_lmc = projdb.count(conn, "lmc_sweep")
        n_prog = projdb.count(conn, "sweep_progress")
    ok &= _check("compute_model wrote lmc_sweep + progress", n_lmc > 0 and n_prog == 3,
                 f"lmc_rows={n_lmc}, progress={n_prog}, processed={res['processed']}")
    # resumability: second run finds nothing pending
    compute._pending = lambda mk, limit=None: []                                    # noqa
    res2 = compute.compute_model("mulan", chunk_size=2)
    ok &= _check("resumable: re-run processes 0", res2["processed"] == 0)

    # 5. build_master merges sweep cols onto a fake existing master
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"track_id": [1, 2, 3], "spotify_popularity": [40, 55, 30],
                  "genre": ["rock", "pop", "folk"], "orientation": ["narrative"] * 3,
                  "song_age_years": [5, 8, 12], "mert_pc01": [0.1, -0.2, 0.3],
                  "mert_pc02": [0.0, 0.1, -0.1]}).to_csv(
        config.RESULTS_DIR / "master_results.csv", index=False)
    summ = build_master.build()
    m = pd.read_csv(config.MASTER_SWEEP_CSV)
    ok &= _check("master_results_sweep.csv has merged sweep + control columns",
                 "mulan_raw_song" in m.columns and "mert_pc01" in m.columns
                 and "spotify_popularity" in m.columns, f"cols={m.shape[1]}, rows={len(m)}")
    ok &= _check("master row per corpus song", len(m) == 3)

    print("-" * 58)
    print("ALL PASSED" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
