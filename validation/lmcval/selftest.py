"""
selftest.py — validate the compute + CSV plumbing with MOCK audio and MOCK models.

No Spotify, no LRCLIB, no downloads, no model weights. It fabricates a few tracks
with synthetic audio and a tiny synced-lyrics string, then runs the full unit →
score → CSV path and checks the column contracts:

  song_wide     : artist + 12 score cols            (= 4 models x 3 prompts + 1)
  segment_wide  : artist + where-columns + 12 scores
  line_by_line  : artist + where-columns + 12 scores
  a model that fails to load -> its columns are all NaN.

Run:  cd validation && python -m lmcval.selftest
"""

from __future__ import annotations
import logging
import tempfile
from pathlib import Path

import numpy as np

from . import config, run, models as models_mod, units as units_mod
from .acquire import Track
from lmc.utils import parse_lrc
from lmc import chorus as chorus_mod

# In-memory synthetic audio, so the self-test needs no soundfile/librosa/ffmpeg.
_MEM: dict[str, np.ndarray] = {}
units_mod._load_audio = lambda path: _MEM.get(path)

LRC = ("[00:01.00] walking in the cold light of morning\n"
       "[00:03.00] mad world hold me now\n"
       "[00:05.00] faces come and go without warning\n"
       "[00:07.00] mad world hold me now\n"
       "[00:09.00] children waiting for the day they feel good\n"
       "[00:11.00] mad world hold me now\n")


def _fake_track(slug, artist, sr, seconds=13):
    rng = np.random.default_rng(abs(hash(slug)) % 2**31)
    wav = (0.1 * rng.standard_normal(int(seconds * sr))).astype(np.float32)
    path = f"mem://{slug}"
    _MEM[path] = wav
    tk = Track(slug=slug, artist=artist, title="Mad World", album="", duration_s=seconds,
               audio_path=path, synced_lyrics=LRC)
    tk.lines = parse_lrc(LRC)
    tk.chorus_flags = chorus_mod.detect_chorus(tk.lines)
    return tk


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    return ok


def main() -> bool:
    logging.basicConfig(level=logging.WARNING)
    print("\nlmcval self-test (mock audio + mock models)\n" + "-" * 44)
    tracks = [_fake_track("a_tears-for-fears", "Tears For Fears", config.BASE_SR),
              _fake_track("b_gary-jules", "Gary Jules", config.BASE_SR),
              _fake_track("c_demi-lovato", "Demi Lovato", config.BASE_SR)]

    # Load 3 of 4 models as mocks; leave 'clamp3' OUT to test the NaN-column path.
    models = {m: models_mod.MockModel(name=m, seed=i)
              for i, m in enumerate(["mulan", "laion_clap", "ms_clap"])}

    out = run.run_all(tracks, models, write=True)
    sc = run.score_columns()
    ok = True

    song = out["song"]
    ok &= _check("song_wide has artist + 12 score cols (=4x3+1)",
                 list(song.columns) == ["artist"] + sc and song.shape[1] == 13,
                 f"cols={song.shape[1]}, rows={len(song)}")
    ok &= _check("song_wide has one row per cover", len(song) == 3, f"rows={len(song)}")

    seg = out["segment"]
    ok &= _check("segment_wide has where-cols + 12 scores",
                 all(c in seg.columns for c in ["artist", "segment_label", "position_pct"])
                 and all(c in seg.columns for c in sc),
                 f"cols={list(seg.columns)[:4]}…, rows={len(seg)}")
    ok &= _check("segment_wide split chorus vs verse", set(seg["segment_label"]) <= {"chorus", "verse"}
                 and len(seg) >= 2, f"labels={sorted(set(seg['segment_label']))}")

    line = out["line"]
    ok &= _check("line_by_line has line_text + position + 12 scores",
                 all(c in line.columns for c in ["artist", "line_index", "position_pct", "line_text"])
                 and all(c in line.columns for c in sc), f"rows={len(line)}")

    ok &= _check("missing model (clamp3) -> all-NaN columns",
                 song[[f"clamp3__{p}" for p in config.PROMPT_KEYS]].isna().all().all(), "")
    ok &= _check("loaded models produce finite scores",
                 np.isfinite(song[[f"mulan__{p}" for p in config.PROMPT_KEYS]].values).all(), "")

    # cleanup CSVs (synthetic audio was in-memory only)
    for f in ("song_wide.csv", "segment_wide.csv", "line_by_line.csv"):
        (config.RESULTS_DIR / f).unlink(missing_ok=True)

    print("-" * 44)
    print("ALL PASSED" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
