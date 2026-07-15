"""
validate.py — Phase 2: validate every generated song and write the tidy results.

Two things are checked on each of the 256 clips:

  Lyric presence   Whisper transcribes the clip; word error rate (WER) vs the couplet
                   scores how faithfully the intended lyrics were sung (asr.py). The
                   echoed Lyria lyrics (<clip>.lyria.txt, real mode) are a free extra.
  VA placement     (a) MuLan audio-vs-anchor cosine to each VA corner — the embedding
                   target the songs were aimed at; argmax = predicted music corner.
                   (b) librosa acoustic VA (va.py) → nearest corner + distance to the
                   numeric target — a genre-robust, model-independent second opinion.

Also records lmc_cross = cos(audio, own-lyric text) in MuLan — the realised
Lyric-Music Congruence, defined the same way as the observational arm.

Writes results/generation/songs.csv (one row per song). MuLan is the primary embedder;
CLAP columns (clap_*) are added when config.VALIDATE['use_clap'] is on. No local
generation model is loaded here (Lyria is a remote API), so MuLan + small Whisper
co-reside comfortably.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

from . import config as C
from . import quadrants as q
from . import lyrics as lyr
from . import mulan
from . import va
from . import generate as G

logger = logging.getLogger(__name__)

SONGS_CSV = C.RESULTS_DIR / "songs.csv"


# ─── embedding caches ────────────────────────────────────────────────────────────
def embed_lyrics(scorer: "mulan.Scorer", force: bool = False) -> dict[str, np.ndarray]:
    """Embed each couplet's TEXT once (cached). Independent of the music generation."""
    from lmc.utils import load_song_embeddings, save_song_embeddings
    path = C.EMB_DIR / f"lyrics_{scorer.model_key}.npz"
    if path.exists() and not force:
        cached = load_song_embeddings(path)
        if cached and set(cached) >= set(lyr.ORDER):
            return {k: cached[k] for k in lyr.ORDER}
    C.EMB_DIR.mkdir(parents=True, exist_ok=True)
    vecs = {c.lyric_id: mulan._unit(scorer.embed_text(c.plain)) for c in lyr.ALL}
    save_song_embeddings(path, vecs)
    return vecs


def embed_audio(scorer: "mulan.Scorer", force: bool = False) -> dict[str, np.ndarray]:
    """Embed every generated clip's AUDIO once (cached to one bundle per model).
    Keyed by clip stem '<lyric_id>__<music_q>__rep<k>'."""
    from lmc.utils import load_song_embeddings, save_song_embeddings
    path = C.EMB_DIR / f"audio_{scorer.model_key}.npz"
    cache = {} if force else (load_song_embeddings(path) or {})
    out: dict[str, np.ndarray] = {}
    changed = False
    for spec in G.build_specs():
        key = Path(spec.out_path).stem
        if key in cache:
            out[key] = cache[key]
            continue
        vec = scorer.embed_audio_file(spec.out_path)
        if vec is None:
            raise FileNotFoundError(f"missing / unreadable clip for {key}; generate first")
        out[key] = mulan._unit(vec)
        changed = True
    if changed or not path.exists():
        C.EMB_DIR.mkdir(parents=True, exist_ok=True)
        save_song_embeddings(path, out)
    return out


# ─── model-independent measures (WER + acoustic VA) ──────────────────────────────
def _measure_lyric_and_va(use_asr: bool) -> dict[str, dict]:
    """Per-clip WER / vocal-presence / acoustic VA, keyed by clip stem. One pass."""
    from . import va
    transcriber = None
    if use_asr and C.ASR["enabled"]:
        from . import asr
        transcriber = asr.Transcriber()

    meta: dict[str, dict] = {}
    specs = G.build_specs()
    for i, spec in enumerate(specs, 1):
        key = Path(spec.out_path).stem
        rec: dict = {}
        if transcriber is not None:
            from . import asr
            info = transcriber.transcribe_detailed(spec.out_path)
            refs = G.wer_references(spec.lyric_id)
            rec["wer"] = min(asr.word_error_rate(r, info["text"]) for r in refs)
            rec["vocal_present"] = info["vocal_present"]
            rec["transcript"] = info["text"]
        else:
            rec["wer"] = rec["vocal_present"] = rec["transcript"] = None
        av = va.audio_va(spec.out_path) or (float("nan"), float("nan"))
        rec["audio_v"], rec["audio_a"] = av
        rec["va_pred_quadrant"] = (q.nearest_quadrant(*av)
                                   if not np.isnan(av[0]) else None)
        rec["va_dist_to_target"] = va.va_distance(av, (spec.valence, spec.arousal))
        meta[key] = rec
        if i % 32 == 0:
            logger.info("  measured %d/%d clips (WER/VA)", i, len(specs))
    return meta


# ─── driver ──────────────────────────────────────────────────────────────────────
def validate_all(models: list[str] | None = None, use_asr: bool = True):
    """Validate all 256 songs and write songs.csv. Returns the tidy DataFrame."""
    import pandas as pd
    C.ensure_dirs()
    if models is None:
        models = ["mulan"] + (["clap"] if C.VALIDATE["use_clap"] else [])

    # one Whisper/librosa pass (model-independent)
    logger.info("Measuring lyric WER + acoustic VA on all clips…")
    meta = _measure_lyric_and_va(use_asr=use_asr)

    # base rows from the design
    specs = G.build_specs()
    lyric_va = {c.lyric_id: va.lyric_va(c.plain) for c in lyr.ALL}
    rows = []
    for spec in specs:
        key = Path(spec.out_path).stem
        lid, mq = spec.lyric_id, spec.music_quadrant
        lq = lyr.get(lid).quadrant
        lv, la, _ = lyric_va[lid]
        rec = {
            "lyric_id": lid, "lyric_quadrant": lq, "music_quadrant": mq,
            "rep": spec.rep, "congruent": (lq == mq),
            "path": str(spec.out_path),
            "design_music_v": spec.valence, "design_music_a": spec.arousal,
            "lyric_v": round(lv, 3), "lyric_a": round(la, 3),
        }
        rec.update(meta[key])
        rows.append(rec)
    df = pd.DataFrame(rows).set_index(["lyric_id", "music_quadrant", "rep"])

    # per-model embedding columns
    for mk in models:
        scorer = mulan.Scorer(mk)
        anchors = scorer.build_anchors()
        lyr_vecs = embed_lyrics(scorer)
        aud_vecs = embed_audio(scorer)
        tgt, pred, lmc = [], [], []
        anchor_cols = {code: [] for code in q.ORDER}
        for lid, mq, rep in df.index:
            a = aud_vecs[f"{lid}__{mq}__rep{rep}"]
            scores = scorer.score_against_anchors(a, anchors)
            tgt.append(scores[mq])
            pred.append(mulan.argmax(scores))
            lmc.append(mulan.cosine(a, lyr_vecs[lid]))
            for code in q.ORDER:
                anchor_cols[code].append(scores[code])
        df[f"{mk}_target_anchor_cos"] = tgt
        df[f"{mk}_pred_quadrant"] = pred
        df[f"{mk}_lmc_cross"] = lmc
        for code in q.ORDER:
            df[f"{mk}_anchor__{code}"] = anchor_cols[code]
        del scorer, anchors, lyr_vecs, aud_vecs
        _gc()

    df = df.reset_index()
    df.to_csv(SONGS_CSV, index=False)
    logger.info("Phase 2 complete: %d rows → %s", len(df), SONGS_CSV)
    return df


def _gc() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                              # noqa: BLE001
        pass
