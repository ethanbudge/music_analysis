"""
pipeline.py — Orchestration: tune recipes -> generate the 8x8 grid -> validate.

Stages (each resumable / cached so you can stop and restart on the Mac):

  1. build_anchors      MuLan (+ optional CLAP) emotion anchors            [mulan.py]
  2. tune_recipes       pick each music emotion's best caption by MuLan match
  3. generate_grid      generate 64 = 8 lyrics x 8 music choruses          [acestep.py]
  4. validate           lyric-side + audio-side embedding evidence -> tidy results

The tidy `grid_results.csv` (one row per cell x embedding model) is what the
analysis module turns into descriptive statistics and figures.

Circularity note: `tune_recipes` picks the caption that MAXIMISES cosine(audio,
MuLan anchor). A MuLan audio-vs-anchor "manipulation check" is therefore partly
self-fulfilling. The two checks that are NOT selected on are (a) the cross-modal
LMC contrast cosine(audio, lyric_text), which tuning never optimised, and (b) the
optional CLAP validator, which tuning never touched. Read those as the honest tests.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np

from . import config as C
from . import emotions as emo
from . import lyrics as lyr
from . import mulan
from .acestep import AceStepGenerator, GenSpec

logger = logging.getLogger(__name__)

RECIPES_JSON = C.RESULTS_DIR / "recipes.json"
GRID_CSV     = C.RESULTS_DIR / "grid_results.csv"


def clean_generated(audio: bool = True, embeddings: bool = True,
                    anchors: bool = True, results: bool = True) -> None:
    """Delete cached generation artifacts.

    IMPORTANT: the pipeline is resumable by file existence, so mock (dry-run) clips,
    embeddings and recipes would otherwise be REUSED when you switch to real ACE-Step
    generation. Call this once when flipping LMCGEN_DRY_RUN from 1 to 0 (or vice versa).
    """
    import shutil
    targets = []
    if audio:      targets += [C.AUDIO_DIR, C.TUNE_DIR]
    if embeddings: targets += [C.EMB_DIR]
    if anchors:    targets += [C.ANCHOR_DIR]
    if results:    targets += [C.RESULTS_DIR]
    for d in targets:
        if Path(d).exists():
            shutil.rmtree(d)
            logger.info("removed %s", d)
    C.ensure_dirs()


# ─── cell identity ───────────────────────────────────────────────────────────────
def cell_seed(lyric_emotion: str, music_emotion: str) -> int:
    """Deterministic per-cell seed so generation is reproducible + resumable."""
    li = emo.ORDER.index(lyric_emotion)
    mi = emo.ORDER.index(music_emotion)
    return C.GRID_SEED + li * 100 + mi


def cell_path(lyric_emotion: str, music_emotion: str) -> Path:
    seed = cell_seed(lyric_emotion, music_emotion)
    return C.AUDIO_DIR / f"{lyric_emotion}__{music_emotion}__{seed}.wav"


# ─── stage 2: recipe tuning ──────────────────────────────────────────────────────
def tune_recipes(scorer: "mulan.Scorer", anchors: dict, generator: AceStepGenerator,
                 force: bool = False) -> dict[str, str]:
    """Pick, per music emotion, the caption whose generated audio is closest to the
    emotion's MuLan anchor. Returns {emotion: chosen_caption}. Cached to recipes.json.
    """
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RECIPES_JSON.exists() and not force:
        chosen = json.loads(RECIPES_JSON.read_text())
        if set(chosen) >= set(emo.ORDER):
            logger.info("Recipes: loaded cached choices.")
            return chosen

    chosen: dict[str, str] = {}
    for e in emo.ALL:
        if not C.TUNE["enabled"]:
            chosen[e.name] = e.caption
            continue
        candidates = e.captions[: C.TUNE["candidates_per_emotion"]]
        best_cap, best_score = candidates[0], -1.0
        for ci, cap in enumerate(candidates):
            scores = []
            for s in range(C.TUNE["seeds_per_candidate"]):
                out = C.TUNE_DIR / f"{e.name}__cand{ci}__seed{s}.wav"
                spec = GenSpec(caption=cap, lyrics="[Chorus]\nla la la la",
                               out_path=out, bpm=e.bpm, keyscale=e.keyscale,
                               duration=min(12.0, C.CHORUS_DURATION_S),
                               seed=C.GRID_SEED + ci * 10 + s, emotion=e.name)
                generator.generate(spec)
                vec = scorer.embed_audio_file(out)
                if vec is not None:
                    scores.append(scorer.score_against_anchors(vec, anchors)[e.name])
            mean_score = float(np.mean(scores)) if scores else -1.0
            logger.info("  tune[%s] cand%d: anchor-cos=%.3f", e.name, ci, mean_score)
            if mean_score > best_score:
                best_cap, best_score = cap, mean_score
        chosen[e.name] = best_cap
        logger.info("  tune[%s] -> best anchor-cos=%.3f", e.name, best_score)

    RECIPES_JSON.write_text(json.dumps(chosen, indent=2))
    return chosen


# ─── stage 3: grid generation ────────────────────────────────────────────────────
def build_grid_specs(recipes: dict[str, str]) -> list[GenSpec]:
    """The 64 GenSpecs: every (lyric emotion) x (music emotion) pairing."""
    specs = []
    for L in emo.ORDER:                       # lyric emotion (row)
        chorus = lyr.get(L)
        for M in emo.ORDER:                   # music emotion (column)
            m = emo.get(M)
            specs.append(GenSpec(
                caption=recipes[M], lyrics=chorus.acestep_lyrics(),
                out_path=cell_path(L, M), bpm=m.bpm, keyscale=m.keyscale,
                duration=C.CHORUS_DURATION_S, seed=cell_seed(L, M), emotion=M,
            ))
    return specs


WER_JSON = C.RESULTS_DIR / "wer.json"


def _load_wer() -> dict:
    import json
    return json.loads(WER_JSON.read_text()) if WER_JSON.exists() else {}


def _save_wer(wer_map: dict) -> None:
    import json
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WER_JSON.write_text(json.dumps(wer_map, indent=2))


def generate_grid(generator: AceStepGenerator, specs: list[GenSpec],
                  transcriber=None, force: bool = False) -> dict:
    """Generate every cell. If `transcriber` is given, screen each cell for lyric
    intelligibility: generate up to config.ASR['max_takes'] takes (different seeds),
    transcribe each, and keep the LOWEST-WER take. Returns the WER map."""
    import time
    screen = transcriber is not None
    logger.info("Generating %d hooks (dry_run=%s, WER screening=%s)…",
                len(specs), generator.dry_run, screen)
    wer_map = _load_wer()
    t0 = time.monotonic()
    for i, spec in enumerate(specs, 1):
        L = Path(spec.out_path).stem.split("__")[0]
        key = f"{L}__{spec.emotion}"
        out = Path(spec.out_path)
        logger.info("[%d/%d] lyric=%s x music=%s …", i, len(specs), L, spec.emotion)
        tc = time.monotonic()

        if out.exists() and not force and (not screen or key in wer_map):
            logger.info("     exists, skip")
        elif not screen:
            generator.generate(spec, force=force)
        else:
            wer_map[key] = _screen_cell(generator, spec, transcriber, L)
            _save_wer(wer_map)
            logger.info("     kept WER=%.2f (take %d/%d)", wer_map[key]["wer"],
                        wer_map[key]["chosen_take"] + 1, wer_map[key]["attempts"])

        elapsed = time.monotonic() - t0
        eta = elapsed / i * (len(specs) - i)
        logger.info("     %.0fs  (elapsed %.0fm, ~%.0fm remaining)",
                    time.monotonic() - tc, elapsed / 60, eta / 60)
    return wer_map


def _screen_cell(generator: AceStepGenerator, spec: GenSpec, transcriber, L: str) -> dict:
    """Generate up to max_takes takes (different seeds), keep the lowest-WER one.
    The chosen take is copied to spec.out_path; takes are removed unless keep_takes."""
    import shutil
    from dataclasses import replace
    refs = _wer_references(L)          # [single hook, doubled hook] — credit 1 or 2 sung reps
    out = Path(spec.out_path)
    takes, best = [], None
    for k in range(C.ASR["max_takes"]):
        take_path = C.AUDIO_DIR / f"{L}__{spec.emotion}__take{k}.wav"
        generator.generate(replace(spec, out_path=take_path, seed=spec.seed + k * 10_007), force=True)
        info = transcriber.transcribe_detailed(take_path)
        wer = _best_wer(refs, info["text"])
        logger.info("     take %d: WER=%.2f%s  «%s»", k, wer,
                    "" if info["vocal_present"] else " [no vocal]", info["text"][:60])
        takes.append(take_path)
        if best is None or wer < best["wer"]:
            best = {"wer": wer, "transcript": info["text"], "chosen_take": k,
                    "vocal_present": info["vocal_present"], "attempts": 0,
                    "seed": spec.seed + k * 10_007}
        if wer <= C.ASR["accept_wer"] and best["vocal_present"]:
            break
    best["attempts"] = len(takes)
    shutil.copyfile(takes[best["chosen_take"]], out)
    if not C.ASR["keep_takes"]:
        for tp in takes:
            Path(tp).unlink(missing_ok=True)
    return best


def _wer_references(lyric_emotion: str) -> list[str]:
    """WER reference variants for a hook: the single hook and the doubled hook, so a
    clip is credited whether the model sang the hook once or twice."""
    hook = " ".join(lyr.get(lyric_emotion).lines)
    return [hook, hook + " " + hook]


def _best_wer(refs: list[str], hyp: str) -> float:
    from . import asr
    return min(asr.word_error_rate(r, hyp) for r in refs)


# ─── stage 4: validation ─────────────────────────────────────────────────────────
def embed_lyrics(scorer: "mulan.Scorer", force: bool = False) -> dict[str, np.ndarray]:
    """Embed each chorus's TEXT once (cached). Independent of the music generation."""
    from lmc.utils import load_song_embeddings, save_song_embeddings
    path = C.EMB_DIR / f"lyrics_{scorer.model_key}.npz"
    if path.exists() and not force:
        cached = load_song_embeddings(path)
        if cached and set(cached) >= set(emo.ORDER):
            return {k: cached[k] for k in emo.ORDER}
    C.EMB_DIR.mkdir(parents=True, exist_ok=True)
    vecs = {ch.emotion: mulan._unit(scorer.embed_text(ch.plain)) for ch in lyr.ALL}
    save_song_embeddings(path, vecs)
    return vecs


def embed_grid_audio(scorer: "mulan.Scorer", force: bool = False) -> dict[str, np.ndarray]:
    """Embed every generated clip's AUDIO once (cached to one bundle per model)."""
    from lmc.utils import load_song_embeddings, save_song_embeddings
    path = C.EMB_DIR / f"grid_audio_{scorer.model_key}.npz"
    cache = {} if force else (load_song_embeddings(path) or {})
    out: dict[str, np.ndarray] = {}
    changed = False
    for L in emo.ORDER:
        for M in emo.ORDER:
            key = f"{L}__{M}"
            if key in cache:
                out[key] = cache[key]
                continue
            vec = scorer.embed_audio_file(cell_path(L, M))
            if vec is None:
                raise FileNotFoundError(f"missing / unreadable clip for {key}; generate first")
            out[key] = mulan._unit(vec)
            changed = True
    if changed or not path.exists():
        C.EMB_DIR.mkdir(parents=True, exist_ok=True)
        save_song_embeddings(path, out)
    return out


def lyric_alignment_embed(scorer: "mulan.Scorer", anchors: dict, lyr_vecs: dict):
    """Embedding evidence that each chorus's TEXT lands in its target emotion.
    Returns a DataFrame: rows=chorus emotion, cols=anchor cosines, +predicted/correct.
    """
    import pandas as pd
    rows = {L: scorer.score_against_anchors(lyr_vecs[L], anchors) for L in emo.ORDER}
    df = pd.DataFrame(rows).T[emo.ORDER]
    df.index.name = "chorus_emotion"
    df["predicted"] = df[emo.ORDER].idxmax(axis=1)
    df["correct"] = df["predicted"] == df.index
    return df


def anchor_similarity(anchors: dict):
    """Designed congruence: cosine between every pair of emotion anchors (8x8)."""
    import pandas as pd
    M = np.array([[mulan.cosine(anchors[a], anchors[b]) for b in emo.ORDER] for a in emo.ORDER])
    return pd.DataFrame(M, index=emo.ORDER, columns=emo.ORDER)


def build_grid_results(scorer: "mulan.Scorer", anchors: dict,
                       lyr_vecs: dict, aud_vecs: dict):
    """Tidy results: one row per (lyric, music) cell for this embedding model.

    Columns:
      lyric_emotion, music_emotion, congruent (diagonal), seed, path, model
      lmc_cross         cosine(audio, own lyric text)         -- the realised LMC
      music_anchor_cos  cosine(audio, target music anchor)    -- music manipulation
      pred_music_emotion argmax anchor for the audio          -- confusion matrix
      anchor_cos__<e>   cosine(audio, anchor e) for all e
    """
    import pandas as pd
    wer_map = _load_wer()               # lyric-intelligibility screen (empty in dry-run)
    recs = []
    for L in emo.ORDER:
        for M in emo.ORDER:
            key = f"{L}__{M}"
            a = aud_vecs[key]
            anchor_cos = scorer.score_against_anchors(a, anchors)
            rec = {
                "model": scorer.model_key,
                "lyric_emotion": L, "music_emotion": M,
                "congruent": (L == M),
                "seed": cell_seed(L, M), "path": str(cell_path(L, M)),
                "lmc_cross": mulan.cosine(a, lyr_vecs[L]),
                "music_anchor_cos": anchor_cos[M],
                "pred_music_emotion": mulan.argmax_emotion(anchor_cos),
                "wer": wer_map.get(key, {}).get("wer"),           # None if screening was off
                "transcript": wer_map.get(key, {}).get("transcript"),
            }
            rec.update({f"anchor_cos__{e}": anchor_cos[e] for e in emo.ORDER})
            recs.append(rec)
    return pd.DataFrame(recs)


# ─── top-level driver ────────────────────────────────────────────────────────────
def run(models: list[str] | None = None, dry_run: bool | None = None,
        force_generate: bool = False):
    """
    Full pipeline for the given embedding models ('mulan' always; 'clap' if enabled).
    Returns a dict of DataFrames and writes CSVs under results/generation/.
    Generation happens ONCE; each model then re-scores the same 64 clips.

    MEMORY WARNING: this convenience driver runs generation and MuLan validation in
    the SAME process. On a 16 GB Mac that can co-resident ACE-Step (its own server
    process) AND MuLan and OOM. Prefer the two-phase path — generate_all() then, with
    the ACE-Step server stopped, validate_all() — see those functions. run() is fine
    for dry-run and for machines with plenty of RAM.
    """
    if dry_run is None:
        dry_run = C.DRY_RUN
    if not dry_run:
        logger.warning("run() loads MuLan and drives ACE-Step in one process — on a "
                       "16 GB Mac use generate_all() + validate_all() instead (see docs).")
    recipes = generate_all(dry_run=dry_run, tune=C.TUNE["enabled"], force=force_generate)
    out = validate_all(models=models)
    out["recipes"] = recipes
    return out


# ─── memory-isolated two-phase driver (recommended for 16 GB Macs) ───────────────
def generate_all(dry_run: bool | None = None, tune: bool | None = None,
                 force: bool = False) -> dict[str, str]:
    """PHASE 1 — generation only. Talks to the ACE-Step API server over HTTP and
    downloads clips; **does not load MuLan** (unless tune=True), so this process
    stays small and the ACE-Step server is the only large model in RAM.

    Run this with the ACE-Step API server up. When it finishes you can STOP the
    server (freeing its memory) before calling validate_all().

    tune: MuLan-guided caption search. Defaults to config.TUNE['enabled'] (off).
    Leave it OFF on low-RAM machines — it forces MuLan and ACE-Step to run at the
    same time, which is exactly the co-residency that causes OOM. With tuning off,
    each emotion uses its (hand-written, already emotion-targeted) default caption.
    """
    import json
    C.ensure_dirs()
    dry_run = C.DRY_RUN if dry_run is None else dry_run
    tune = C.TUNE["enabled"] if tune is None else tune
    generator = AceStepGenerator(dry_run=dry_run)
    if not dry_run:
        from . import acestep
        acestep.check_server()

    if tune:
        logger.warning("Recipe tuning loads MuLan alongside ACE-Step — memory-heavy. "
                       "On a 16 GB Mac keep tune=False.")
        tuner = mulan.Scorer("mulan")
        anchors = tuner.build_anchors()
        recipes = tune_recipes(tuner, anchors, generator, force=force)
        del tuner, anchors               # free MuLan before the (larger) grid run
        _gc()
    else:
        recipes = {e.name: e.caption for e in emo.ALL}
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        RECIPES_JSON.write_text(json.dumps(recipes, indent=2))
        logger.info("Recipes: using default per-emotion captions (tuning off).")

    transcriber = None
    if not dry_run and C.ASR["enabled"]:
        from . import asr
        transcriber = asr.Transcriber()      # small Whisper (CPU/int8) — screens lyric WER

    specs = build_grid_specs(recipes)
    generate_grid(generator, specs, transcriber=transcriber, force=force)
    if transcriber is not None:
        wer = _load_wer()
        vals = [v["wer"] for v in wer.values()]
        if vals:
            logger.info("Lyric WER across %d cells: median=%.2f, worst=%.2f",
                        len(vals), sorted(vals)[len(vals) // 2], max(vals))
    logger.info("Phase 1 complete: %d clips under %s. You can now STOP the ACE-Step "
                "server and run validate_all().", len(specs), C.AUDIO_DIR)
    return recipes


def validate_all(models: list[str] | None = None) -> dict:
    """PHASE 2 — validation only. Loads MuLan (+ optional CLAP), embeds the already-
    generated clips + lyrics + anchors, and builds the tidy results. The ACE-Step
    server is NOT needed here and should be stopped first to free its memory.

    Assumes generate_all() has produced all 64 clips (it errors clearly if not).
    Models are processed one at a time and freed between, to cap peak memory.
    """
    import pandas as pd
    C.ensure_dirs()
    if models is None:
        models = ["mulan"] + (["clap"] if C.VALIDATE["use_clap"] else [])

    grid_frames, lyric_frames = [], {}
    for mk in models:
        scorer = mulan.Scorer(mk)
        anchors = scorer.build_anchors()
        lyr_vecs = embed_lyrics(scorer)
        aud_vecs = embed_grid_audio(scorer)

        lyric_frames[mk] = lyric_alignment_embed(scorer, anchors, lyr_vecs)
        lyric_frames[mk].to_csv(C.RESULTS_DIR / f"lyric_alignment_{mk}.csv")
        anchor_similarity(anchors).to_csv(C.RESULTS_DIR / f"anchor_similarity_{mk}.csv")
        grid_frames.append(build_grid_results(scorer, anchors, lyr_vecs, aud_vecs))
        del scorer, anchors, lyr_vecs, aud_vecs      # free before the next model
        _gc()

    grid = pd.concat(grid_frames, ignore_index=True)
    grid.to_csv(GRID_CSV, index=False)
    lyr.lexical_alignment().to_csv(C.RESULTS_DIR / "lyric_alignment_lexical.csv")

    logger.info("Phase 2 complete: %d rows in %s", len(grid), GRID_CSV)
    return {"grid": grid, "lyric_embed": lyric_frames, "lexical": lyr.lexical_alignment()}


def _gc() -> None:
    """Release model memory promptly (Python GC + torch allocator caches)."""
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


# ─── Tier-1 re-scoring on EXISTING clips (no regeneration) ───────────────────────
def rescore_and_va(models: list[str] | None = None):
    """Re-measure the already-generated 64 clips with the Tier-1 fixes — no ACE-Step
    needed:
      • fair WER  — transcribe with hallucination guards, score vs the SINGLE hook
                    (crediting one or two sung reps), flag vocal-absent clips.
      • valence/arousal — audio VA (librosa) + lyric VA (VAD lexicon) as the emotion
                    instrument that survives a fixed genre (unlike MuLan anchors).

    Writes results/generation/va_results.csv and returns the DataFrame. Loads Whisper
    (small) + librosa only — MuLan is NOT needed, so this is light on memory.
    """
    import pandas as pd
    from . import asr, va
    C.ensure_dirs()
    transcriber = asr.Transcriber()
    recs = []
    n = len(emo.ORDER) ** 2
    for idx, L in enumerate(emo.ORDER):
        lv, la, lmatch = va.lyric_va(" ".join(lyr.get(L).lines))
        refs = _wer_references(L)
        for M in emo.ORDER:
            path = cell_path(L, M)
            info = transcriber.transcribe_detailed(path)
            wer = _best_wer(refs, info["text"])
            av = va.audio_va(path) or (float("nan"), float("nan"))
            dm = emo.get(M); dl = emo.get(L)
            rec = {
                "lyric_emotion": L, "music_emotion": M, "congruent": (L == M),
                "wer": wer, "vocal_present": info["vocal_present"],
                "transcript": info["text"],
                "audio_v": av[0], "audio_a": av[1],
                "lyric_v": lv, "lyric_a": la, "lyric_matches": lmatch,
                "design_music_v": dm.valence, "design_music_a": dm.arousal,
                "design_lyric_v": dl.valence, "design_lyric_a": dl.arousal,
                "va_congruence": va.va_congruence(av, (lv, la)),
                "va_congruence_design": va.va_congruence((dl.valence, dl.arousal),
                                                          (dm.valence, dm.arousal)),
            }
            recs.append(rec)
        logger.info("rescored lyric=%s (%d/%d)", L, (idx + 1) * len(emo.ORDER), n)
    df = pd.DataFrame(recs)
    out = C.RESULTS_DIR / "va_results.csv"
    df.to_csv(out, index=False)
    logger.info("Tier-1 rescore complete → %s", out)
    return df


# ─── A/B: does re-enabling the LM planner restore tempo control? ──────────────────
def ab_tempo_test(emotions=("grief", "rage"), thinking_values=(False, True),
                  lyric_emotion: str | None = None):
    """Gating experiment for the v2-run finding that ACE-Step ignored our per-emotion
    tempo. For a couple of contrasting emotions, generate the SAME caption/bpm with the
    LM planner OFF vs ON and measure the resulting tempo. If measured BPM tracks the
    requested BPM only when thinking=True, re-enable the planner before any full regen.

    Needs the ACE-Step server up; loads librosa only (no MuLan). Writes clips to
    data/generation/ab_tempo/ and returns a tidy DataFrame.
    """
    import time, warnings
    import numpy as np, pandas as pd
    from dataclasses import replace
    import librosa
    from . import acestep
    C.ensure_dirs()
    ab_dir = C.GEN_DIR / "ab_tempo"; ab_dir.mkdir(parents=True, exist_ok=True)
    gen = AceStepGenerator(dry_run=False)
    acestep.check_server()
    L = lyric_emotion or emotions[0]
    lyrics_text = lyr.get(L).acestep_lyrics()

    rows = []
    for M in emotions:
        m = emo.get(M)
        for think in thinking_values:
            out = ab_dir / f"{M}__think{int(think)}.wav"
            spec = GenSpec(caption=m.caption, lyrics=lyrics_text, out_path=out,
                           bpm=m.bpm, keyscale=m.keyscale, duration=C.CHORUS_DURATION_S,
                           seed=C.GRID_SEED, emotion=M, thinking=think)
            t0 = time.monotonic()
            gen.generate(spec, force=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(str(out), sr=22050, mono=True)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rows.append({"music_emotion": M, "thinking": think,
                         "requested_bpm": m.bpm, "measured_bpm": round(float(tempo), 1),
                         "gen_s": round(time.monotonic() - t0, 0)})
            logger.info("  %s think=%s: requested %d, measured %.0f BPM",
                        M, think, m.bpm, float(tempo))
    df = pd.DataFrame(rows)
    for think in thinking_values:
        sub = df[df.thinking == think]
        if len(sub) > 1:
            r = np.corrcoef(sub.requested_bpm, sub.measured_bpm)[0, 1]
            logger.info("thinking=%s: corr(requested, measured BPM) = %.2f "
                        "(spread %.0f BPM)", think, r, sub.measured_bpm.max() - sub.measured_bpm.min())
    return df


# ─── Multi-backend generation + pilot comparison ─────────────────────────────────
def get_backend(name: str | None = None, dry_run: bool | None = None):
    """Return a generation backend (same generate(GenSpec)->wav interface for all)."""
    name = (name or C.BACKEND).lower()
    if name == "acestep":
        return AceStepGenerator(dry_run=dry_run)
    if name == "lyria":
        from .lyria import LyriaGenerator
        return LyriaGenerator(dry_run=dry_run)
    if name == "suno":
        from .suno import SunoGenerator
        return SunoGenerator(dry_run=dry_run)
    raise ValueError(f"unknown backend {name!r} (acestep | lyria | suno)")


def pilot(backends=("lyria", "suno"), emotions=("ecstasy", "grief", "rage", "terror"),
          cross: bool = False, dry_run: bool | None = None, force: bool = False):
    """Generate the SAME hooks on two+ backends and compare lyric intelligibility (WER)
    and emotion accuracy (audio valence/arousal vs the design target). Clips go to
    data/generation/pilot/<backend>/<L>__<M>.wav; returns a tidy comparison DataFrame
    (also results/generation/pilot_comparison.csv).

    By default: the congruent diagonal of a few emotions (cheap). cross=True does the
    full subset grid. Light — uses Whisper + librosa only (no MuLan), so it's fine to
    run right after generation.
    """
    import pandas as pd
    from . import asr, va
    C.ensure_dirs()
    dry = C.DRY_RUN if dry_run is None else dry_run
    transcriber = None if dry else asr.Transcriber()
    cells = [(e, e) for e in emotions] if not cross else [(L, M) for L in emotions for M in emotions]

    rows = []
    for bk in backends:
        gen = get_backend(bk, dry_run=dry)
        if not dry:
            gen.check()
        adir = C.GEN_DIR / "pilot" / bk
        adir.mkdir(parents=True, exist_ok=True)
        for (L, M) in cells:
            m, ch = emo.get(M), lyr.get(L)
            out = adir / f"{L}__{M}.wav"
            spec = GenSpec(caption=m.caption, lyrics=ch.acestep_lyrics(), out_path=out,
                           bpm=m.bpm, keyscale=m.keyscale, duration=C.CHORUS_DURATION_S,
                           seed=cell_seed(L, M), emotion=M)
            logger.info("[pilot:%s] %s__%s", bk, L, M)
            gen.generate(spec, force=force)
            rec = {"backend": bk, "lyric_emotion": L, "music_emotion": M,
                   "congruent": (L == M), "path": str(out)}
            av = va.audio_va(out) or (float("nan"), float("nan"))
            rec["audio_v"], rec["audio_a"] = av
            rec["va_dist_to_design"] = va.va_distance(av, (m.valence, m.arousal))
            if transcriber is not None:
                hook = " ".join(ch.lines)
                info = transcriber.transcribe_detailed(out)
                rec["wer"] = min(asr.word_error_rate(r, info["text"]) for r in (hook, hook + " " + hook))
                rec["vocal_present"] = info["vocal_present"]
                rec["transcript"] = info["text"]
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "pilot_comparison.csv", index=False)
    if "wer" in df.columns:
        summary = df.groupby("backend").agg(
            median_wer=("wer", "median"), mean_va_dist=("va_dist_to_design", "mean"),
            vocal_present=("vocal_present", "mean")).round(3)
        logger.info("Pilot comparison:\n%s", summary.to_string())
    return df
