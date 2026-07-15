"""
generate.py — Phase 1: build the song specs and generate them with Lyria.

Full design: 16 lyrics x 4 music corners x REPS_PER_CELL reps = 256 songs. Each cell is
a (lyric_id, music_quadrant) pairing; its congruence is fixed by whether the lyric's own
corner matches the music corner (the 4x4 matrix diagonal = congruent). The couplet is
sung verbatim; the corner supplies the musical style. Generation is resumable by file
existence, so a stopped batch just re-runs and skips finished clips.

Smoke-test toggles (config.ACTIVE_LYRICS_PER_QUADRANT, config.REPS_PER_CELL) let you
shrink the batch before spending API quota — see active_lyrics() and config.py.

Lyria is non-deterministic and cannot take an embedding, so by default we generate each
rep once (config.CANDIDATES_PER_SLOT == 1) and let validate.py score all specs — winners
are chosen afterwards. Set LMCGEN_CANDIDATES>1 to instead keep, per rep, the best-of-N
take by lyric WER (needs Whisper + a real backend).
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

from . import config as C
from . import quadrants as q
from . import lyrics as lyr
from .audioio import GenSpec
from .lyria import LyriaGenerator

logger = logging.getLogger(__name__)


# ─── cell / path identity ────────────────────────────────────────────────────────
def cell_path(lyric_id: str, music_quadrant: str, rep: int) -> Path:
    return C.AUDIO_DIR / f"{lyric_id}__{music_quadrant}__rep{rep}.wav"


def build_prompt(couplet: "lyr.Couplet", quad: "q.Quadrant") -> str:
    """Compose the Lyria `input`: fixed voice + corner music style + the verbatim lyrics."""
    return (f"{C.VOICE_BLURB}. {quad.style_words}. "
            f"Around {quad.bpm} BPM, {quad.keyscale}.\n\n{couplet.chorus_block()}")


def active_lyrics() -> list["lyr.Couplet"]:
    """The lyrics actually included in this run: the first N per quadrant, where
    N = config.ACTIVE_LYRICS_PER_QUADRANT clamped to [1, config.LYRICS_PER_QUADRANT].
    Lower ACTIVE_LYRICS_PER_QUADRANT (and/or config.REPS_PER_CELL) in the notebook
    before generating for a cheap smoke test — e.g. 1 lyric/corner x 1 rep covers all
    16 cells of the 4x4 congruence matrix in a single song each."""
    n = C.ACTIVE_LYRICS_PER_QUADRANT
    if not (1 <= n <= C.LYRICS_PER_QUADRANT):
        clamped = max(1, min(n, C.LYRICS_PER_QUADRANT))
        logger.warning("ACTIVE_LYRICS_PER_QUADRANT=%d out of range [1,%d]; using %d",
                       n, C.LYRICS_PER_QUADRANT, clamped)
        n = clamped
    out: list["lyr.Couplet"] = []
    for code in q.ORDER:
        out.extend(lyr.by_quadrant(code)[:n])
    return out


def build_specs() -> list[GenSpec]:
    """GenSpecs for the active design: active_lyrics() x music corners x REPS_PER_CELL
    (256 at the full-study defaults; fewer during a smoke test)."""
    specs: list[GenSpec] = []
    for couplet in active_lyrics():
        for mcode in q.ORDER:                     # 4 music corners
            quad = q.get(mcode)
            prompt = build_prompt(couplet, quad)
            for rep in range(C.REPS_PER_CELL):    # reps
                specs.append(GenSpec(
                    prompt=prompt, sung_text=couplet.plain,
                    out_path=cell_path(couplet.lyric_id, mcode, rep),
                    music_quadrant=mcode, valence=quad.valence, arousal=quad.arousal,
                    bpm=quad.bpm, keyscale=quad.keyscale,
                    lyric_id=couplet.lyric_id, rep=rep,
                ))
    return specs


# ─── WER reference (credit one or two sung reps of the couplet) ──────────────────
def wer_references(lyric_id: str) -> list[str]:
    hook = lyr.get(lyric_id).plain
    return [hook, hook + " " + hook]


# ─── Phase 1 driver ──────────────────────────────────────────────────────────────
def generate_all(dry_run: bool | None = None, force: bool = False) -> int:
    """Generate all 256 songs (resumable). Returns the number of specs processed."""
    C.ensure_dirs()
    dry_run = C.DRY_RUN if dry_run is None else dry_run
    gen = LyriaGenerator(dry_run=dry_run)
    if not dry_run:
        gen.check()

    candidates = C.CANDIDATES_PER_SLOT
    transcriber = None
    if not dry_run and candidates > 1 and C.ASR["enabled"]:
        from . import asr
        transcriber = asr.Transcriber()          # best-of-N per rep by WER

    specs = build_specs()
    logger.info("Generating %d songs (dry_run=%s, candidates/slot=%d)…",
                len(specs), dry_run, candidates)
    t0 = time.monotonic()
    for i, spec in enumerate(specs, 1):
        out = Path(spec.out_path)
        logger.info("[%d/%d] %s x %s rep%d", i, len(specs), spec.lyric_id,
                    spec.music_quadrant, spec.rep)
        if out.exists() and not force:
            logger.info("     exists, skip")
        elif transcriber is None:
            gen.generate(spec, force=force)
        else:
            _best_of_n(gen, spec, transcriber)
        elapsed = time.monotonic() - t0
        eta = elapsed / i * (len(specs) - i)
        logger.info("     elapsed %.0fm, ~%.0fm remaining", elapsed / 60, eta / 60)
    logger.info("Phase 1 complete: %d clips under %s", len(specs), C.AUDIO_DIR)
    return len(specs)


def _best_of_n(gen: LyriaGenerator, spec: GenSpec, transcriber) -> None:
    """Generate config.ASR['max_takes'] takes for one rep and keep the lowest-WER one."""
    import shutil
    from dataclasses import replace
    refs = wer_references(spec.lyric_id)
    out = Path(spec.out_path)
    takes, best = [], None
    for k in range(C.ASR["max_takes"]):
        tp = out.with_name(out.stem + f"__take{k}.wav")
        gen.generate(replace(spec, out_path=tp), force=True)
        info = transcriber.transcribe_detailed(tp)
        from . import asr
        wer = min(asr.word_error_rate(r, info["text"]) for r in refs)
        logger.info("     take %d: WER=%.2f%s", k, wer,
                    "" if info["vocal_present"] else " [no vocal]")
        takes.append(tp)
        if best is None or wer < best[0]:
            best = (wer, tp)
        if wer <= C.ASR["accept_wer"] and info["vocal_present"]:
            break
    shutil.copyfile(best[1], out)
    if not C.ASR["keep_takes"]:
        for tp in takes:
            Path(tp).unlink(missing_ok=True)


# ─── housekeeping ────────────────────────────────────────────────────────────────
def clean_generated(audio: bool = True, embeddings: bool = True,
                    anchors: bool = True, results: bool = True) -> None:
    """Delete cached generation artifacts. The pipeline is resumable by file existence,
    so mock (dry-run) clips would otherwise be REUSED when you switch to real Lyria —
    call this once when flipping LMCGEN_DRY_RUN between 1 and 0."""
    import shutil
    targets = []
    if audio:      targets.append(C.AUDIO_DIR)
    if embeddings: targets.append(C.EMB_DIR)
    if anchors:    targets.append(C.ANCHOR_DIR)
    if results:    targets.append(C.RESULTS_DIR)
    for d in targets:
        if Path(d).exists():
            shutil.rmtree(d)
            logger.info("removed %s", d)
    C.ensure_dirs()


def coverage() -> dict:
    """How many of the 256 clips currently exist on disk (for the notebook)."""
    specs = build_specs()
    done = sum(1 for s in specs if Path(s.out_path).exists())
    return {"total": len(specs), "generated": done, "remaining": len(specs) - done}
