"""
validate.py — Validate rendered SVS vocals with the existing WER / VA instruments.

Once the MusicXML scores are rendered to data/svs/audio/<L>__<M>.wav in the SVS app
(one fixed voice), this reuses the lmcgen harness — Whisper WER for lyric
intelligibility (should be near-zero now, since the lyrics are sung by construction)
and valence/arousal for the emotion manipulation.
"""
from __future__ import annotations
import logging

from . import config as C
from . import pipeline
from lmcgen import emotions as emo, lyrics as lyr

logger = logging.getLogger(__name__)


def audio_path(lyric_emotion: str, music_emotion: str):
    return C.AUDIO_DIR / f"{pipeline.cell_name(lyric_emotion, music_emotion)}.wav"


def validate(emotions: list[str] | None = None):
    """Score every rendered cell for WER + valence/arousal. Skips cells whose wav
    hasn't been rendered yet (warns). Writes results/svs/svs_results.csv."""
    import pandas as pd
    from lmcgen import asr, va
    C.ensure_dirs()
    emotions = emotions or C.EMOTIONS
    transcriber = asr.Transcriber()
    recs, missing = [], 0
    for L in emotions:
        hook = " ".join(lyr.get(L).lines)
        refs = [hook, hook + " " + hook]
        lv, la, lmatch = va.lyric_va(hook)
        for M in emotions:
            wav = audio_path(L, M)
            if not wav.exists():
                missing += 1
                continue
            info = transcriber.transcribe_detailed(wav)
            wer = min(va_wer(r, info["text"]) for r in refs)
            av = va.audio_va(wav) or (float("nan"), float("nan"))
            dm, dl = emo.get(M), emo.get(L)
            recs.append({
                "lyric_emotion": L, "music_emotion": M, "congruent": (L == M),
                "wer": wer, "vocal_present": info["vocal_present"], "transcript": info["text"],
                "audio_v": av[0], "audio_a": av[1], "lyric_v": lv, "lyric_a": la,
                "design_music_v": dm.valence, "design_music_a": dm.arousal,
                "va_congruence": va.va_congruence(av, (lv, la)),
                "va_congruence_design": va.va_congruence((dl.valence, dl.arousal),
                                                         (dm.valence, dm.arousal)),
            })
    if missing:
        logger.warning("%d/%d cells not rendered yet (no wav in %s)", missing,
                       len(emotions) ** 2, C.AUDIO_DIR)
    df = pd.DataFrame(recs)
    if not df.empty:
        df.to_csv(C.RESULTS_DIR / "svs_results.csv", index=False)
        logger.info("Validated %d cells → %s", len(df), C.RESULTS_DIR / "svs_results.csv")
    return df


def va_wer(reference: str, hypothesis: str) -> float:
    from lmcgen import asr
    return asr.word_error_rate(reference, hypothesis)
