"""
lmcgen — the *generative / experimental* arm of the Lyric-Music Congruence project.

Where the observational arm (`lmc`) measures LMC on real songs sampled from LRCLIB,
this arm *manufactures* stimuli with LMC held under experimental control, for a
human-validation survey. The design is a 2x2 valence/arousal (VA) circumplex:

    4 extreme VA corners (music targets)  x  16 two-line lyrics (4 authored per corner)
    x  4 repetitions                      =  256 song-lyric pairs.

Every crossing of lyric-corner and music-corner is covered, so congruence varies
systematically from perfectly matched (the 4x4 diagonal) to opposite-corner mismatched.

The music is generated with Google **Lyria 3 Clip** (Gemini API). Lyria conditions on
prompt TEXT only — it cannot take a target embedding, has no seed and does not reproduce
— so each corner is hit by prompt engineering and validated post-hoc against a MuLan
quadrant anchor (audio-vs-anchor cosine, the embedding "target") plus an independent
librosa acoustic-VA measure and Whisper lyric-WER.

Modules
-------
config     paths, VA design constants, Lyria config, dry-run switch
quadrants  the 4 extreme VA corners: coordinates, Lyria style words, MuLan anchor
           prompts, lexicon, representative bpm/key
lyrics     the 16 authored two-line couplets + lexical / VA placement checks
audioio    GenSpec + audio I/O helpers (mock synth, WAV write, transcode)
lyria      Google Lyria 3 Clip backend (real + dry-run mock), resumable
mulan      MuLan/CLAP scorer + VA-corner anchor builder (reuses `lmc.embeddings`)
asr        Whisper transcription + word error rate (lyric-presence screen)
va         librosa acoustic VA + lexicon lyric VA + VA-congruence
generate   Phase 1: build the 256 specs and generate them with Lyria
validate   Phase 2: WER + MuLan + VA on every clip -> results/generation/songs.csv
analysis   descriptive stats, figures, winner selection, survey export
"""
from __future__ import annotations

__all__ = ["config", "quadrants", "lyrics", "audioio", "lyria", "mulan",
           "asr", "va", "generate", "validate", "analysis"]
