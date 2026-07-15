"""
lmcsvs — Singing-Voice-Synthesis route for the generative LMC arm.

The text-to-song route (`lmcgen` + ACE-Step) failed the experiment's hard
requirements: lyrics weren't sung verbatim or intelligibly, the voice drifted, and
the requested tempo/emotion was ignored. SVS flips the problem: instead of prompting
a stochastic model and hoping, we *specify* the performance — exact notes, exact
lyrics, one fixed voice — so lyrics and voice are guaranteed by construction and only
the musical emotion is manipulated.

Division of labour
------------------
This package does everything up to the vocal render, headlessly and deterministically:

  1. take each 2-line hook (lmcgen.lyrics) and split it into per-note **syllables**,
  2. generate a **melody per music-emotion** (valence/arousal → mode, tempo, register,
     rhythm, contour) — the user chose to vary the melody per emotion,
  3. assemble each grid cell (hook_L sung to melody_M) into a **Score**,
  4. export **MusicXML** (notes with lyrics attached, key + tempo) — imports straight
     into Synthesizer V Studio 2 Pro / ACE Studio.

The user then assigns ONE fixed voice in the SVS app and batch-renders → wavs, which
flow back into the existing WER / valence-arousal validation harness (lmcgen.asr /
lmcgen.va). Instrumental backing is a separate, later stage.

Modules: config, syllables, melody, score, musicxml, pipeline.
"""
from __future__ import annotations

__all__ = ["config", "syllables", "melody", "score", "musicxml", "pipeline"]
