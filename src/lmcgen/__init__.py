"""
lmcgen — the *generative / experimental* arm of the Lyric-Music Congruence
project.

Where the observational arm (`lmc`) measures LMC on real songs sampled from
LRCLIB, this arm *manufactures* stimuli with LMC held under experimental control:
eight fixed choruses (lyric emotion) are each set to eight musical emotions,
giving an 8x8 grid of 64 choruses in which lyric-music congruence varies
systematically from perfectly matched (the diagonal) to strongly mismatched.

The music is generated with ACE-Step 1.5 (open-source, local). Because ACE-Step
conditions on *text* (a caption + lyrics) rather than a raw target embedding, the
eight musical emotions are realised as caption / bpm / key recipes that are then
*tuned* against MuQ-MuLan emotion anchors. The same MuLan space (plus optionally
LAION-CLAP) is used to validate that (a) the lyrics land in their target emotion,
(b) the generated music lands in its target emotion, and (c) congruent lyric-music
pairs embed more similarly than incongruent ones — i.e. that LMC is being captured.

Modules
-------
config    paths, ACE-Step + MuLan settings, the dry-run switch
emotions  the 8 Plutchik emotions: valence/arousal, MuLan anchor prompts,
          lexicon, and ACE-Step caption/bpm/key recipes (+ search variants)
lyrics    the 8 authored choruses + rationale, and lexical alignment scoring
mulan     MuLan/CLAP scorer + emotion-anchor builder (reuses `lmc.embeddings`)
acestep   ACE-Step 1.5 generation wrapper (real + dry-run), resumable
pipeline  orchestration: tune recipes -> generate 64 -> validate -> tidy results
analysis  descriptive statistics + plots (the deliverable figures)
"""
from __future__ import annotations

__all__ = ["config", "emotions", "lyrics", "mulan", "acestep", "pipeline", "analysis"]
