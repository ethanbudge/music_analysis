"""
lmc — Lyric-Music Congruence pipeline.

A resumable, LRCLIB-sourced pipeline that samples songs with time-synced
lyrics, gathers audio and popularity metrics, computes joint audio-text
embeddings (MuQ-MuLan, LAION-CLAP), and measures lyric-music congruence at the
song, segment (chorus vs. non-chorus), and line level (with audio context
windows of 0/1/5/10 s).

Typical session (see notebooks/pipeline.ipynb):

    from lmc import config, lrclib, audio, popularity, mood, embeddings, alignment, combine
    config.ensure_dirs()
    lrclib.setup()                 # report universe / sampled / remaining
    lrclib.sample(50)              # draw a session target
    audio.download_pending()       # YouTube official-audio download
    popularity.fetch_pending()     # Spotify + YouTube + (optional) Last.fm/Deezer
    mood.extract_pending()         # librosa mood proxies
    embeddings.embed_pending("mulan"); embeddings.embed_pending("clap")
    alignment.compute_pending()    # song / segment / line-window LMC
    combine.build_master()         # results/master_results.csv  (+ line/long tables)
"""

from . import config  # noqa: F401

__all__ = [
    "config", "utils", "db", "lrclib", "audio",
    "popularity", "mood", "chorus", "embeddings", "alignment", "combine",
]

__version__ = "2.0.0"
