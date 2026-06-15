"""
audio.py — Download official song audio (not music videos) for sampled songs.

For each song in the corpus we search YouTube and prefer the *audio* upload over
a music video, because music-video edits (intros, skits, extended outros) shift
the timeline and would desync the LRCLIB timestamps. Preference order:

  1. Auto-generated "<Artist> - Topic" channels (label-delivered official audio),
  2. titles containing "audio" / "official audio" / "lyric(s)",
  3. closest duration match to the LRCLIB track duration,
  4. penalise titles containing "video" / "live" / "performance".

Audio is saved as data/audio/<track_id>.mp3. Public YouTube metrics
(view / like / comment counts) are captured at the same time for the
popularity model. The stage is resumable: songs already marked 'done' or
'not_found' are skipped.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import AUDIO_DIR
from . import db as projdb

logger = logging.getLogger(__name__)

_VIDEO_NEG = ("video", "live", "performance", "concert", "vevo presents",
              "behind the scenes", "interview", "reaction", "trailer")
_AUDIO_POS = ("official audio", "audio", "lyric", "lyrics", "full song", "hq audio")


def audio_path(track_id: int) -> Path:
    return Path(AUDIO_DIR) / f"{track_id}.mp3"


def _score_candidate(info: dict, target_dur: float | None) -> float:
    """Higher is better. Encodes the audio-over-video preference order."""
    title   = (info.get("title") or "").lower()
    channel = (info.get("channel") or info.get("uploader") or "").lower()
    score   = 0.0

    if channel.endswith("- topic") or channel.endswith("- topic "):
        score += 100.0                       # auto-generated official audio
    if "vevo" in channel:
        score -= 5.0                         # usually the video
    score += sum(8.0 for kw in _AUDIO_POS if kw in title)
    score -= sum(12.0 for kw in _VIDEO_NEG if kw in title)

    dur = info.get("duration")
    if target_dur and dur:
        diff = abs(dur - target_dur)
        if diff <= 3:    score += 30.0
        elif diff <= 8:  score += 15.0
        elif diff <= 15: score += 5.0
        else:            score -= min(diff, 60.0)   # large mismatch = wrong track
    return score


def _search_best(ydl, query: str, target_dur: float | None, n: int = 6) -> dict | None:
    """Return the best-scoring candidate's full info dict, or None."""
    try:
        res = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    except Exception as e:                                     # noqa: BLE001
        logger.warning("  search failed: %s", e)
        return None
    entries = [e for e in (res or {}).get("entries", []) if e]
    if not entries:
        return None
    entries.sort(key=lambda e: _score_candidate(e, target_dur), reverse=True)
    return entries[0]


def _download_one(song: dict) -> dict:
    """Search + download one song. Returns a row dict for the `audio` table."""
    import yt_dlp

    track_id = song["track_id"]
    target   = song["duration"]
    query    = f'{song["artist"]} {song["title"]} audio'
    out_tmpl = str(Path(AUDIO_DIR) / f"{track_id}.%(ext)s")
    now      = datetime.now(timezone.utc).isoformat()

    base_opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "default_search": "ytsearch", "skip_download": True,
    }
    with yt_dlp.YoutubeDL(base_opts) as ydl:
        best = _search_best(ydl, query, target)

    if best is None:
        return {"track_id": track_id, "status": "not_found", "source": "youtube",
                "error": "no candidates", "fetched_at": now}

    video_id = best.get("id")
    channel  = best.get("channel") or best.get("uploader") or ""
    dl_opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "postprocessors": [{
            "key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192",
        }],
    }
    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}",
                                    download=True)
    except Exception as e:                                     # noqa: BLE001
        return {"track_id": track_id, "status": "failed", "source": "youtube",
                "youtube_id": video_id, "error": str(e)[:300], "fetched_at": now}

    fp = audio_path(track_id)
    if not fp.exists():
        return {"track_id": track_id, "status": "failed", "source": "youtube",
                "youtube_id": video_id, "error": "file missing after download",
                "fetched_at": now}

    return {
        "track_id":      track_id,
        "status":        "done",
        "source":        "youtube",
        "youtube_id":    video_id,
        "youtube_title": info.get("title"),
        "channel":       channel,
        "is_topic":      int(channel.lower().strip().endswith("- topic")),
        "duration_s":    info.get("duration"),
        "view_count":    info.get("view_count"),
        "like_count":    info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "file_path":     str(fp),
        "error":         None,
        "fetched_at":    now,
    }


def download_pending(limit: int | None = None) -> dict:
    """
    Download audio for every corpus song that doesn't have it yet.

    `limit` caps how many to attempt this call (handy for incremental sessions).
    Each result is written to the DB immediately, so interrupting is safe.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with projdb.connect() as conn:
        pending = projdb.pending_for_audio(conn)
    if limit:
        pending = pending[:limit]

    if not pending:
        logger.info("Audio: nothing pending.")
        return {"attempted": 0, "done": 0, "failed": 0}

    logger.info("Audio: %d songs to fetch.", len(pending))
    done = failed = 0
    for i, song in enumerate(pending, 1):
        s = dict(song)
        logger.info("[%d/%d] %s — %s", i, len(pending), s["artist"], s["title"])
        row = _download_one(s)
        with projdb.connect() as conn:
            projdb.upsert(conn, "audio", row)
        if row["status"] == "done":
            done += 1
            logger.info("    ✓ %s (%s views)", row.get("channel"),
                        f'{row.get("view_count"):,}' if row.get("view_count") else "?")
        else:
            failed += 1
            logger.info("    ✗ %s", row.get("status"))

    logger.info("Audio done: %d ok, %d failed/not-found.", done, failed)
    return {"attempted": len(pending), "done": done, "failed": failed}
