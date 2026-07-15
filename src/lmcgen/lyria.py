"""
lyria.py — Google Lyria 3 Clip backend (via the Gemini API / google-genai).

Thin transport with a single interface: generate(GenSpec) -> wav Path. The prompt is
pre-composed by generate.build_prompt() (voice descriptor + music style + [Chorus]
lyrics) and carried on spec.prompt; this module just sends it and saves the audio.

Verified API surface (ai.google.dev/gemini-api/docs/interactions/music-generation):

    interaction = client.interactions.create(
        model="lyria-3-clip-preview",     # fixed 30 s clip
        input=prompt,                     # lyrics as [Chorus] tags + style in prose
    )
    audio  = interaction.output_audio     # base64 MP3 in .data
    lyrics = interaction.output_text      # echoed / generated lyrics (fidelity cross-check)

There is no seed, negative prompt, or embedding field — voice/tempo/key/mood live in the
prompt text, and results are non-deterministic (hence best-of-N selection lives upstream).
All output carries a SynthID watermark. Setup: `pip install google-genai`; GEMINI_API_KEY.
"""
from __future__ import annotations
import base64
import logging
import time
from pathlib import Path

from . import config as C
from . import audioio
from .audioio import GenSpec

logger = logging.getLogger(__name__)


class LyriaGenerator:
    """Lyria 3 Clip client. generate(spec) writes a canonical wav to spec.out_path."""

    def __init__(self, dry_run: bool | None = None):
        self.dry_run = C.DRY_RUN if dry_run is None else dry_run
        self._client = None

    def check(self) -> bool:
        if self.dry_run:
            return True
        try:
            import google.genai  # noqa: F401
        except ImportError as e:
            raise ImportError("Lyria needs google-genai: pip install google-genai") from e
        if not C.LYRIA["api_key"]:
            raise RuntimeError("GEMINI_API_KEY is not set (config.LYRIA['api_key']). "
                               "Put it in notebooks/.env and load it before generating.")
        return True

    def _client_(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=C.LYRIA["api_key"])
        return self._client

    def generate(self, spec: GenSpec, force: bool = False) -> Path:
        out = Path(spec.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not force:
            logger.info("  exists, skip: %s", out.name)
            return out
        if self.dry_run:
            wav = audioio._mock_waveform(spec, C.SAMPLE_RATE, C.CLIP_DURATION_S,
                                         seed=abs(hash(out.name)) % (2 ** 32))
            audioio._write_wav(out, wav, C.SAMPLE_RATE)
            logger.info("  [MOCK lyria] %s", out.name)
            return out
        self._generate_real(spec, out)
        return out

    def _generate_real(self, spec: GenSpec, out: Path) -> None:
        self.check()
        client = self._client_()
        interaction = self._create_with_retry(client, spec.prompt, out.name)

        data = _extract_audio_b64(interaction)
        if not data:
            raise RuntimeError(f"Lyria returned no audio for {out.name}: {interaction!r}")
        tmp = out.with_suffix(".mp3")
        tmp.write_bytes(base64.b64decode(data))
        audioio._transcode_to_wav(tmp, out, sr=C.SAMPLE_RATE)
        tmp.unlink(missing_ok=True)

        # Echoed lyrics: a free lyric-fidelity cross-check alongside the Whisper WER.
        text = _extract_output_text(interaction)
        if text:
            out.with_suffix(".lyria.txt").write_text(text)
        logger.info("  generated (lyria): %s", out.name)

    def _create_with_retry(self, client, prompt: str, name: str):
        last = None
        for attempt in range(C.LYRIA["max_retries"]):
            try:
                return client.interactions.create(model=C.LYRIA["model"], input=prompt)
            except Exception as e:                                  # noqa: BLE001
                last = e
                wait = C.LYRIA["retry_backoff_s"] * (2 ** attempt)
                logger.warning("  lyria attempt %d/%d for %s failed (%s); retrying in %.0fs",
                               attempt + 1, C.LYRIA["max_retries"], name, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"Lyria failed after {C.LYRIA['max_retries']} attempts for {name}") from last


# ─── response accessors (defensive across attr / dict shapes) ────────────────────
def _get(obj, key):
    if obj is None:
        return None
    return getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key)


def _extract_audio_b64(interaction):
    audio = _get(interaction, "output_audio")
    return _get(audio, "data")


def _extract_output_text(interaction):
    return _get(interaction, "output_text")
