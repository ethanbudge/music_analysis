"""
lyria.py — Google Lyria 3 backend (via the Gemini API / google-genai).

Drop-in generator for the LMC pipeline: same interface as AceStepGenerator
(generate(GenSpec)->wav), so pipeline generation + the shared WER/VA/MuLan
validation work unchanged. Lyria 3 takes the lyrics as `[Chorus]…` tags inside the
prompt and returns base64 audio; emotion/tempo are expressed in natural language in
the prompt (no seed/negative-prompt fields are exposed).

Setup: `pip install google-genai`; set GEMINI_API_KEY. See config.LYRIA.
Verify field/method names against your google-genai version — the media/interactions
API is young; every error surfaces the raw response.
"""
from __future__ import annotations
import base64
import logging
from pathlib import Path

from . import config as C
from .acestep import GenSpec, _mock_waveform, _write_wav, _transcode_to_wav

logger = logging.getLogger(__name__)


def build_prompt(spec: GenSpec) -> str:
    """Compose the Lyria prompt: emotion/style caption + tempo + the exact lyrics."""
    tempo = f" Around {spec.bpm} BPM." if spec.bpm else ""
    key = f" Key: {spec.keyscale}." if spec.keyscale else ""
    # spec.lyrics already carries [Chorus]… section tags.
    return f"{spec.caption}.{tempo}{key}\n\n{spec.lyrics}"


class LyriaGenerator:
    """Lyria 3 client with the same interface as AceStepGenerator."""

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
            raise RuntimeError("Set GEMINI_API_KEY for Lyria (config.LYRIA['api_key']).")
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
            _write_wav(out, _mock_waveform(spec, int(C.ACESTEP["sample_rate"])),
                       int(C.ACESTEP["sample_rate"]))
            logger.info("  [MOCK lyria] %s", out.name)
            return out
        self._generate_real(spec, out)
        return out

    def _generate_real(self, spec: GenSpec, out: Path) -> None:
        self.check()
        client = self._client_()
        prompt = build_prompt(spec)
        kwargs = {"model": C.LYRIA["model"], "input": prompt}
        if C.LYRIA["wav"]:
            kwargs["response_format"] = {"type": "audio"}   # WAV (Pro); MP3 default otherwise
        interaction = client.interactions.create(**kwargs)

        audio = getattr(interaction, "output_audio", None) or \
            (interaction.get("output_audio") if isinstance(interaction, dict) else None)
        data = getattr(audio, "data", None) or (audio.get("data") if isinstance(audio, dict) else None)
        if not data:
            raise RuntimeError(f"Lyria returned no audio for {out.name}: {interaction}")
        raw = base64.b64decode(data)
        suffix = ".wav" if C.LYRIA["wav"] else ".mp3"
        tmp = out.with_suffix(suffix)
        tmp.write_bytes(raw)
        _transcode_to_wav(tmp, out, sr=int(C.ACESTEP["sample_rate"]))
        if tmp != out:
            tmp.unlink(missing_ok=True)
        logger.info("  generated (lyria): %s", out.name)
