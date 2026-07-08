"""
suno.py — Suno v5.5 backend (via a third-party API provider).

Drop-in generator for the LMC pipeline (same interface as AceStepGenerator). Suno has
no official public API, so this targets a provider-compatible REST surface (default
api.sunoapi.org): submit a customMode job (where `prompt` is used STRICTLY as the sung
lyrics and `style` carries the emotion), poll the task, download the result.

Setup: pick a provider + set SUNO_API_KEY (and SUNO_API_URL if not the default). See
config.SUNO. LICENSING: Suno grants a commercial license, not ownership — confirm it
covers research + stimulus sharing. Provider request/response shapes vary — every HTTP
error dumps the raw body; adjust _submit/_poll to your provider's docs if needed.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path

from . import config as C
from .acestep import GenSpec, _mock_waveform, _write_wav, _transcode_to_wav

logger = logging.getLogger(__name__)


class SunoGenerator:
    """Suno client with the same interface as AceStepGenerator."""

    def __init__(self, dry_run: bool | None = None):
        self.dry_run = C.DRY_RUN if dry_run is None else dry_run
        self._session = None

    def check(self) -> bool:
        if self.dry_run:
            return True
        import requests  # noqa: F401
        if not C.SUNO["api_key"]:
            raise RuntimeError("Set SUNO_API_KEY for Suno (config.SUNO['api_key']).")
        return True

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {C.SUNO['api_key']}", "Content-Type": "application/json"}

    def generate(self, spec: GenSpec, force: bool = False) -> Path:
        out = Path(spec.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not force:
            logger.info("  exists, skip: %s", out.name)
            return out
        if self.dry_run:
            _write_wav(out, _mock_waveform(spec, int(C.ACESTEP["sample_rate"])),
                       int(C.ACESTEP["sample_rate"]))
            logger.info("  [MOCK suno] %s", out.name)
            return out
        self._generate_real(spec, out)
        return out

    def _generate_real(self, spec: GenSpec, out: Path) -> None:
        self.check()
        s = self._http()
        base = C.SUNO["base_url"].rstrip("/")
        # customMode: prompt is used strictly as lyrics; style carries the emotion.
        payload = {
            "customMode": True, "instrumental": False, "model": C.SUNO["model"],
            "prompt": _plain_lyrics(spec.lyrics),
            "style": spec.caption, "title": out.stem,
        }
        r = s.post(f"{base}/api/v1/generate", json=payload, headers=self._headers(), timeout=30)
        _raise(r, f"generate({out.name})")
        task_id = _dig(r.json(), "data", "task_id") or _dig(r.json(), "data", "taskId") \
            or _dig(r.json(), "task_id")
        if not task_id:
            raise RuntimeError(f"Suno: no task_id for {out.name}: {r.text[:800]}")

        url = self._poll(base, s, task_id, out.name)
        ar = s.get(url, timeout=180)
        ar.raise_for_status()
        tmp = out.with_suffix(".mp3")
        tmp.write_bytes(ar.content)
        _transcode_to_wav(tmp, out, sr=int(C.ACESTEP["sample_rate"]))
        if tmp != out:
            tmp.unlink(missing_ok=True)
        logger.info("  generated (suno): %s", out.name)

    def _poll(self, base: str, s, task_id: str, name: str) -> str:
        deadline = time.monotonic() + C.SUNO["poll_timeout_s"]
        while True:
            try:
                r = s.get(f"{base}/api/v1/tasks/{task_id}", headers=self._headers(), timeout=30)
            except Exception:                                      # noqa: BLE001
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Suno poll timed out for {name}")
                time.sleep(C.SUNO["poll_interval_s"]); continue
            _raise(r, f"tasks({name})")
            body = r.json()
            url = _first_audio_url(body)
            status = str(_dig(body, "data", "status") or _dig(body, "status") or "").lower()
            if url:
                return url
            if status in ("failed", "error"):
                raise RuntimeError(f"Suno generation failed for {name}: {body}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Suno generation timed out for {name} (task {task_id})")
            time.sleep(C.SUNO["poll_interval_s"])


def _plain_lyrics(acestep_lyrics: str) -> str:
    """Suno wants the lyrics text; keep [Chorus] tags (Suno understands them)."""
    return acestep_lyrics


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _first_audio_url(body) -> str | None:
    """Best-effort: find a downloadable audio URL in a provider task response."""
    data = body.get("data") if isinstance(body, dict) else None
    items = data if isinstance(data, list) else (
        data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), list) else None)
    for item in (items or []):
        if isinstance(item, dict):
            for k in ("audio_url", "audioUrl", "url", "source_audio_url"):
                if item.get(k):
                    return item[k]
    if isinstance(data, dict):
        for k in ("audio_url", "audioUrl", "url"):
            if data.get(k):
                return data[k]
    return None


def _raise(r, context: str) -> None:
    if r.status_code >= 400:
        raise RuntimeError(f"Suno API error during {context}: HTTP {r.status_code}\n{r.text[:1500]}")
