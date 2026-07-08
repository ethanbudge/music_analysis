"""
acestep.py — ACE-Step 1.5 generation wrapper (real + dry-run), resumable.

ACE-Step conditions on TEXT (a caption + lyrics) plus scalar metadata (bpm,
keyscale, duration) — it does NOT accept a raw target embedding. We therefore
drive the DiT directly with an explicit caption/bpm/key recipe and DISABLE the
5Hz LM planner ("thinking") so it doesn't rewrite our chosen emotion metadata.

ACE-Step 1.5 lives in ITS OWN `uv`-managed environment (separate from the `lmc`
conda env this package/notebook runs in) — it is not `pip install`-able here, and
importing it in-process would risk clashing with `lmc`'s pinned torch/numpy stack.
So `_generate_real()` talks to ACE-Step's REST API SERVER as a separate process
(async task API: POST /release_task -> poll POST /query_result -> GET /v1/audio).
Start the server first (see config.py's ACESTEP block for the command) and call
`check_server()` to fail fast with a clear message if it isn't reachable.

The exact request/response field names are young and under-documented. Every HTTP
error here surfaces the full server response body, and `check_server()` points you
at the server's own auto-docs (<ACESTEP_API_URL>/docs) — if a call 422s, that's
your fastest way to see the real schema and fix a field name.

Dry-run mode (config.DRY_RUN, default on) synthesises a cheap emotion-dependent
waveform instead, so the full pipeline — lyrics, anchors, MuLan validation,
statistics, plots, notebook — runs end-to-end before you spend hours generating.
"""
from __future__ import annotations
import logging
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C
from . import emotions as emo

logger = logging.getLogger(__name__)


@dataclass
class GenSpec:
    caption: str
    lyrics: str                 # ACE-Step lyric field (with [Chorus] tag)
    out_path: Path
    bpm: int
    keyscale: str
    duration: float = C.CHORUS_DURATION_S
    seed: int = C.GRID_SEED
    emotion: str = ""           # target musical emotion (dry-run synth + logging)
    thinking: bool | None = None  # per-call override of config.ACESTEP['thinking']


class AceStepGenerator:
    """Talks to a running ACE-Step API server (real mode) and generates clips;
    resumable per file. Call `check_server()` before a long batch to fail fast."""

    def __init__(self, dry_run: bool | None = None):
        self.dry_run = C.DRY_RUN if dry_run is None else dry_run
        self._session = None        # requests.Session, created lazily (connection reuse)

    def check(self) -> bool:
        """Uniform backend interface: verify the ACE-Step server is reachable."""
        return True if self.dry_run else check_server()

    # ── public API ───────────────────────────────────────────────────────────────
    def generate(self, spec: GenSpec, force: bool = False) -> Path:
        """Generate one clip to spec.out_path (skipping if it already exists)."""
        out = Path(spec.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not force:
            logger.info("  exists, skip: %s", out.name)
            return out
        if self.dry_run:
            self._generate_mock(spec, out)
        else:
            self._generate_real(spec, out)
        return out

    # ── real ACE-Step path (the single integration point: the REST API) ──────────
    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if C.ACESTEP["api_key"]:
            h["Authorization"] = f"Bearer {C.ACESTEP['api_key']}"
        return h

    def _generate_real(self, spec: GenSpec, out: Path) -> None:
        base = C.ACESTEP["api_url"].rstrip("/")
        s = self._http()

        # Field names verified against acestep/api/http/release_task_models.py
        # (GenerateMusicRequest): prompt / key_scale / audio_duration / seed etc.
        payload = {
            "task_type": "text2music",
            "prompt": spec.caption,          # the caption / style-mood description
            "lyrics": spec.lyrics,
            "bpm": int(spec.bpm),
            "key_scale": spec.keyscale,
            "vocal_language": "en",
            "audio_duration": float(spec.duration),
            "seed": int(spec.seed),
            "use_random_seed": False,        # honour our deterministic per-cell seed
            "inference_steps": int(C.ACESTEP["inference_steps"]),
            "guidance_scale": float(C.ACESTEP["guidance_scale"]),
            "thinking": bool(C.ACESTEP["thinking"] if spec.thinking is None else spec.thinking),
            "audio_format": "wav",           # request wav directly (still normalised below)
        }
        if C.ACESTEP["model"]:
            payload["model"] = C.ACESTEP["model"]

        r = s.post(f"{base}/release_task", json=payload, headers=self._headers(),
                   timeout=C.ACESTEP["request_timeout_s"])
        _raise_for_api_error(r, f"release_task({out.name})")
        task_id = _dig(r.json(), "data", "task_id") or _dig(r.json(), "task_id")
        if not task_id:
            raise RuntimeError(f"release_task({out.name}) returned no task_id: {r.text[:1000]}")

        entry = self._poll(base, s, task_id, out.name)
        server_path = _extract_audio_path(entry)
        if not server_path:
            raise RuntimeError(f"generation succeeded for {out.name} but no audio file path was "
                              f"found in the query_result entry — inspect and adjust "
                              f"_extract_audio_path(): {entry}")

        audio_url = base + _path_to_url(server_path)
        ar = s.get(audio_url, headers=self._headers(), timeout=180)
        ar.raise_for_status()
        suffix = Path(server_path).suffix or ".wav"
        tmp = out.with_suffix(suffix)
        tmp.write_bytes(ar.content)
        _transcode_to_wav(tmp, out, sr=int(C.ACESTEP["sample_rate"]))
        if tmp != out:
            tmp.unlink(missing_ok=True)
        logger.info("  generated: %s", out.name)

    def _poll(self, base: str, s, task_id: str, name: str) -> dict:
        # Endpoint + payload verified against acestep/api/http/query_result_route.py
        # + query_result_service.py: key is `task_id_list` (a list is accepted),
        # each entry is {task_id, status(int 0/1/2), result(JSON string)}.
        #
        # IMPORTANT: while ACE-Step is generating (MLX compute holds the GIL), the
        # server may not answer this poll within a single request timeout. That is
        # EXPECTED, not a failure — a per-request Read/Connect timeout just means
        # "still working". We swallow those and keep polling until the overall
        # deadline (config.ACESTEP['poll_timeout_s']) is genuinely exceeded.
        import requests as _rq
        start = time.monotonic()
        deadline = start + C.ACESTEP["poll_timeout_s"]
        req_to = C.ACESTEP["request_timeout_s"]
        waited_msg_at = 0.0
        while True:
            try:
                r = s.post(f"{base}/query_result", json={"task_id_list": [task_id]},
                          headers=self._headers(), timeout=req_to)
            except (_rq.exceptions.Timeout, _rq.exceptions.ConnectionError) as e:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"ACE-Step generation for {name} exceeded "
                        f"{C.ACESTEP['poll_timeout_s']:.0f}s (task_id={task_id}). The server "
                        f"never returned a result — check its terminal for progress/errors; "
                        f"raise config.ACESTEP['poll_timeout_s'] if it's just slow.") from e
                elapsed = time.monotonic() - start
                if elapsed - waited_msg_at >= 30:        # heartbeat every ~30s
                    logger.info("     …still generating %s (%.0fs elapsed; server busy)", name, elapsed)
                    waited_msg_at = elapsed
                time.sleep(C.ACESTEP["poll_interval_s"])
                continue

            _raise_for_api_error(r, f"query_result({name})")
            data = _envelope(r.json())
            entry = (data[0] if isinstance(data, list) and data else data) or {}
            status = entry.get("status")
            if status == 1:                              # succeeded
                return entry
            if status == 2:                              # failed
                raise RuntimeError(f"ACE-Step generation FAILED for {name}: {_result_items(entry)}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"ACE-Step generation timed out for {name} "
                                  f"(task_id={task_id}, > {C.ACESTEP['poll_timeout_s']:.0f}s) — "
                                  f"raise config.ACESTEP['poll_timeout_s'] if your Mac is just slow")
            time.sleep(C.ACESTEP["poll_interval_s"])

    # ── dry-run mock synthesis ───────────────────────────────────────────────────
    def _generate_mock(self, spec: GenSpec, out: Path) -> None:
        sr = int(C.ACESTEP["sample_rate"])
        wav = _mock_waveform(spec, sr)
        _write_wav(out, wav, sr)
        logger.info("  [MOCK] synthesised: %s", out.name)


def check_server(url: str | None = None, timeout: float = 5.0) -> bool:
    """Fail-fast connectivity check for the ACE-Step API server. Call this at the
    top of a notebook run before kicking off a long batch. Confirms the target is
    actually the ACE-Step REST API (by finding /release_task in its OpenAPI schema),
    not e.g. the Gradio web UI on port 7860. Raises with actionable guidance if not."""
    import requests
    base = (url or C.ACESTEP["api_url"]).rstrip("/")
    _startup_help = (
        f"Start it in a SEPARATE terminal from your ACE-Step-1.5 clone:\n"
        f"  cd /path/to/ACE-Step-1.5 && ./start_api_server_macos.sh\n"
        f"  (or: uv run python -m acestep.api_server)\n"
        f"The REST API listens on port 8001 by default — NOT 7860, which is the "
        f"Gradio web UI and has no /release_task endpoint. Set ACESTEP_API_URL if "
        f"you changed the port."
    )
    try:
        r = requests.get(f"{base}/openapi.json", timeout=timeout)
    except requests.exceptions.RequestException:
        raise ConnectionError(f"ACE-Step API server not reachable at {base}.\n{_startup_help}")
    paths = {}
    try:
        paths = r.json().get("paths", {}) if r.status_code == 200 else {}
    except ValueError:
        pass
    if "/release_task" not in paths:
        raise ConnectionError(
            f"Something is listening at {base}, but it doesn't expose the ACE-Step "
            f"/release_task endpoint (this looks like the wrong service — the Gradio "
            f"UI on 7860 is the usual mix-up).\n{_startup_help}"
        )
    logger.info("ACE-Step API server confirmed at %s (%d routes; schema at %s/docs).",
                base, len(paths), base)
    return True


def list_routes(url: str | None = None, timeout: float = 5.0) -> list[str]:
    """Return the API server's route paths (from its OpenAPI schema). Handy for
    debugging if a call 404s — call this to see what the running server actually
    exposes, in case your installed version renamed an endpoint."""
    import requests
    base = (url or C.ACESTEP["api_url"]).rstrip("/")
    r = requests.get(f"{base}/openapi.json", timeout=timeout)
    r.raise_for_status()
    routes = sorted(r.json().get("paths", {}))
    for p in routes:
        logger.info("  route: %s", p)
    return routes


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _envelope(resp):
    """Unwrap the API's response envelope: {code, data, ...} -> data; else resp."""
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    return resp


def _result_items(entry: dict) -> list:
    """A query_result entry's `result` is a JSON-encoded STRING (a list of file
    dicts). Decode it defensively; return [] on anything unexpected."""
    raw = entry.get("result")
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])


def _extract_audio_path(entry: dict) -> str | None:
    """Pull the generated file's server-side path out of a succeeded query_result
    entry. Verified shape: result -> [{"file": "/abs/path....wav", ...}, ...]."""
    for item in _result_items(entry):
        if isinstance(item, dict):
            for key in ("file", "path", "audio_path"):
                if item.get(key):
                    return item[key]
        elif isinstance(item, str) and item:
            return item
    return None


def _path_to_url(path: str) -> str:
    """A raw server-side filesystem path -> the /v1/audio?path=... download route
    (verified in acestep/api/http/audio_route.py)."""
    if path.startswith("/v1/audio") or path.startswith("http"):
        return path
    return f"/v1/audio?path={urllib.parse.quote(path, safe='')}"


def _raise_for_api_error(r, context: str) -> None:
    if r.status_code >= 400:
        raise RuntimeError(f"ACE-Step API error during {context}: HTTP {r.status_code}\n{r.text[:2000]}")
    body = {}
    try:
        body = r.json()
    except ValueError:
        return
    if isinstance(body, dict) and body.get("code") not in (0, None, "0", 200):
        raise RuntimeError(f"ACE-Step API returned an error during {context}: {body}")


def _transcode_to_wav(src: Path, dst: Path, sr: int) -> None:
    """Decode whatever ACE-Step returned (likely mp3) and re-save as canonical WAV,
    so every clip on disk — real or mock — is the same format for downstream MuLan
    embedding regardless of generation mode."""
    import librosa
    wav, _ = librosa.load(str(src), sr=sr, mono=True)
    _write_wav(dst, wav.astype(np.float32), sr)


# ─── Emotion-dependent mock waveform ─────────────────────────────────────────────
# Not meant to sound like the target emotion to MuLan — only to (a) be a valid
# audio file and (b) vary systematically by emotion so downstream plumbing isn't
# degenerate. Valence -> major/minor + brightness; arousal -> tempo + noise.
_NOTE_HZ = {"C": 261.63, "D": 293.66, "E": 329.63, "F": 349.23,
            "G": 392.00, "A": 440.00, "B": 493.88}


def _mock_waveform(spec: GenSpec, sr: int) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    dur = float(spec.duration)
    t = np.arange(int(dur * sr)) / sr

    root = _NOTE_HZ.get((spec.keyscale[:1] or "A").upper(), 220.0) / 2.0  # down an octave
    major = "major" in spec.keyscale.lower()
    third = root * (2 ** (4 / 12)) if major else root * (2 ** (3 / 12))    # major/minor 3rd
    fifth = root * (2 ** (7 / 12))

    e = emo.get(spec.emotion) if spec.emotion in emo.ORDER else None
    valence = e.valence if e else 0.5
    arousal = e.arousal if e else 0.5

    # Chord: brighter (more upper harmonics) with higher valence.
    n_harm = 1 + int(round(3 * valence))
    chord = np.zeros_like(t)
    for f0 in (root, third, fifth):
        for h in range(1, n_harm + 1):
            chord += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    chord /= np.max(np.abs(chord) + 1e-9)

    # Rhythm: amplitude pulse at a tempo rising with arousal.
    bpm = float(spec.bpm) if spec.bpm else (60 + 120 * arousal)
    beat_hz = bpm / 60.0
    env = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * beat_hz * t), 0, 1) ** (1 + 2 * arousal)

    noise = arousal * 0.15 * rng.standard_normal(t.shape)
    wav = 0.8 * env * chord + noise
    # gentle fade to avoid clicks
    fade = int(0.05 * sr)
    if fade and wav.size > 2 * fade:
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
    return (wav / (np.max(np.abs(wav)) + 1e-9) * 0.9).astype(np.float32)


def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    try:
        import soundfile as sf
        sf.write(str(path), wav, sr)
    except Exception:                                              # noqa: BLE001
        # Fallback: 16-bit PCM WAV via the stdlib (no soundfile dependency).
        import wave
        pcm = np.clip(wav, -1, 1)
        pcm = (pcm * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
