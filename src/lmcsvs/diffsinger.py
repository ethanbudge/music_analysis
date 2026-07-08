"""
diffsinger.py — Headless SVS render via DiffSinger (open-source, no GUI).

This is the fully-scriptable alternative to importing MusicXML into Synthesizer V by
hand: our Score → a DiffSinger `.ds` score → DiffSinger inference → vocal wav, all
from Python. Exact lyrics + one fixed voicebank, reproducibly.

Two moving parts you provide once (see config.DIFFSINGER):
  • a DiffSinger **voicebank** (acoustic model + vocoder + phoneme dictionary), and
  • the DiffSinger (OpenVPI) inference code, in its own env.

Then `export_ds()` writes .ds files and `render()` shells out to the inference script.

IMPORTANT: the `.ds` schema and phoneme set are voicebank/version specific. The writer
below produces a standard acoustic-mode `.ds` (note_seq/note_dur + ph_seq/ph_dur), but
you should sanity-check one file against your voicebank's expected format (open it in
OpenUtau, or diff against a working .ds). Phonemes need a grapheme-to-phoneme step:
`g2p_en` (ARPABET) is used when installed; otherwise a crude fallback keeps the
pipeline running but will mispronounce — install g2p_en for real renders:

    pip install g2p_en
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

from . import config as C
from . import pipeline, score as score_mod

logger = logging.getLogger(__name__)

_g2p = None
_g2p_tried = False


def _get_g2p():
    global _g2p, _g2p_tried
    if not _g2p_tried:
        _g2p_tried = True
        try:
            from g2p_en import G2p
            _g2p = G2p()
        except Exception:                                          # noqa: BLE001
            _g2p = None
    return _g2p


def _phonemes(word: str) -> list[str]:
    """ARPABET phonemes (lowercased, stress stripped) for a word. g2p_en if present."""
    g = _get_g2p()
    if g is not None:
        return [p.rstrip("012").lower() for p in g(word) if p.strip() and p != " "]
    # crude fallback: letters as pseudo-phonemes (keeps timing valid; pronunciation poor)
    return [c for c in word.lower() if c.isalpha()] or ["sp"]


def _midi_to_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


_ARPA_VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy",
                "ow", "oy", "uh", "uw"}


def _is_vowel(ph: str) -> bool:
    return ph in _ARPA_VOWELS or ph[:1] in "aeiou"


def _word_groups(notes):
    """Group notes into words using the syllabic role (single/begin/…/end)."""
    groups, cur = [], []
    for n in notes:
        if n.syllabic in ("single", "begin") and cur:
            groups.append(cur); cur = []
        cur.append(n)
    if cur:
        groups.append(cur)
    return groups


def _split_phonemes(phs: list[str], k: int) -> list[list[str]]:
    """Split a word's phonemes into k note-chunks, anchoring one vowel per chunk."""
    if k <= 1:
        return [phs or ["sp"]]
    vowels = [i for i, p in enumerate(phs) if _is_vowel(p)]
    if len(vowels) >= k:
        # cut just before every vowel after the first, keep k chunks
        cuts = vowels[1:k]
        chunks, prev = [], 0
        for c in cuts:
            chunks.append(phs[prev:c]); prev = c
        chunks.append(phs[prev:])
        return [c or ["sp"] for c in chunks]
    # fewer vowels than notes → even split (some notes get a sustained consonant/vowel)
    out, per = [], max(1, len(phs) // k)
    for i in range(k):
        seg = phs[i * per:(i + 1) * per] if i < k - 1 else phs[i * per:]
        out.append(seg or [phs[-1] if phs else "sp"])
    return out


def to_ds(score) -> list[dict]:
    """Score → a one-segment DiffSinger acoustic `.ds` (list with a single dict).

    Emits the OpenUtau/OpenVPI acoustic fields: note_seq/note_dur/note_slur and the
    phoneme alignment ph_seq/ph_dur/ph_num (ph_num[i] = #phonemes on note i)."""
    sec16 = (60.0 / score.tempo_bpm) / 4.0            # seconds per sixteenth

    note_names, note_durs, note_slurs = [], [], []
    ph_seq, ph_dur, ph_num = [], [], []
    for group in _word_groups(score.notes):
        word = group[0].word
        phs = _phonemes(word)
        chunks = _split_phonemes(phs, len(group))
        for n, chunk in zip(group, chunks):
            dur = n.dur16 * sec16
            note_names.append(_midi_to_name(n.midi))
            note_durs.append(round(dur, 4))
            note_slurs.append(0)
            ph_num.append(len(chunk))
            weights = [2.0 if _is_vowel(p) else 1.0 for p in chunk]
            tot = sum(weights) or 1.0
            for p, w in zip(chunk, weights):
                ph_seq.append(p)
                ph_dur.append(round(dur * w / tot, 4))

    return [{
        "offset": 0.0,
        "text": " ".join(n.lyric for n in score.notes),
        "note_seq": " ".join(note_names),
        "note_dur": " ".join(str(d) for d in note_durs),
        "note_slur": " ".join(str(s) for s in note_slurs),
        "ph_seq": " ".join(ph_seq),
        "ph_dur": " ".join(str(d) for d in ph_dur),
        "ph_num": " ".join(str(x) for x in ph_num),
    }]


def export_ds(emotions: list[str] | None = None) -> dict:
    """Write a `.ds` for every grid cell. Returns {cell_name: path}."""
    C.ensure_dirs()
    C.DS_DIR.mkdir(parents=True, exist_ok=True)
    emotions = emotions or C.EMOTIONS
    out = {}
    if _get_g2p() is None:
        logger.warning("g2p_en not installed — .ds will use a crude phoneme fallback "
                       "(poor pronunciation). `pip install g2p_en` for real renders.")
    for L in emotions:
        for M in emotions:
            s = score_mod.assemble(L, M)
            name = pipeline.cell_name(L, M)
            path = C.DS_DIR / f"{name}.ds"
            path.write_text(json.dumps(to_ds(s), ensure_ascii=False, indent=1))
            out[name] = str(path)
    logger.info("Exported %d .ds scores → %s", len(out), C.DS_DIR)
    return out


def _check_configured() -> None:
    missing = [k for k in ("repo", "exp") if not C.DIFFSINGER[k]]
    if missing:
        raise RuntimeError(
            "DiffSinger not configured: set " + ", ".join(f"DIFFSINGER_{m.upper()}" for m in missing) +
            ".\nSetup (once): clone OpenVPI DiffSinger, `pip install -r requirements.txt` in "
            "its OWN env, download an ENGLISH acoustic voicebank into its checkpoints/ "
            "(e.g. Peiton), then set DIFFSINGER_REPO=<clone>, DIFFSINGER_EXP=<experiment "
            "name>, DIFFSINGER_SPK=<speaker if multi>, DIFFSINGER_PYTHON=<its python>. "
            "Also `pip install g2p_en` here for ARPABET phonemes. See the module docstring.")


def render(emotions: list[str] | None = None, force: bool = False) -> dict:
    """Render every exported .ds to data/svs/audio/<cell>.wav via DiffSinger inference.

    Shells out to OpenVPI DiffSinger's inference in its own env:
        python scripts/infer.py acoustic <cell>.ds --exp <EXP> [--spk <SPK>] --out <dir>
    (verified form; adjust here if your DiffSinger version differs). DiffSinger writes
    <ds-stem>.wav into --out, which matches our <cell>.wav naming.
    """
    import subprocess
    _check_configured()
    emotions = emotions or C.EMOTIONS
    paths = export_ds(emotions)
    C.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    for name, ds_path in paths.items():
        out_wav = C.AUDIO_DIR / f"{name}.wav"
        if out_wav.exists() and not force:
            done[name] = str(out_wav); continue
        cmd = [C.DIFFSINGER["python"], str(Path(C.DIFFSINGER["repo"]) / "scripts" / "infer.py"),
               "acoustic", ds_path, "--exp", C.DIFFSINGER["exp"], "--out", str(C.AUDIO_DIR)]
        if C.DIFFSINGER["spk"]:
            cmd += ["--spk", C.DIFFSINGER["spk"]]
        logger.info("render %s: %s", name, " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=C.DIFFSINGER["repo"])
        done[name] = str(out_wav)
    logger.info("Rendered %d clips → %s", len(done), C.AUDIO_DIR)
    return done
