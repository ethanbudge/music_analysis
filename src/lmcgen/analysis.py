"""
analysis.py — Descriptive statistics, figures, winner selection + survey export.

Consumes results/generation/songs.csv (validate.validate_all) and turns it into:

  lyric_placement        the 16 couplets' lexical + VA placement (do they land in-corner)
  congruence_matrix      4x4 lyric-corner x music-corner mean LMC / VA-congruence
  manipulation_accuracy  did the music land in its target corner? (MuLan argmax + VA-nearest)
  lmc_summary            realised LMC on congruent vs incongruent cells
  wer_summary            lyric intelligibility (median / by cell)
  winner_selection       best rep per cell for the survey (on-target + intelligible)
  export_survey          copy the winners to data/generation/survey/, loudness-normalise,
                         write a manifest for the respondent study
  plots                  VA scatter, congruence heatmap, MuLan confusion
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

from . import config as C
from . import quadrants as q
from . import lyrics as lyr

logger = logging.getLogger(__name__)


# ─── lyric-side placement ────────────────────────────────────────────────────────
def lyric_placement():
    """Merge the lexical and VA placement checks for the 16 couplets."""
    import pandas as pd
    lex = lyr.lexical_alignment()[["quadrant", "predicted", "correct"]].rename(
        columns={"predicted": "lex_pred", "correct": "lex_correct"})
    vaa = lyr.va_alignment()[["lyric_v", "lyric_a", "predicted", "correct"]].rename(
        columns={"predicted": "va_pred", "correct": "va_correct"})
    return lex.join(vaa)


# ─── congruence matrix ───────────────────────────────────────────────────────────
def congruence_matrix(df, value: str = "mulan_lmc_cross"):
    """4x4 mean of `value` by (lyric_quadrant rows) x (music_quadrant cols)."""
    import pandas as pd
    m = df.pivot_table(index="lyric_quadrant", columns="music_quadrant",
                       values=value, aggfunc="mean")
    return m.reindex(index=q.ORDER, columns=q.ORDER)


def lmc_summary(df, value: str = "mulan_lmc_cross"):
    """Realised LMC on congruent (diagonal) vs incongruent (off-diagonal) cells."""
    import pandas as pd
    g = df.groupby("congruent")[value].agg(["mean", "std", "count"])
    diag = df[df.congruent][value].mean()
    off = df[~df.congruent][value].mean()
    g.loc["gap (congruent − incongruent)"] = [diag - off, np.nan, len(df)]
    return g


# ─── manipulation checks (did the MUSIC land in its target corner?) ──────────────
def manipulation_accuracy(df):
    """Fraction of songs whose predicted music corner == the target, by instrument."""
    import pandas as pd
    out = {}
    if "mulan_pred_quadrant" in df:
        out["mulan_anchor"] = (df["mulan_pred_quadrant"] == df["music_quadrant"]).mean()
    if "va_pred_quadrant" in df and df["va_pred_quadrant"].notna().any():
        sub = df[df["va_pred_quadrant"].notna()]
        out["acoustic_va"] = (sub["va_pred_quadrant"] == sub["music_quadrant"]).mean()
    return pd.Series(out, name="accuracy_vs_target")


def confusion(df, pred_col: str = "mulan_pred_quadrant"):
    """Confusion matrix: rows = target music corner, cols = predicted corner."""
    import pandas as pd
    sub = df[df[pred_col].notna()]
    m = pd.crosstab(sub["music_quadrant"], sub[pred_col])
    return m.reindex(index=q.ORDER, columns=q.ORDER).fillna(0).astype(int)


# ─── lyric intelligibility ───────────────────────────────────────────────────────
def wer_summary(df):
    """WER overall + by music corner. Returns None if WER wasn't measured (dry-run)."""
    import pandas as pd
    if "wer" not in df or df["wer"].isna().all():
        return None
    overall = df["wer"].median()
    by_music = df.groupby("music_quadrant")["wer"].median().reindex(q.ORDER)
    vocal = df["vocal_present"].mean() if "vocal_present" in df else np.nan
    return {"median_wer": overall, "pct_vocal_present": vocal,
            "median_wer_by_music": by_music}


# ─── winner selection for the survey ─────────────────────────────────────────────
def winner_selection(df, per_cell: int | None = None):
    """Rank the reps within each (lyric_id, music_quadrant) cell and keep the best
    `per_cell` for the survey. Lower score = better: on-target (low VA distance, high
    MuLan target-anchor cosine) and intelligible (low WER). Components missing (dry-run
    WER) are skipped. Returns the winning rows with a 'rank' and 'winner_score'."""
    import pandas as pd
    per_cell = C.SURVEY["per_cell"] if per_cell is None else per_cell
    d = df.copy()

    def _z(col):
        if col not in d or d[col].isna().all():
            return pd.Series(0.0, index=d.index)
        s = d[col].astype(float)
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd else pd.Series(0.0, index=d.index)

    # lower is better
    score = _z("va_dist_to_target") - _z("mulan_target_anchor_cos") + _z("wer")
    d["winner_score"] = score
    d["rank"] = d.groupby(["lyric_id", "music_quadrant"])["winner_score"] \
                 .rank(method="first")
    winners = d[d["rank"] <= per_cell].sort_values(
        ["lyric_id", "music_quadrant", "rank"]).reset_index(drop=True)
    return winners


# ─── survey export (loudness-normalised stimuli + manifest) ──────────────────────
def export_survey(winners, target_lufs: float | None = None):
    """Copy the winning clips into data/generation/survey/, loudness-normalise each to
    ~target LUFS (pyloudnorm if available, else copied unchanged), and write manifest.csv.
    Returns the manifest DataFrame."""
    import shutil
    import pandas as pd
    C.ensure_dirs()
    target_lufs = C.SURVEY["target_lufs"] if target_lufs is None else target_lufs
    rows = []
    for _, r in winners.iterrows():
        src = Path(r["path"])
        dst = C.SURVEY_DIR / f"{r['lyric_id']}__{r['music_quadrant']}.wav"
        norm = _loudness_normalise(src, dst, target_lufs)
        rows.append({
            "stimulus": dst.name, "path": str(dst),
            "lyric_id": r["lyric_id"], "lyric_quadrant": r["lyric_quadrant"],
            "music_quadrant": r["music_quadrant"], "congruent": r["congruent"],
            "lyric_text": lyr.get(r["lyric_id"]).plain,
            "loudness_normalised": norm,
            "wer": r.get("wer"),
            "mulan_target_anchor_cos": r.get("mulan_target_anchor_cos"),
            "audio_v": r.get("audio_v"), "audio_a": r.get("audio_a"),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(C.SURVEY_DIR / "manifest.csv", index=False)
    logger.info("Exported %d survey stimuli → %s", len(manifest), C.SURVEY_DIR)
    return manifest


def _loudness_normalise(src: Path, dst: Path, target_lufs: float) -> bool:
    import shutil
    try:
        import soundfile as sf
        import pyloudnorm as pyln
        data, sr = sf.read(str(src))
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(data)
        out = pyln.normalize.loudness(data, loudness, target_lufs)
        sf.write(str(dst), out, sr)
        return True
    except Exception as e:                                          # noqa: BLE001
        logger.warning("  loudness normalise unavailable (%s); copying %s as-is "
                       "(pip install pyloudnorm soundfile)", e, src.name)
        shutil.copyfile(src, dst)
        return False


# ─── figures ─────────────────────────────────────────────────────────────────────
def _save(fig, name: str) -> Path:
    C.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURE_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    logger.info("  wrote %s", path)
    return path


def plot_va_scatter(df, save: bool = True):
    """Acoustic valence/arousal of every song, coloured by target music corner, with
    the four numeric targets marked — shows whether the corners separate."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = dict(zip(q.ORDER, ["#e4572e", "#2e9e5b", "#8a2be2", "#3b6ea5"]))
    for code in q.ORDER:
        sub = df[df.music_quadrant == code]
        ax.scatter(sub["audio_v"], sub["audio_a"], s=22, alpha=0.6,
                   color=colors[code], label=code)
        quad = q.get(code)
        ax.scatter([quad.valence], [quad.arousal], marker="*", s=320,
                   edgecolor="k", color=colors[code], zorder=5)
    ax.set_xlabel("valence (acoustic)"); ax.set_ylabel("arousal (acoustic)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="0.8", lw=0.8); ax.axvline(0.5, color="0.8", lw=0.8)
    ax.set_title("Song acoustic VA by target corner (★ = target)")
    ax.legend(title="music corner", loc="best", fontsize=8)
    fig.tight_layout()
    return _save(fig, "va_scatter.png") if save else fig


def plot_congruence_heatmap(df, value: str = "mulan_lmc_cross", save: bool = True):
    import matplotlib.pyplot as plt
    m = congruence_matrix(df, value)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(m.values, cmap="viridis")
    ax.set_xticks(range(len(q.ORDER))); ax.set_xticklabels(q.ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(q.ORDER))); ax.set_yticklabels(q.ORDER)
    ax.set_xlabel("music corner"); ax.set_ylabel("lyric corner")
    for i in range(len(q.ORDER)):
        for j in range(len(q.ORDER)):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    ax.set_title(f"Mean {value}\n(diagonal = congruent)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return _save(fig, f"congruence_{value}.png") if save else fig


def plot_confusion(df, pred_col: str = "mulan_pred_quadrant", save: bool = True):
    import matplotlib.pyplot as plt
    m = confusion(df, pred_col)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(m.values, cmap="Blues")
    ax.set_xticks(range(len(q.ORDER))); ax.set_xticklabels(q.ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(q.ORDER))); ax.set_yticklabels(q.ORDER)
    ax.set_xlabel("predicted corner"); ax.set_ylabel("target music corner")
    for i in range(len(q.ORDER)):
        for j in range(len(q.ORDER)):
            ax.text(j, i, int(m.values[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title(f"Music-corner confusion ({pred_col})")
    fig.tight_layout()
    return _save(fig, f"confusion_{pred_col}.png") if save else fig
