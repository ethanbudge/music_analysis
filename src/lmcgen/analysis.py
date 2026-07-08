"""
analysis.py — Descriptive statistics + figures that answer the question:
"is lyric-music congruence actually being captured?"

Inputs are the tidy frames produced by pipeline.run():
  grid            one row per (lyric, music) cell x embedding model
  lyric_embed[mk] chorus-text-vs-anchor cosines (embedding evidence)
  lexical         chorus-vs-lexicon counts (model-independent evidence)

Everything takes an explicit `grid`/DataFrame so the notebook can call pieces in
any order. Figures are saved to config.FIGURE_DIR and also returned.
"""
from __future__ import annotations
import logging

import numpy as np

from . import config as C
from . import emotions as emo

logger = logging.getLogger(__name__)

ORDER = emo.ORDER


# ═══ Statistics ═══════════════════════════════════════════════════════════════════
def lmc_summary(grid):
    """Congruent (diagonal) vs incongruent (off-diagonal) realised LMC, per model.

    The core test: emotionally-matched lyric+music should embed more similarly
    (higher lmc_cross) than mismatched pairs. Reports means, the gap, Cohen's d and
    a Welch t-test.
    """
    import pandas as pd
    rows = []
    for mk, g in grid.groupby("model"):
        con = g.loc[g["congruent"], "lmc_cross"].to_numpy()
        inc = g.loc[~g["congruent"], "lmc_cross"].to_numpy()
        t, p = _welch_t(con, inc)
        rows.append({
            "model": mk,
            "congruent_mean": con.mean(), "congruent_sd": con.std(ddof=1),
            "incongruent_mean": inc.mean(), "incongruent_sd": inc.std(ddof=1),
            "gap": con.mean() - inc.mean(),
            "cohens_d": _cohens_d(con, inc),
            "welch_t": t, "p_value": p,
            "n_congruent": len(con), "n_incongruent": len(inc),
        })
    return pd.DataFrame(rows).set_index("model")


def manipulation_accuracy(grid):
    """Did the music land in its target emotion? Overall + per-emotion, per model."""
    import pandas as pd
    overall, per_emotion = [], []
    for mk, g in grid.groupby("model"):
        hit = (g["pred_music_emotion"] == g["music_emotion"])
        overall.append({"model": mk, "accuracy": hit.mean(), "n": len(g)})
        for M, gm in g.groupby("music_emotion"):
            per_emotion.append({
                "model": mk, "music_emotion": M,
                "accuracy": (gm["pred_music_emotion"] == M).mean(),
                "mean_anchor_cos": gm["music_anchor_cos"].mean(), "n": len(gm),
            })
    return pd.DataFrame(overall).set_index("model"), pd.DataFrame(per_emotion)


def congruence_gradient(grid, anchor_sim: dict):
    """Graded test: does realised LMC rise with *designed* congruence?

    For each cell, designed congruence = cosine(anchor_lyric, anchor_music) from the
    per-model anchor_similarity matrix. Correlates that with lmc_cross (Pearson &
    Spearman). A positive correlation means the continuous congruence signal — not
    just the diagonal — is recovered.
    """
    import pandas as pd
    rows = []
    for mk, g in grid.groupby("model"):
        sim = anchor_sim[mk]
        designed = np.array([sim.loc[r.lyric_emotion, r.music_emotion] for r in g.itertuples()])
        realised = g["lmc_cross"].to_numpy()
        rows.append({
            "model": mk,
            "pearson_r": _safe_corr(designed, realised, "pearson"),
            "spearman_r": _safe_corr(designed, realised, "spearman"),
            "n": len(g),
        })
    return pd.DataFrame(rows).set_index("model")


def wer_summary(grid):
    """Lyric-intelligibility (word error rate) summary, if WER screening was on.

    WER is a property of the clip (same across embedding models), so we dedupe to one
    model. Returns (overall dict, by_music_emotion df, by_lyric_emotion df) or None if
    no WER was recorded (screening off / dry-run).
    """
    import pandas as pd
    if "wer" not in grid.columns or grid["wer"].isna().all():
        return None
    g = grid[grid["model"] == grid["model"].iloc[0]].dropna(subset=["wer"])
    overall = {"median_wer": float(g["wer"].median()), "mean_wer": float(g["wer"].mean()),
               "worst_wer": float(g["wer"].max()), "n_cells": int(len(g)),
               "frac_over_0.30": float((g["wer"] > 0.30).mean())}
    by_music = g.groupby("music_emotion")["wer"].agg(["mean", "max"]).reindex(ORDER)
    by_lyric = g.groupby("lyric_emotion")["wer"].agg(["mean", "max"]).reindex(ORDER)
    return overall, by_music, by_lyric


def va_manipulation(vadf):
    """Did the music's ACOUSTICS track the intended valence/arousal? (The MuLan-anchor
    check couldn't see this within a fixed genre.) Correlates measured audio VA with
    the design targets across the grid. Positive r = the emotion manipulation landed."""
    import pandas as pd
    v = _safe_corr(vadf["audio_v"].to_numpy(), vadf["design_music_v"].to_numpy(), "pearson")
    a = _safe_corr(vadf["audio_a"].to_numpy(), vadf["design_music_a"].to_numpy(), "pearson")
    by = vadf.groupby("music_emotion")[["audio_v", "audio_a", "design_music_v",
                                        "design_music_a"]].mean().reindex(ORDER)
    return {"valence_r": v, "arousal_r": a}, by


def va_congruence_summary(vadf):
    """VA-space LMC: congruent (diagonal) vs incongruent audio↔lyric VA congruence,
    plus the graded correlation against the *designed* congruence."""
    con = vadf.loc[vadf["congruent"], "va_congruence"].to_numpy()
    inc = vadf.loc[~vadf["congruent"], "va_congruence"].to_numpy()
    t, p = _welch_t(con, inc)
    grad = _safe_corr(vadf["va_congruence_design"].to_numpy(),
                      vadf["va_congruence"].to_numpy(), "pearson")
    return {"congruent_mean": float(con.mean()), "incongruent_mean": float(inc.mean()),
            "gap": float(con.mean() - inc.mean()), "cohens_d": _cohens_d(con, inc),
            "welch_t": t, "p_value": p, "gradient_r": grad,
            "n_congruent": len(con), "n_incongruent": len(inc)}


def wer_summary_va(vadf):
    """WER summary from the Tier-1 rescore frame (single-hook WER + vocal-present)."""
    import pandas as pd
    g = vadf
    overall = {"median_wer": float(g["wer"].median()), "mean_wer": float(g["wer"].mean()),
               "frac_vocal_present": float(g["vocal_present"].mean()),
               "n_clean_wer<0.34": int((g["wer"] < 0.34).sum()),
               "n_clean_wer<0.5": int((g["wer"] < 0.5).sum()), "n": int(len(g))}
    by_music = g.groupby("music_emotion")[["wer", "vocal_present"]].mean().reindex(ORDER)
    return overall, by_music


def plot_audio_va(vadf, mock: bool = False):
    """Measured audio valence/arousal per clip (grey) vs the 8 design targets (colour).
    If the manipulation worked, clips cluster near their design point."""
    import matplotlib.pyplot as plt
    va_design = emo.valence_arousal()
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for name, (v, a) in va_design.items():
        ax.scatter(v, a, s=160, marker="*", zorder=3)
        ax.annotate(name, (v, a), fontsize=8, xytext=(4, 4), textcoords="offset points")
    for M, gm in vadf.groupby("music_emotion"):
        ax.scatter(gm["audio_v"], gm["audio_a"], s=22, alpha=0.5, zorder=1)
    ax.axhline(0.5, color="k", lw=0.5); ax.axvline(0.5, color="k", lw=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("valence"); ax.set_ylabel("arousal")
    ax.set_title(_t("Measured audio VA (dots) vs design targets (stars)", mock), fontsize=10)
    return _save(fig, "audio_va.png"), fig


def lmc_matrix(grid, model: str, value: str = "lmc_cross"):
    """Pivot the tidy grid to an 8x8 (lyric x music) matrix for one model."""
    g = grid[grid["model"] == model]
    return g.pivot(index="lyric_emotion", columns="music_emotion",
                   values=value).reindex(index=ORDER, columns=ORDER)


def confusion_matrix(grid, model: str):
    """8x8 true-music-emotion (rows) x predicted-emotion (cols) counts, one model."""
    import pandas as pd
    g = grid[grid["model"] == model]
    M = pd.crosstab(g["music_emotion"], g["pred_music_emotion"])
    return M.reindex(index=ORDER, columns=ORDER, fill_value=0)


# ── small stats helpers (scipy optional) ─────────────────────────────────────────
def _welch_t(a, b):
    try:
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except Exception:                                              # noqa: BLE001
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        t = (a.mean() - b.mean()) / np.sqrt(va + vb + 1e-12)
        return float(t), float("nan")


def _cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2) + 1e-12)
    return float((a.mean() - b.mean()) / sp)


def _safe_corr(x, y, kind):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    try:
        from scipy import stats
        f = stats.pearsonr if kind == "pearson" else stats.spearmanr
        return float(f(x, y)[0])
    except Exception:                                              # noqa: BLE001
        if kind == "spearman":
            x = _rank(x); y = _rank(y)
        return float(np.corrcoef(x, y)[0, 1])


def _rank(v):
    order = np.argsort(v); r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(v))
    return r


# ═══ Figures ══════════════════════════════════════════════════════════════════════
def _save(fig, name):
    C.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=140)
    logger.info("  figure: %s", path)
    return path


def _heatmap(ax, M, title, cmap="viridis", fmt="{:.2f}", diag=False, cbar=True):
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=8)
    ax.set_xlabel("music emotion"); ax.set_ylabel("lyric emotion")
    ax.set_title(title, fontsize=10)
    Mv = np.asarray(M, dtype=float)
    vmid = (np.nanmax(Mv) + np.nanmin(Mv)) / 2
    for i in range(Mv.shape[0]):
        for j in range(Mv.shape[1]):
            ax.text(j, i, fmt.format(Mv[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if Mv[i, j] < vmid else "black")
    if diag:
        for k in range(len(ORDER)):
            ax.add_patch(_rect(k))
    if cbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def _rect(k):
    import matplotlib.patches as mpatches
    return mpatches.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                              edgecolor="red", lw=2)


def plot_lmc_heatmap(grid, model: str, mock: bool = False):
    """8x8 realised-LMC heatmap; the red diagonal is the congruent (matched) cells."""
    import matplotlib.pyplot as plt
    M = lmc_matrix(grid, model)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _heatmap(ax, M.to_numpy(), _t(f"Realised LMC  cos(audio, lyric-text) — {model}", mock), diag=True)
    return _save(fig, f"lmc_heatmap_{model}.png"), fig


def plot_confusion(grid, model: str, mock: bool = False):
    """Music manipulation check: does each music emotion's audio land on its anchor?"""
    import matplotlib.pyplot as plt
    M = confusion_matrix(grid, model)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _heatmap(ax, M.to_numpy(), _t(f"Music manipulation check — {model}", mock),
             cmap="Blues", fmt="{:.0f}", diag=True)
    ax.set_xlabel("predicted (nearest anchor)"); ax.set_ylabel("target music emotion")
    return _save(fig, f"confusion_{model}.png"), fig


def plot_diagonal_vs_off(grid, mock: bool = False):
    """Congruent vs incongruent realised LMC, per model (the headline contrast)."""
    import matplotlib.pyplot as plt
    models = sorted(grid["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), squeeze=False)
    for ax, mk in zip(axes[0], models):
        g = grid[grid["model"] == mk]
        con = g.loc[g["congruent"], "lmc_cross"]; inc = g.loc[~g["congruent"], "lmc_cross"]
        ax.boxplot([con, inc], labels=["congruent\n(diagonal)", "incongruent\n(off-diag)"],
                   widths=0.6, showmeans=True)
        for xi, d in ((1, con), (2, inc)):
            ax.scatter(np.random.normal(xi, 0.05, len(d)), d, alpha=0.5, s=18, color="steelblue")
        ax.set_title(_t(f"{mk}: gap={con.mean() - inc.mean():+.3f}", mock), fontsize=10)
        ax.set_ylabel("cos(audio, lyric-text)")
    fig.suptitle(_t("Congruent lyric+music embed more similarly", mock), fontsize=11)
    fig.tight_layout()
    return _save(fig, "diagonal_vs_off.png"), fig


def plot_lyric_alignment(lyric_df, model: str, mock: bool = False):
    """Embedding evidence: each chorus's TEXT vs the 8 emotion anchors (argmax boxed)."""
    import matplotlib.pyplot as plt
    M = lyric_df[ORDER].reindex(ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M.to_numpy(), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=8)
    ax.set_xlabel("emotion anchor"); ax.set_ylabel("chorus (target emotion)")
    ax.set_title(_t(f"Lyric→anchor cosine ({model}) — diagonal = intended", mock), fontsize=10)
    for i, L in enumerate(ORDER):
        j = ORDER.index(lyric_df.loc[L, "predicted"])
        ax.add_patch(_rect_at(i, j))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, f"lyric_alignment_{model}.png"), fig


def plot_wer_heatmap(grid, mock: bool = False):
    """8x8 lyric word-error-rate heatmap (lyric × music); lower = clearer lyrics.
    Returns (path, fig) or None if WER screening wasn't run."""
    import matplotlib.pyplot as plt
    if "wer" not in grid.columns or grid["wer"].isna().all():
        return None
    g = grid[grid["model"] == grid["model"].iloc[0]]
    M = g.pivot(index="lyric_emotion", columns="music_emotion",
                values="wer").reindex(index=ORDER, columns=ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _heatmap(ax, M.to_numpy(), _t("Lyric intelligibility — word error rate (lower=clearer)", mock),
             cmap="RdYlGn_r", fmt="{:.2f}")
    return _save(fig, "wer_heatmap.png"), fig


def plot_valence_arousal(grid=None, model: str = "mulan", mock: bool = False):
    """The design's circumplex map; if `grid` given, overlay each clip's estimated V/A
    (anchor-cosine-weighted average of the 8 emotions' coordinates)."""
    import matplotlib.pyplot as plt
    va = emo.valence_arousal()
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    for name, (v, a) in va.items():
        ax.scatter(v, a, s=140, zorder=3)
        ax.annotate(name, (v, a), fontsize=8, xytext=(4, 4), textcoords="offset points")
    if grid is not None:
        g = grid[grid["model"] == model]
        V = np.array([va[e][0] for e in ORDER]); A = np.array([va[e][1] for e in ORDER])
        for r in g.itertuples():
            w = np.array([getattr(r, f"anchor_cos__{e}") for e in ORDER])
            w = np.exp(6 * (w - w.max())); w /= w.sum()
            ax.scatter((w * V).sum(), (w * A).sum(), s=12, alpha=0.35, color="grey", zorder=1)
    ax.axhline(0.5, color="k", lw=0.5); ax.axvline(0.5, color="k", lw=0.5)
    ax.set_xlabel("valence  (negative → positive)"); ax.set_ylabel("arousal  (calm → excited)")
    ax.set_title(_t("Emotion design on the affect circumplex", mock), fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return _save(fig, "valence_arousal.png"), fig


def _rect_at(i, j):
    import matplotlib.patches as mpatches
    return mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", lw=2)


def _t(title, mock):
    return ("[MOCK] " + title) if mock else title
