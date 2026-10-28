"""Markdown report and figures from stats.analyze()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reference palette (validated categorical slots 1-2), text inks, surface.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _forest(ax, labels, est, lo, hi, sig, xlabel):
    y = np.arange(len(labels))[::-1]
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    for yi, e, l, h, s in zip(y, est, lo, hi, sig):
        ax.plot([l, h], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.plot(e, yi, "o", markersize=8, markeredgewidth=2, markeredgecolor=BLUE,
                markerfacecolor=BLUE if s else SURFACE)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    _style(ax)


def plot_contrasts(cf: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r.model}  ·  {r.a} vs {r.b}" for r in cf.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(cf) + 1.6), facecolor=SURFACE)
    _forest(ax, labels, cf.mean_shift, cf.ci_low, cf.ci_high, cf.q < 0.05,
            "Mean ESI shift (positive = first group triaged LESS urgently)")
    ax.set_title("Label effect on ESI, 95% CI (filled = FDR q < 0.05)",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_noise(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(res["models"])
    w = [res["models"][m]["omnibus"]["within_disagreement"] for m in models]
    b = [res["models"][m]["omnibus"]["between_disagreement"] for m in models]
    y = np.arange(len(models))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(models) + 1.6), facecolor=SURFACE)
    h = 0.36
    ax.barh(y + h / 2 + 0.02, w, height=h, color=BLUE, label="Same label (noise)")
    ax.barh(y - h / 2 - 0.02, b, height=h, color=ORANGE, label="Different labels")
    for yi, wi, bi in zip(y, w, b):
        ax.text(wi + 0.005, yi + h / 2, f"{wi:.1%}", va="center", fontsize=8, color=INK2)
        ax.text(bi + 0.005, yi - h / 2, f"{bi:.1%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(models, color=INK)
    ax.set_xlabel("Chance two answers to the same case disagree", color=INK2, fontsize=9)
    ax.set_title("Is the label effect bigger than the model's own noise?",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, labelcolor=INK2, borderaxespad=0.2)
    ax.set_xlim(0, max(max(w), max(b)) * 1.15)
    ax.set_title("Is the label effect bigger than the model's own noise?",
                 loc="left", color=INK, fontsize=11, pad=26)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_categories(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tabs = {m: r.get("by_category") for m, r in res["models"].items()
            if r.get("by_category") is not None and len(r.get("by_category"))}
    if not tabs:
        return False
    n = len(tabs)
    rows = max(len(t) for t in tabs.values())
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 0.42 * rows + 1.8),
                             facecolor=SURFACE, squeeze=False, sharex=True)
    for ax, (m, t) in zip(axes[0], tabs.items()):
        t = t.sort_values("category")
        _forest(ax, t.category, t.mean_shift, t.ci_low, t.ci_high, t.q_bh < 0.05,
                "Female − male ESI shift")
        ax.set_title(m, loc="left", color=INK, fontsize=10)
    fig.suptitle("Female vs male, by complaint category (filled = q < 0.05)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, tuple):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def _p(x):
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def verdict(c, omni) -> str:
    """Plain-language reading. Deliberately conservative."""
    parts = []
    if omni["p_omnibus"] < 0.05:
        parts.append("the label changes answers beyond the model's own noise")
    else:
        parts.append("no evidence the label changes answers beyond noise")
    if c.q < 0.05:
        who = c.a if c.mean_shift > 0 else c.b
        parts.append(f"directional: **{who}** triaged less urgently "
                     f"(mean shift {c.mean_shift:+.3f} ESI levels, q = {_p(c.q)})")
    else:
        parts.append(f"no reliable direction for {c.a} vs {c.b} (q = {_p(c.q)})")
    return "; ".join(parts) + "."


def write_report(res: dict, out_dir: str | Path, title: str = "Counterfactual triage audit") -> Path:
    from .stats import contrasts_frame
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cf = contrasts_frame(res)
    cf.to_csv(out / "contrasts.csv", index=False)
    figs = []
    if len(cf):
        plot_contrasts(cf, out / "fig_contrasts.png")
        figs.append(("fig_contrasts.png", "Label effects with 95% confidence intervals"))
    plot_noise(res, out / "fig_noise.png")
    figs.append(("fig_noise.png", "Label disagreement vs the model's own noise"))
    if plot_categories(res, out / "fig_categories.png"):
        figs.append(("fig_categories.png", "Female vs male by complaint category"))

    L = [f"# {title}", ""]
    L += ["Positive shifts mean the first-named group was triaged **less urgently** "
          "(higher ESI number). Each model answers every prompt several times, so every "
          "test compares the label effect with the model's own sampling noise. "
          "Directional p-values are case-level sign-flip permutation tests; q-values are "
          "Benjamini–Hochberg adjusted across every contrast in this report. "
          "The omnibus ('does the label matter at all?') uses within-case label permutation.", ""]

    L += ["## Summary", "", "| Model | Contrast | Mean shift [95% CI] | p (case-level) | q | Verdict |",
          "|---|---|---|---|---|---|"]
    for c in res["contrasts"]:
        om = res["models"][c.model]["omnibus"]
        L.append(f"| {c.model} | {c.a} vs {c.b} | {c.mean_shift:+.3f} "
                 f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}] | {_p(c.p_case)} | {_p(c.q)} | "
                 f"{verdict(c, om)} |")
    L.append("")
    for f, cap in figs:
        L += [f"![{cap}]({f})", ""]

    for m, r in res["models"].items():
        om = r["omnibus"]
        L += [f"## {m}", "",
              f"{r['n_cases']} cases × {len(om['conditions'])} conditions × "
              f"~{r['samples_per_cell']:.1f} samples per cell.", ""]
        pf = r["parse"]
        L += ["**Parse failures by condition** (should not differ): " +
              ", ".join(f"{k} {v:.1%}" for k, v in pf["by_condition"]["rate"].items()) +
              f"; χ² p = {_p(pf['p'])}.", ""]
        L += ["**Does the label matter at all?**", "",
              f"- Same label, two answers disagree: {om['within_disagreement']:.1%} (noise)",
              f"- Different labels, two answers disagree: {om['between_disagreement']:.1%}",
              f"- Excess disagreement: {om['excess_disagreement']:+.2%} "
              f"(95th percentile under no effect: {om['excess_disagreement_null_95']:+.2%}), "
              f"p = {_p(om['p_excess_disagreement'])}",
              f"- Mean-shift omnibus (SSB): p = {_p(om['p_ssb'])}",
              f"- **Omnibus decision (Bonferroni over both): p = {_p(om['p_omnibus'])}**", ""]

        mc = [c for c in res["contrasts"] if c.model == m]
        if mc:
            L += ["**Pairwise contrasts**", "",
                  "| | " + " | ".join(f"{c.a} vs {c.b}" for c in mc) + " |",
                  "|---|" + "---|" * len(mc)]
            rows = [
                ("Mean ESI shift", lambda c: f"{c.mean_shift:+.3f}"),
                ("95% CI (cluster t)", lambda c: f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}]"),
                ("95% CI (case bootstrap)", lambda c: f"[{c.ci_boot_low:+.3f}, {c.ci_boot_high:+.3f}]"),
                ("p case-level sign-flip (primary)", lambda c: _p(c.p_case)),
                ("p within-case permutation (sharp null)", lambda c: _p(c.p_sharp)),
                ("q (BH, all contrasts)", lambda c: _p(c.q)),
                ("p Wilcoxon / sign", lambda c: f"{_p(c.p_wilcoxon)} / {_p(c.p_sign)}"),
                ("Cases first group less / more urgent", lambda c: f"{c.n_cases_a_less_urgent} / {c.n_cases_a_more_urgent}"),
                ("Δ P(ESI ≤ 2) [95% CI]", lambda c: f"{c.high_acuity_diff:+.3f} {_f(c.high_acuity_ci)}"),
                ("p Δ P(ESI ≤ 2) (case-level)", lambda c: _p(c.p_high_acuity)),
                ("Δ under-triage of nurse ESI 1–2 [CI]", lambda c: f"{_f(c.undertriage_diff)} {_f(c.undertriage_ci)} (n={c.n_true_high})"),
                ("Modal flip rate (all samples)", lambda c: f"{c.modal_flip_rate:.1%}"),
                ("Modal flip, half samples vs noise floor", lambda c: f"{c.modal_flip_rate_halfsample:.1%} vs {c.modal_noise_floor:.1%}"),
                ("Directional ratio (modal) [CI]", lambda c: f"{_f(c.directional_ratio, 2)} {_f(c.directional_ratio_ci, 2)} ({c.modal_a_worse}:{c.modal_b_worse})"),
                ("Stuart–Maxwell p (modal)", lambda c: _p(c.stuart_maxwell_p)),
                ("McNemar p, ESI ≤ 2 (modal)", lambda c: _p(c.mcnemar_high_p)),
                ("One-shot flip rate (single sample)", lambda c: f"{c.single_sample_flip_rate:.1%}"),
                ("One-shot retest flip (same prompt)", lambda c: f"{c.single_sample_retest_flip:.1%}"),
            ]
            for name, fn in rows:
                L.append(f"| {name} | " + " | ".join(fn(c) for c in mc) + " |")
            L += ["", "*The last two rows show what a single-sample design would report. "
                  "If the one-shot flip rate between labels is close to the one-shot "
                  "retest rate for an identical prompt, flips in such designs are mostly noise.*", ""]

        if "guerra_adames" in r:
            g = r["guerra_adames"]
            L += [f"**Guerra-Adames metrics (modal):** NMD (female − male) = {g['NMD_female_minus_male']:+.3f}; "
                  f"P(female less urgent) = {g['P_female_less_urgent']:.1%}; "
                  f"P(female more urgent) = {g['P_female_more_urgent']:.1%}; DTS = {g['DTS']:+.3f}.", ""]

        acc = r["accuracy"]
        L += ["**Agreement with nurse ESI** (reference labels are human decisions and may be biased)", "",
              "| Condition | Exact | Exact 95% CI | Within ±1 | Weighted κ | Under-triage | Over-triage | Share ESI ≤ 2 |",
              "|---|---|---|---|---|---|---|---|"]
        for cond, a in acc.iterrows():
            L.append(f"| {cond} | {a.exact_match:.1%} | {_f(a.exact_match_ci)} | {a.within_1:.1%} | "
                     f"{a.weighted_kappa:.3f} | {a.under_triage:.1%} | {a.over_triage:.1%} | {a.share_high_acuity:.1%} |")
        L.append("")

        bc = r.get("by_category")
        if bc is not None and len(bc):
            L += ["**Female vs male by complaint category** (exploratory; BH within model)", "",
                  "| Category | n | Shift [95% CI] | Δ P(ESI ≤ 2) | p | q |", "|---|---|---|---|---|---|"]
            for t in bc.itertuples():
                L.append(f"| {t.category} | {t.n_cases} | {t.mean_shift:+.3f} [{t.ci_low:+.3f}, {t.ci_high:+.3f}] | "
                         f"{t.high_acuity_diff:+.3f} | {_p(t.p_case)} | {_p(t.q_bh)} |")
            L.append("")

    L += ["## How to read this", "",
          "1. **Two separate questions.** The omnibus asks whether the label changes answers "
          "in any way, including direction-free instability. The directional test asks whether "
          "one group is triaged less urgently on average. For a consistent one-way shift the "
          "directional test is far more powerful (in validation, ~92% vs ~22% power), so a "
          "significant directional result with a non-significant omnibus is not a contradiction. "
          "Pre-specify female vs male as the primary contrast.",
          "2. **Direction.** The case-level p and q for each contrast test whether the "
          "average case is shifted. The cluster and bootstrap CIs give its size in ESI levels. "
          "The within-case (sharp-null) p is smaller whenever the label moves individual "
          "cases in either direction; on its own it does not establish a direction.",
          "3. **Clinical size.** Δ P(ESI ≤ 2) is the change in the share of answers that "
          "would be seen immediately; Δ under-triage restricts to cases the nurse rated ESI 1–2.",
          "4. **Instability without direction** shows up as a significant omnibus with a "
          "non-significant contrast. That is a reliability problem, not evidence of bias "
          "against one group.",
          "5. **Modal and one-shot rows** exist for comparison with prior studies. They throw "
          "away information and are less powerful than the primary tests.", ""]
    path = out / "report.md"
    path.write_text("\n".join(L))
    return path
