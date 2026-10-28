"""Statistics for counterfactual demographic tests of LLM triage.

Data model
----------
``y[c, j, s]`` is the ESI a model returned for source case ``c`` under
condition ``j`` (e.g. male / female / unlabeled) on sample ``s``. Cases are
the unit of independent replication; samples are repeated draws of the same
prompt and measure the model's own noise.

The central question is whether the condition label moves the answer by
more than the model already moves when nothing changes. Every test below is
built on that comparison.

Methods and sources
-------------------
* Within-case permutation tests (exact under the null that the label is
  irrelevant, because samples of a case are then exchangeable across
  conditions). This is the primary test: it handles noise, ties, ordinal
  outcomes and unequal sample counts without distributional assumptions.
* Case-clustered standard errors and case-level bootstrap, following Miller
  (2024), "Adding Error Bars to Evals", arXiv:2411.00640.
* Wilcoxon signed-rank and sign tests on per-case mean shifts.
* Stuart-Maxwell marginal-homogeneity test and exact McNemar on modal
  answers, for comparability with prior work.
* Counterfactual flip rate and directional F/M ratio (EQUITRIAGE,
  arXiv:2605.03998), reported alongside a within-condition noise floor.
* Net Mean Disadvantage and Directional Triage Skew (Guerra-Adames et al.,
  arXiv:2511.17124).
* Benjamini-Hochberg false discovery rate control across contrasts.
* Quadratic-weighted kappa against nurse-assigned ESI (with the caveat that
  the reference labels are themselves human decisions).

Sign convention: higher ESI = less urgent. A positive shift for contrast
"female - male" means the female-labelled version was triaged as LESS urgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

ESI_LEVELS = np.arange(1, 6)


# =============================================================== preparation

@dataclass
class Prepared:
    model: str
    df: pd.DataFrame              # valid samples: case_id, condition, sample, esi
    cases: pd.DataFrame           # case table (esi_true, category, ...)
    conditions: list[str]
    parse_failures: pd.DataFrame  # per-condition failure counts
    case_index: pd.Index = field(default=None)


def prepare(results: pd.DataFrame, cases: pd.DataFrame, model: str,
            conditions: list[str] | None = None) -> Prepared:
    r = results[results["model"] == model].copy()
    r["case_id"] = r["case_id"].astype(str)
    conditions = conditions or sorted(r["condition"].unique())
    r = r[r["condition"].isin(conditions)]
    failed = r["esi"].isna() | r.get("error", pd.Series(None, index=r.index)).notna()
    pf = (r.assign(failed=failed).groupby("condition")["failed"]
          .agg(n_calls="size", n_failed="sum").reindex(conditions).fillna(0))
    valid = r[~failed].copy()
    valid["esi"] = valid["esi"].astype(int)
    # Keep only cases observed in every condition (paired design).
    have = valid.groupby("case_id")["condition"].nunique()
    keep = have[have == len(conditions)].index
    valid = valid[valid["case_id"].isin(keep)]
    cs = cases.copy()
    cs["case_id"] = cs["case_id"].astype(str)
    cs = cs[cs["case_id"].isin(keep)].set_index("case_id")
    return Prepared(model, valid[["case_id", "condition", "sample", "esi"]],
                    cs, conditions, pf, pd.Index(sorted(keep)))


def parse_failure_test(p: Prepared) -> dict:
    """Do parse failures differ by condition? (They should not; if they do,
    dropping failures is itself a biased filter.)"""
    pf = p.parse_failures
    table = np.vstack([pf["n_failed"].values, (pf["n_calls"] - pf["n_failed"]).values])
    out = {"by_condition": pf.assign(rate=pf["n_failed"] / pf["n_calls"].clip(lower=1))}
    if table[0].sum() > 0 and table.shape[1] > 1:
        chi2, pval, _, _ = sps.chi2_contingency(table)
        out.update(chi2=chi2, p=pval)
    else:
        out.update(chi2=0.0, p=1.0)
    return out


# ============================================================ cell summaries

def cell_table(p: Prepared) -> pd.DataFrame:
    """Per (case, condition): n, mean ESI, P(ESI<=2), modal ESI, disagreement."""
    g = p.df.groupby(["case_id", "condition"])["esi"]

    def mode(x):
        c = _bincount(x, minlength=6)
        return int(np.argmax(c))  # ties -> lower ESI (more urgent), deterministic

    def within_disagree(x):
        n = len(x)
        if n < 2:
            return np.nan
        c = _bincount(x, minlength=6)
        same = (c * (c - 1)).sum()
        return 1 - same / (n * (n - 1))

    t = g.agg(n="size", mean="mean", mode=mode,
              p_high=lambda x: (x <= 2).mean(),
              w_disagree=within_disagree)
    return t.reset_index()


# ======================================================== permutation engine

def _bincount(x, **kw):
    # np.bincount needs the platform integer; pandas gives int64, which
    # 32-bit builds (e.g. Pyodide in the browser) refuse to cast.
    return np.bincount(np.asarray(x, dtype=np.intp), **kw)


def _encode(p: Prepared, conds: list[str]):
    d = p.df[p.df["condition"].isin(conds)]
    case_codes, case_uniques = pd.factorize(d["case_id"], sort=True)
    cond_map = {c: i for i, c in enumerate(conds)}
    lab = d["condition"].map(cond_map).to_numpy()
    y = d["esi"].to_numpy().astype(float)
    order = np.argsort(case_codes, kind="stable")
    return case_codes[order], lab[order], y[order], len(case_uniques)


def _group_means(case, lab, y, n_case, n_cond):
    idx = case * n_cond + lab
    s = _bincount(idx, weights=y, minlength=n_case * n_cond)
    n = _bincount(idx, minlength=n_case * n_cond)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / n).reshape(n_case, n_cond), n.reshape(n_case, n_cond)


def _permute_within(case, lab, rng):
    """Shuffle labels within each case (case array must be sorted)."""
    key = case + rng.random(len(case))
    return lab[np.argsort(key, kind="stable")]


def _ssb(case, lab, y, n_case, n_cond):
    m, n = _group_means(case, lab, y, n_case, n_cond)
    cm = np.nansum(m * n, axis=1) / n.sum(axis=1)
    return float(np.nansum(n * (m - cm[:, None]) ** 2))


def _excess_disagreement(case, lab, y, n_case, n_cond):
    """Between-condition minus within-condition pairwise disagreement,
    averaged over cases. ~0 when the label is irrelevant."""
    yi = y.astype(int)
    idx_cell = case * n_cond + lab
    counts = np.zeros((n_case * n_cond, 6))
    np.add.at(counts, (idx_cell, yi), 1)
    counts = counts.reshape(n_case, n_cond, 6)
    n = counts.sum(axis=2)
    within_same = (counts * (counts - 1)).sum(axis=2)
    within_pairs = n * (n - 1)
    w = 1 - within_same.sum(axis=1) / np.maximum(within_pairs.sum(axis=1), 1)
    btw_same = np.zeros(n_case)
    btw_pairs = np.zeros(n_case)
    for a, b in combinations(range(n_cond), 2):
        btw_same += (counts[:, a, :] * counts[:, b, :]).sum(axis=1)
        btw_pairs += n[:, a] * n[:, b]
    b_ = 1 - btw_same / np.maximum(btw_pairs, 1)
    return float(np.mean(b_ - w)), float(np.mean(b_)), float(np.mean(w))


def omnibus_test(p: Prepared, conds: list[str] | None = None,
                 n_perm: int = 2000, seed: int = 0) -> dict:
    """Does the demographic label change the answer distribution at all?

    Two statistics, each with an exact within-case permutation p-value:
    * SSB: between-condition sum of squares of mean ESI within cases
      (sensitive to systematic shifts).
    * Excess disagreement: how much more often two answers disagree when
      they come from different labels than from the same label
      (sensitive to any change, including direction-free instability).
    """
    conds = conds or p.conditions
    case, lab, y, nc = _encode(p, conds)
    k = len(conds)
    rng = np.random.default_rng(seed)
    obs_ssb = _ssb(case, lab, y, nc, k)
    obs_ex, between, within = _excess_disagreement(case, lab, y, nc, k)
    ge_ssb = ge_ex = 0
    null_ex = np.empty(n_perm)
    for i in range(n_perm):
        pl = _permute_within(case, lab, rng)
        ge_ssb += _ssb(case, pl, y, nc, k) >= obs_ssb - 1e-12
        e = _excess_disagreement(case, pl, y, nc, k)[0]
        null_ex[i] = e
        ge_ex += e >= obs_ex - 1e-12
    return {
        "conditions": conds, "n_cases": nc,
        "within_disagreement": within,
        "between_disagreement": between,
        "excess_disagreement": obs_ex,
        "excess_disagreement_null_95": float(np.quantile(null_ex, 0.95)),
        "p_excess_disagreement": (ge_ex + 1) / (n_perm + 1),
        "ssb": obs_ssb,
        "p_ssb": (ge_ssb + 1) / (n_perm + 1),
        # Bonferroni over the two statistics: the omnibus decision uses this.
        "p_omnibus": min(1.0, 2 * min((ge_ex + 1) / (n_perm + 1), (ge_ssb + 1) / (n_perm + 1))),
    }


# ======================================================= pairwise contrasts

def _bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def benjamini_hochberg(pvals) -> np.ndarray:
    return _bh(pvals)


def stuart_maxwell(table: np.ndarray) -> tuple[float, int, float]:
    """Marginal homogeneity for a square paired contingency table."""
    t = np.asarray(table, float)
    used = (t.sum(0) + t.sum(1)) > 0
    t = t[np.ix_(used, used)]
    k = t.shape[0]
    if k < 2:
        return 0.0, 0, 1.0
    d = (t.sum(1) - t.sum(0))[:-1]
    S = -(t + t.T)
    np.fill_diagonal(S, t.sum(1) + t.sum(0) - 2 * np.diag(t))
    S = S[:-1, :-1]
    try:
        chi2 = float(d @ np.linalg.solve(S, d))
    except np.linalg.LinAlgError:
        chi2 = float(d @ np.linalg.pinv(S) @ d)
    df = k - 1
    return chi2, df, float(sps.chi2.sf(chi2, df))


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5)))


def quadratic_kappa(y_true, y_pred, k: int = 5) -> float:
    y_true = np.asarray(y_true, int) - 1
    y_pred = np.asarray(y_pred, int) - 1
    O = np.zeros((k, k))
    np.add.at(O, (y_true, y_pred), 1)
    W = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / max((W * E).sum(), 1e-12))


@dataclass
class ContrastResult:
    model: str
    a: str
    b: str
    n_cases: int
    mean_shift: float            # mean over cases of (mean ESI_a - mean ESI_b)
    se_cluster: float
    ci_low: float
    ci_high: float
    ci_boot_low: float
    ci_boot_high: float
    p_case: float                # PRIMARY: case-level sign-flip permutation
                                 # (does the average case shift? generalises over cases)
    p_sharp: float               # within-case label permutation: does the label
                                 # change ANY case's answers? (conditional on these cases)
    p_wilcoxon: float
    p_sign: float
    cohen_dz: float
    n_cases_a_less_urgent: int   # per-case mean shift > 0
    n_cases_a_more_urgent: int
    high_acuity_diff: float      # P(ESI<=2 | a) - P(ESI<=2 | b), case-averaged
    high_acuity_ci: tuple
    p_high_acuity: float         # case-level sign-flip on per-case P(ESI<=2) shift
    undertriage_diff: float      # among true ESI 1-2: P(pred>=3|a)-P(pred>=3|b)
    undertriage_ci: tuple
    n_true_high: int
    modal_flip_rate: float       # EQUITRIAGE-style, on modal answers (all samples)
    modal_flip_rate_halfsample: float  # same, using half the samples per side
    modal_noise_floor: float     # split-half modal flip rate within the same label
                                 # (compare with modal_flip_rate_halfsample)
    modal_a_worse: int           # modal ESI_a > ESI_b (a less urgent)
    modal_b_worse: int
    directional_ratio: float     # a_worse / b_worse  (EQUITRIAGE F/M when a=female)
    directional_ratio_ci: tuple
    p_directional_binom: float
    stuart_maxwell_chi2: float
    stuart_maxwell_p: float
    mcnemar_high_p: float
    single_sample_flip_rate: float   # what a one-shot study (e.g. YentlBench) would see
    single_sample_retest_flip: float # same prompt twice, one shot each
    q: float = np.nan                # BH-adjusted p_case, filled by analyze()

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def _split_half_modal_flip(p: Prepared, cond: str, rng) -> float:
    d = p.df[p.df["condition"] == cond]
    flips, tot = 0, 0
    for _, g in d.groupby("case_id"):
        x = g["esi"].to_numpy()
        if len(x) < 4:
            continue
        x = rng.permutation(x)
        h = len(x) // 2
        m1 = np.argmax(_bincount(x[:h], minlength=6))
        m2 = np.argmax(_bincount(x[h:2 * h], minlength=6))
        flips += m1 != m2
        tot += 1
    return flips / tot if tot else np.nan


def _split_half_modal_flip_between(p: Prepared, a: str, b: str, rng) -> float:
    """Modal flip a vs b using the SAME half-sample size as the noise floor,
    so the two numbers are directly comparable."""
    da = p.df[p.df["condition"] == a].groupby("case_id")["esi"].apply(np.array)
    db = p.df[p.df["condition"] == b].groupby("case_id")["esi"].apply(np.array)
    flips, tot = 0, 0
    for cid in da.index.intersection(db.index):
        xa, xb = rng.permutation(da[cid]), rng.permutation(db[cid])
        h = min(len(xa), len(xb)) // 2
        if h < 2:
            continue
        flips += np.argmax(_bincount(xa[:h], minlength=6)) != \
            np.argmax(_bincount(xb[:h], minlength=6))
        tot += 1
    return flips / tot if tot else np.nan


def _signflip_p(d: np.ndarray, n_perm: int, rng) -> float:
    """Two-sided sign-flip permutation test that mean(d) = 0. Exact
    enumeration when there are at most 16 non-zero cases."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    nzi = np.abs(d) > 1e-12
    k = int(nzi.sum())
    if k == 0:
        return 1.0
    dz = d[nzi]
    if k <= 16:
        signs = ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) * 2 - 1
        null = np.abs((signs * dz).sum(1) / len(d))
        return float((null >= obs - 1e-12).mean())
    ge = 0
    for chunk in range(0, n_perm, 1000):
        m = min(1000, n_perm - chunk)
        signs = rng.choice([-1.0, 1.0], size=(m, k))
        ge += int((np.abs((signs * dz).sum(1) / len(d)) >= obs - 1e-12).sum())
    return (ge + 1) / (n_perm + 1)


def contrast(p: Prepared, a: str, b: str, n_perm: int = 5000,
             n_boot: int = 2000, seed: int = 0) -> ContrastResult:
    rng = np.random.default_rng(seed)
    case, lab, y, nc = _encode(p, [a, b])
    m, n = _group_means(case, lab, y, nc, 2)
    d = m[:, 0] - m[:, 1]
    hi = (y <= 2).astype(float)
    mh, _ = _group_means(case, lab, hi, nc, 2)
    dh = mh[:, 0] - mh[:, 1]

    mean_d = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if nc > 1 else np.nan
    se = sd / np.sqrt(nc) if nc > 1 else np.nan
    tcrit = sps.t.ppf(0.975, max(nc - 1, 1))
    dz = mean_d / sd if sd and sd > 0 else 0.0

    # (1) Within-case label permutation: exact test of the SHARP null that the
    #     label changes nothing for any case. Treats these cases as fixed, so
    #     a model whose label effects are random in direction across cases is
    #     (correctly) flagged here even if the average is zero.
    ge = 0
    for _ in range(n_perm):
        pl = _permute_within(case, lab, rng)
        mp, _ = _group_means(case, pl, y, nc, 2)
        ge += abs(np.mean(mp[:, 0] - mp[:, 1])) >= abs(mean_d) - 1e-12
    p_sharp = (ge + 1) / (n_perm + 1)

    # (2) Case-level sign-flip permutation on per-case shifts: tests whether
    #     the AVERAGE case is shifted, treating cases as a sample from a
    #     population. Sampling noise is already inside each d_c. This is the
    #     right test for a directional claim ("women are triaged less urgently").
    p_case = _signflip_p(d, n_perm, rng)
    p_hi = _signflip_p(dh, n_perm, rng)

    # case bootstrap
    bi = rng.integers(0, nc, size=(n_boot, nc))
    boot = d[bi].mean(axis=1)
    boot_h = dh[bi].mean(axis=1)

    nz = d[np.abs(d) > 1e-12]
    p_w = float(sps.wilcoxon(nz).pvalue) if len(nz) >= 2 else 1.0
    pos, neg = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    p_sign = float(sps.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else 1.0

    # under-triage among cases the nurse rated ESI 1-2
    true_hi = p.cases.reindex(sorted(p.df["case_id"].unique()))["esi_true"].to_numpy() <= 2
    ut = 1 - mh  # share of answers ESI>=3
    if true_hi.sum() > 1:
        dut = ut[true_hi, 0] - ut[true_hi, 1]
        boot_ut = dut[rng.integers(0, len(dut), size=(n_boot, len(dut)))].mean(axis=1)
        ut_diff = float(dut.mean())
        ut_ci = (float(np.quantile(boot_ut, .025)), float(np.quantile(boot_ut, .975)))
    else:
        ut_diff, ut_ci = np.nan, (np.nan, np.nan)

    # modal-answer analyses (comparability with EQUITRIAGE / YentlBench)
    cells = cell_table(p)
    ma = cells[cells.condition == a].set_index("case_id")["mode"]
    mb = cells[cells.condition == b].set_index("case_id")["mode"]
    ma, mb = ma.align(mb, join="inner")
    table = np.zeros((5, 5))
    np.add.at(table, (ma.values - 1, mb.values - 1), 1)
    a_worse = int((ma > mb).sum())
    b_worse = int((ma < mb).sum())
    ratio = a_worse / b_worse if b_worse else np.inf
    # bootstrap the ratio over cases
    diffs = np.sign(ma.values - mb.values)
    rb = []
    for _ in range(n_boot):
        s = diffs[rng.integers(0, len(diffs), len(diffs))]
        bw = (s < 0).sum()
        rb.append((s > 0).sum() / bw if bw else np.nan)
    rb = np.array(rb, float)
    rb = rb[np.isfinite(rb)]
    ratio_ci = (float(np.quantile(rb, .025)), float(np.quantile(rb, .975))) if len(rb) else (np.nan, np.nan)
    p_dir = float(sps.binomtest(a_worse, a_worse + b_worse, 0.5).pvalue) if a_worse + b_worse else 1.0
    sm_chi2, _, sm_p = stuart_maxwell(table)
    hb = int(((ma <= 2) & (mb >= 3)).sum())
    hc = int(((ma >= 3) & (mb <= 2)).sum())

    # what a single-shot design would have reported
    s0 = p.df[p.df["sample"] == p.df.groupby(["case_id", "condition"])["sample"].transform("min")]
    sa = s0[s0.condition == a].set_index("case_id")["esi"]
    sb = s0[s0.condition == b].set_index("case_id")["esi"]
    sa, sb = sa.align(sb, join="inner")
    single_flip = float((sa != sb).mean()) if len(sa) else np.nan
    retest = _retest_flip(p, a)

    nf = [x for x in (_split_half_modal_flip(p, a, rng),
                      _split_half_modal_flip(p, b, rng)) if np.isfinite(x)]
    noise_floor = float(np.mean(nf)) if nf else np.nan
    between_half = _split_half_modal_flip_between(p, a, b, rng)

    return ContrastResult(
        model=p.model, a=a, b=b, n_cases=nc,
        mean_shift=mean_d, se_cluster=se,
        ci_low=mean_d - tcrit * se, ci_high=mean_d + tcrit * se,
        ci_boot_low=float(np.quantile(boot, .025)), ci_boot_high=float(np.quantile(boot, .975)),
        p_case=p_case, p_sharp=p_sharp, p_wilcoxon=p_w, p_sign=p_sign, cohen_dz=dz,
        n_cases_a_less_urgent=pos, n_cases_a_more_urgent=neg,
        high_acuity_diff=float(np.mean(dh)),
        high_acuity_ci=(float(np.quantile(boot_h, .025)), float(np.quantile(boot_h, .975))),
        p_high_acuity=p_hi,
        undertriage_diff=ut_diff, undertriage_ci=ut_ci, n_true_high=int(true_hi.sum()),
        modal_flip_rate=float((ma != mb).mean()),
        modal_flip_rate_halfsample=float(between_half),
        modal_noise_floor=float(noise_floor),
        modal_a_worse=a_worse, modal_b_worse=b_worse,
        directional_ratio=ratio, directional_ratio_ci=ratio_ci,
        p_directional_binom=p_dir,
        stuart_maxwell_chi2=sm_chi2, stuart_maxwell_p=sm_p,
        mcnemar_high_p=exact_mcnemar(hb, hc),
        single_sample_flip_rate=single_flip, single_sample_retest_flip=retest,
    )


def _retest_flip(p: Prepared, cond: str) -> float:
    """Flip rate between two single samples of the SAME prompt."""
    d = p.df[p.df["condition"] == cond].sort_values("sample")
    g = d.groupby("case_id")["esi"]
    first = g.nth(0)
    second = g.nth(1)
    f = d.loc[first.index, ["case_id", "esi"]].set_index("case_id")["esi"]
    s = d.loc[second.index, ["case_id", "esi"]].set_index("case_id")["esi"]
    f, s = f.align(s, join="inner")
    return float((f != s).mean()) if len(f) else np.nan


# ======================================================== accuracy & groups

def accuracy_by_condition(p: Prepared, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Agreement with nurse ESI, per condition. Reference labels are human
    decisions and may themselves be biased; treat as context, not truth."""
    rng = np.random.default_rng(seed)
    d = p.df.join(p.cases["esi_true"], on="case_id")
    rows = []
    for cond, g in d.groupby("condition"):
        err = g["esi"] - g["esi_true"]
        per_case = g.assign(ok=(err == 0)).groupby("case_id")["ok"].mean().to_numpy()
        boot = per_case[rng.integers(0, len(per_case), (n_boot, len(per_case)))].mean(1)
        rows.append({
            "condition": cond,
            "exact_match": float((err == 0).mean()),
            "exact_match_ci": (float(np.quantile(boot, .025)), float(np.quantile(boot, .975))),
            "within_1": float((err.abs() <= 1).mean()),
            "weighted_kappa": quadratic_kappa(g["esi_true"], g["esi"]),
            "under_triage": float((err > 0).mean()),
            "over_triage": float((err < 0).mean()),
            "mean_signed_error": float(err.mean()),
            "share_high_acuity": float((g["esi"] <= 2).mean()),
        })
    return pd.DataFrame(rows).set_index("condition").reindex(p.conditions)


def guerra_adames_metrics(p: Prepared, female: str = "female", male: str = "male") -> dict:
    """NMD and DTS on modal answers (Guerra-Adames et al., 2025).
    Our convention: positive = less urgent. NMD > 0 => women triaged less urgently."""
    cells = cell_table(p)
    f = cells[cells.condition == female].set_index("case_id")["mode"]
    m = cells[cells.condition == male].set_index("case_id")["mode"]
    f, m = f.align(m, join="inner")
    up = float((f > m).mean())      # female version less urgent
    down = float((f < m).mean())    # female version more urgent
    return {"NMD_female_minus_male": float((f - m).mean()),
            "P_female_less_urgent": up, "P_female_more_urgent": down,
            "DTS": up - down}


def by_category(p: Prepared, a: str, b: str, min_cases: int = 8,
                n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    if "category" not in p.cases.columns:
        return pd.DataFrame()
    rows = []
    for cat, ids in p.cases.groupby("category").groups.items():
        if len(ids) < min_cases:
            continue
        sub = Prepared(p.model, p.df[p.df.case_id.isin(ids)], p.cases.loc[ids],
                       p.conditions, p.parse_failures)
        r = contrast(sub, a, b, n_perm=n_perm, n_boot=500, seed=seed)
        rows.append({"category": cat, "n_cases": r.n_cases, "mean_shift": r.mean_shift,
                     "ci_low": r.ci_boot_low, "ci_high": r.ci_boot_high,
                     "high_acuity_diff": r.high_acuity_diff, "p_case": r.p_case})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = _bh(out["p_case"])
        out = out.sort_values("mean_shift", ascending=False)
    return out


# ================================================================== driver

DEFAULT_CONTRASTS = [("female", "male"), ("unlabeled", "male"),
                     ("female", "unlabeled"), ("nonbinary", "male")]


def analyze(results: pd.DataFrame, cases: pd.DataFrame,
            models: list[str] | None = None,
            contrasts: list[tuple[str, str]] | None = None,
            n_perm: int = 5000, seed: int = 0) -> dict:
    models = models or sorted(results["model"].unique())
    contrasts = contrasts or DEFAULT_CONTRASTS
    out = {"models": {}, "contrasts": []}
    for mdl in models:
        conds = sorted(results.loc[results.model == mdl, "condition"].unique())
        p = prepare(results, cases, mdl, conds)
        rec = {
            "n_cases": len(p.case_index),
            "samples_per_cell": float(p.df.groupby(["case_id", "condition"]).size().mean()),
            "parse": parse_failure_test(p),
            "omnibus": omnibus_test(p, n_perm=min(n_perm, 2000), seed=seed),
            "accuracy": accuracy_by_condition(p, seed=seed),
        }
        if {"female", "male"} <= set(conds):
            rec["guerra_adames"] = guerra_adames_metrics(p)
            rec["by_category"] = by_category(p, "female", "male", seed=seed)
        out["models"][mdl] = rec
        for a, b in contrasts:
            if a in conds and b in conds:
                out["contrasts"].append(contrast(p, a, b, n_perm=n_perm, seed=seed))
    q = _bh([c.p_case for c in out["contrasts"]])
    for c, qq in zip(out["contrasts"], q):
        c.q = float(qq)
    return out


def contrasts_frame(res: dict) -> pd.DataFrame:
    return pd.DataFrame([c.as_dict() for c in res["contrasts"]])


# =================================================================== power

def power_simulation(n_cases: int, n_samples: int, shift_prob: float,
                     noise: float = 0.25, affected_share: float = 1.0,
                     n_sims: int = 300, alpha: float = 0.05,
                     n_perm: int = 300, seed: int = 0) -> dict:
    """Simulated power for the female-vs-male mean-shift test.

    Each case gets a latent acuity; each sample is that acuity plus noise
    (probability ``noise`` of moving one level up or down). Under the
    alternative, for an ``affected_share`` of cases the female label pushes
    each sample one level less urgent with probability ``shift_prob``.
    """
    from .simulate import simulate_results, synthetic_cases
    rng = np.random.default_rng(seed)
    hits = 0
    effects = []
    for s in range(n_sims):
        cases = synthetic_cases(n_cases, seed=int(rng.integers(1e9)))
        res = simulate_results(cases, model="sim", n_samples=n_samples, noise=noise,
                               female_shift=shift_prob, affected_share=affected_share,
                               conditions=["male", "female"], seed=int(rng.integers(1e9)))
        p = prepare(res, cases, "sim", ["male", "female"])
        c = contrast(p, "female", "male", n_perm=n_perm, n_boot=100,
                     seed=int(rng.integers(1e9)))
        hits += c.p_case < alpha
        effects.append(c.mean_shift)
    return {"power": hits / n_sims, "mean_effect": float(np.mean(effects)),
            "n_cases": n_cases, "n_samples": n_samples,
            "shift_prob": shift_prob, "noise": noise, "n_sims": n_sims}
