"""Check that the tests mean what they say.

For each scenario, simulate many studies and count how often each test
rejects at alpha = 0.05.

  null          label does nothing        -> every test should reject ~5%
  biased        female label shifts ESI   -> directional tests should reject often
  unstable      label moves cases in random directions, no average shift
                -> omnibus should reject; directional tests should stay ~5%
  one-shot      the YentlBench design: 1 sample per prompt, flip rate reported
                -> shows how big a flip rate pure noise produces

Run:  python -m tests.validate_calibration  [--reps 300]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from cftriage import stats as S
from cftriage.simulate import simulate_results, synthetic_cases

SCENARIOS = {
    "null": dict(),
    "biased (shift 0.10)": dict(female_shift=0.10),
    "unstable, no direction": dict(instability=0.20),
}


def one(scn_kw, n_cases, n_samples, seed, n_perm):
    rng = np.random.default_rng(seed)
    cases = synthetic_cases(n_cases, seed=int(rng.integers(1e9)))
    res = simulate_results(cases, model="m", n_samples=n_samples, noise=0.25,
                           conditions=("male", "female"), seed=int(rng.integers(1e9)),
                           **scn_kw)
    p = S.prepare(res, cases, "m", ["male", "female"])
    om = S.omnibus_test(p, n_perm=n_perm, seed=int(rng.integers(1e9)))
    c = S.contrast(p, "female", "male", n_perm=n_perm, n_boot=200, seed=int(rng.integers(1e9)))
    covered = c.ci_low <= c.mean_shift_true <= c.ci_high if hasattr(c, "mean_shift_true") else np.nan
    return {
        "omnibus": om["p_omnibus"], "excess_disagreement": om["p_excess_disagreement"],
        "ssb": om["p_ssb"], "case_signflip": c.p_case, "within_case_perm": c.p_sharp,
        "wilcoxon": c.p_wilcoxon, "sign": c.p_sign, "stuart_maxwell_modal": c.stuart_maxwell_p,
        "directional_binom_modal": c.p_directional_binom,
        "high_acuity": c.p_high_acuity,
        "ci_excludes_0": float(not (c.ci_low <= 0 <= c.ci_high)),
        "one_shot_flip": c.single_sample_flip_rate, "one_shot_retest": c.single_sample_retest_flip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--cases", type=int, default=70)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=499)
    a = ap.parse_args()
    rows = []
    t0 = time.time()
    for name, kw in SCENARIOS.items():
        recs = [one(kw, a.cases, a.samples, 1000 * i + hash(name) % 997, a.n_perm)
                for i in range(a.reps)]
        df = pd.DataFrame(recs)
        rate = {k: (df[k] < 0.05).mean() for k in df.columns
                if k not in ("ci_excludes_0", "one_shot_flip", "one_shot_retest")}
        rate["ci_excludes_0"] = df["ci_excludes_0"].mean()
        rate["one_shot_flip (mean)"] = df["one_shot_flip"].mean()
        rate["one_shot_retest (mean)"] = df["one_shot_retest"].mean()
        rows.append(pd.Series(rate, name=name))
        print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows).T
    se = np.sqrt(0.05 * 0.95 / a.reps)
    print(f"\nRejection rate at alpha=0.05 over {a.reps} simulated studies "
          f"({a.cases} cases x 2 labels x {a.samples} samples). "
          f"Monte Carlo SE at 5% = {se:.3f}\n")
    print(out.round(3).to_string())
    out.to_csv("calibration_results.csv")


if __name__ == "__main__":
    main()
