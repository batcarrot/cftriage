"""Fast correctness checks. Run: python -m pytest tests  (or python -m tests.test_basics)"""

import numpy as np
import pandas as pd

from cftriage import stats as S
from cftriage.runner import parse_esi
from cftriage.simulate import simulate_results, synthetic_cases
from cftriage.vignettes import DEFAULT_CONDITIONS, build_vignettes


def test_parse_esi():
    assert parse_esi('{"esi": 3}') == 3
    assert parse_esi('```json\n{"esi": 2}\n```') == 2
    assert parse_esi("Reasoning... ESI Level: 4") == 4
    assert parse_esi("Could be ESI 2, but final answer ESI: 3") == 3
    assert parse_esi("2") == 2
    assert parse_esi("I cannot determine this.") is None
    assert parse_esi('{"esi": 7}') is None


def test_stuart_maxwell_known_value():
    # Classic Stuart (1955) vision data, chi2 = 11.96 on 3 df.
    t = np.array([[1520, 266, 124, 66], [234, 1512, 432, 78],
                  [117, 362, 1772, 205], [36, 82, 179, 492]])
    chi2, df, p = S.stuart_maxwell(t)
    assert df == 3 and abs(chi2 - 11.96) < 0.05


def test_quadratic_kappa():
    y = np.array([1, 2, 3, 4, 5, 3, 3, 2])
    assert abs(S.quadratic_kappa(y, y) - 1) < 1e-12
    assert S.quadratic_kappa(y, 6 - y) < 0


def test_mcnemar_exact():
    assert S.exact_mcnemar(0, 0) == 1.0
    assert abs(S.exact_mcnemar(1, 9) - 0.0215) < 1e-3


def test_signflip_exact_small():
    rng = np.random.default_rng(0)
    assert S._signflip_p(np.array([1, 1, 1, 1, 1.0]), 1000, rng) == 2 / 32
    assert S._signflip_p(np.zeros(5), 1000, rng) == 1.0


def test_bh():
    q = S.benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    # sorted p: .01 .03 .04 .5 -> raw .04 .06 .0533 .5 -> step-up .04 .0533 .0533 .5
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])


def test_vignettes_only_demographics_differ():
    cases = synthetic_cases(20, seed=3)
    cases.loc[0, "chief_complaint"] = "Vaginal bleeding"
    vs = build_vignettes(cases, DEFAULT_CONDITIONS)
    assert len(vs.excluded) == 1
    assert len(vs.vignettes) == 19 * 3
    g = vs.vignettes.groupby("case_id")["prompt"]
    assert (g.nunique() == 3).all()


def test_planted_bias_is_found_and_null_is_not():
    cases = synthetic_cases(150, seed=11)
    biased = simulate_results(cases, model="b", n_samples=10, female_shift=0.2, seed=1)
    null = simulate_results(cases, model="n", n_samples=10, seed=2)
    res = S.analyze(pd.concat([biased, null]), cases, n_perm=999,
                    contrasts=[("female", "male")])
    c = {x.model: x for x in res["contrasts"]}
    assert c["b"].p_case < 0.01 and c["b"].mean_shift > 0.1
    assert c["n"].p_case > 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
