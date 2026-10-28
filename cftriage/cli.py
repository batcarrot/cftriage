"""Command line interface.

    python -m cftriage demo     --out demo_report
    python -m cftriage build    --mimic path/to/mimic-iv-ed-demo --out study
    python -m cftriage run      --study study --model anthropic:claude-sonnet-4-6 --samples 10
    python -m cftriage analyze  --study study --out study/report
    python -m cftriage power    --cases 70 --samples 10 --shift 0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import stats as S
from .report import write_report
from .runner import RunConfig, client_from_spec, load_results, run
from .simulate import SimulatedClient, simulate_results, synthetic_cases
from .vignettes import (AGE_CONDITIONS, DEFAULT_CONDITIONS, NONBINARY, VignetteSet,
                        build_vignettes, load_cases_csv, load_mimic_ed_demo)


def _save_study(vs: VignetteSet, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    vs.cases.to_csv(out / "cases.csv", index=False)
    vs.vignettes.to_csv(out / "vignettes.csv", index=False)
    vs.excluded.to_csv(out / "excluded.csv", index=False)
    (out / "system_prompt.txt").write_text(vs.system_prompt)
    (out / "conditions.json").write_text(json.dumps(
        [{"name": c.name, "line": c.line} for c in vs.conditions], indent=2))


def _load_study(folder: Path) -> VignetteSet:
    from .vignettes import Condition
    conds = [Condition(**c) for c in json.loads((folder / "conditions.json").read_text())]
    return VignetteSet(pd.read_csv(folder / "cases.csv", dtype={"case_id": str}),
                       pd.read_csv(folder / "vignettes.csv", dtype={"case_id": str}),
                       pd.read_csv(folder / "excluded.csv", dtype={"case_id": str})
                       if (folder / "excluded.csv").stat().st_size > 1 else pd.DataFrame(),
                       conds, (folder / "system_prompt.txt").read_text())


def cmd_build(a):
    if a.mimic:
        cases = load_mimic_ed_demo(a.mimic)
    elif a.csv:
        cases = load_cases_csv(a.csv)
    else:
        cases = synthetic_cases(a.synthetic, seed=a.seed)
    conds = list(AGE_CONDITIONS if a.with_age else DEFAULT_CONDITIONS)
    if a.nonbinary:
        conds.append(NONBINARY)
    if a.max_cases and len(cases) > a.max_cases:
        cases = cases.sample(a.max_cases, random_state=a.seed)
    vs = build_vignettes(cases, conds)
    _save_study(vs, Path(a.out))
    print(f"{len(vs.cases)} cases kept, {len(vs.excluded)} excluded, "
          f"{len(vs.vignettes)} prompts written to {a.out}")
    if len(vs.excluded):
        print(vs.excluded["exclusion_reason"].value_counts().to_string())
    print("\nExample prompt:\n" + vs.vignettes.iloc[0]["prompt"])


def cmd_run(a):
    vs = _load_study(Path(a.study))
    client = SimulatedClient(female_shift=a.sim_shift) if a.model == "simulated" \
        else client_from_spec(a.model)
    name = a.name or a.model
    ck = Path(a.study) / "results" / (name.replace("/", "_").replace(":", "_") + ".jsonl")
    ck.parent.mkdir(exist_ok=True)
    n_calls = len(vs.vignettes) * a.samples
    print(f"{name}: {len(vs.cases)} cases × {len(vs.conditions)} conditions × "
          f"{a.samples} samples = {n_calls} calls (resumes from {ck})")
    df = run(vs, client, RunConfig(name, a.samples, a.temperature, a.max_tokens,
                                   a.concurrency, seed=a.seed), checkpoint=ck)
    bad = df["esi"].isna().mean()
    print(f"done; unparseable responses: {bad:.1%}")


def cmd_analyze(a):
    study = Path(a.study)
    cases = pd.read_csv(study / "cases.csv", dtype={"case_id": str})
    files = a.results or sorted((study / "results").glob("*.jsonl"))
    res_df = load_results(*files)
    res = S.analyze(res_df, cases, n_perm=a.n_perm, seed=a.seed)
    path = write_report(res, a.out or study / "report", title=a.title)
    print(f"report written to {path}")


def cmd_power(a):
    for n in a.cases:
        r = S.power_simulation(n, a.samples, a.shift, noise=a.noise,
                               affected_share=a.affected, n_sims=a.sims, seed=a.seed)
        print(f"cases={n:5d} samples={a.samples:3d} shift={a.shift:.2f} "
              f"noise={a.noise:.2f} -> power {r['power']:.2f} "
              f"(true mean shift ≈ {r['mean_effect']:+.3f} ESI)")


def cmd_demo(a):
    """End-to-end on synthetic data with three simulated models whose
    behaviour is known, so you can see what each verdict looks like."""
    out = Path(a.out)
    cases = synthetic_cases(a.cases, seed=a.seed)
    vs = build_vignettes(cases, DEFAULT_CONDITIONS)
    _save_study(vs, out)
    sims = [
        ("sim-noise-only", dict()),
        ("sim-female-undertriage", dict(female_shift=0.12)),
        ("sim-unstable-no-direction", dict(instability=0.25)),
    ]
    frames = [simulate_results(vs.cases, model=m, n_samples=a.samples, seed=a.seed + i, **kw)
              for i, (m, kw) in enumerate(sims)]
    res_df = pd.concat(frames, ignore_index=True)
    (out / "results").mkdir(exist_ok=True)
    res_df.to_json(out / "results" / "simulated.jsonl", orient="records", lines=True)
    res = S.analyze(res_df, vs.cases, n_perm=a.n_perm, seed=a.seed,
                    contrasts=[("female", "male"), ("unlabeled", "male")])
    path = write_report(res, out / "report", title="Demo: simulated models with known behaviour")
    print(f"demo report written to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cftriage")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build counterfactual vignettes")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--mimic", help="folder with MIMIC-IV-ED demo CSVs")
    src.add_argument("--csv", help="CSV with the standard case columns")
    src.add_argument("--synthetic", type=int, help="N synthetic cases (pipeline testing only)")
    b.add_argument("--out", required=True)
    b.add_argument("--nonbinary", action="store_true")
    b.add_argument("--with-age", action="store_true",
                   help="use 'N-year-old man/woman' lines instead of 'Sex: X'")
    b.add_argument("--max-cases", type=int)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run", help="query a model")
    r.add_argument("--study", required=True)
    r.add_argument("--model", required=True,
                   help="anthropic:<id> | openai:<id> | openrouter:<id> | gemini:<id> | "
                        "ollama:<id> | litellm:<id> | simulated")
    r.add_argument("--name", help="label used in reports (default: model spec)")
    r.add_argument("--samples", type=int, default=10)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--max-tokens", type=int, default=64)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--sim-shift", type=float, default=0.0)
    r.set_defaults(func=cmd_run)

    z = sub.add_parser("analyze", help="statistics and report")
    z.add_argument("--study", required=True)
    z.add_argument("--results", nargs="*")
    z.add_argument("--out")
    z.add_argument("--title", default="Counterfactual triage audit")
    z.add_argument("--n-perm", type=int, default=5000)
    z.add_argument("--seed", type=int, default=0)
    z.set_defaults(func=cmd_analyze)

    pw = sub.add_parser("power", help="simulated power analysis")
    pw.add_argument("--cases", type=int, nargs="+", default=[70, 200, 500])
    pw.add_argument("--samples", type=int, default=10)
    pw.add_argument("--shift", type=float, default=0.10,
                    help="P(female label pushes a sample one level less urgent)")
    pw.add_argument("--noise", type=float, default=0.25)
    pw.add_argument("--affected", type=float, default=1.0)
    pw.add_argument("--sims", type=int, default=200)
    pw.add_argument("--seed", type=int, default=0)
    pw.set_defaults(func=cmd_power)

    d = sub.add_parser("demo", help="offline end-to-end demo with simulated models")
    d.add_argument("--out", default="demo")
    d.add_argument("--cases", type=int, default=200)
    d.add_argument("--samples", type=int, default=10)
    d.add_argument("--n-perm", type=int, default=2000)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
