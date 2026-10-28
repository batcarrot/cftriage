# cftriage

A toolkit for testing whether an LLM changes emergency triage decisions when only the patient's sex label changes, with statistics that separate a real label effect from the model's ordinary randomness.

It was built after reviewing YentlBench (Rytsareva, 2026) and the stronger studies around it: EQUITRIAGE (Young & Matthews, arXiv:2605.03998), Sense and Sensitivity (Gourabathina et al., arXiv:2609.38600), Lee et al. (J Healthc Inform Res, 2026), Guerra-Adames et al. (arXiv:2511.17124) and Wong (arXiv:2606.03641). It borrows their best design choices and fixes the problems that made some of their headline numbers hard to interpret.

## What it fixes

| Problem in single-shot designs | What cftriage does |
|---|---|
| One answer per prompt, so a coin-flip looks like bias | Every prompt is sampled repeatedly (default 10). Disagreement between two answers to the *same* prompt is measured and every test is relative to it. |
| A cache returned identical answers for identical prompts, hiding noise in one arm only | No output cache. Call order is randomized across conditions. |
| Flip rates reported without a noise floor | Flip rates are reported next to the flip rate between two samples of the identical prompt. |
| Most tests non-significant, yet directional claims made | One omnibus test ("does the label matter at all?") and one directional test per contrast, with Benjamini–Hochberg correction. The report only states a direction when the corrected test supports it. |
| Instability confused with bias | The toolkit distinguishes "the label moves answers" (omnibus) from "the label moves answers *in one direction on average*" (directional). A model can fail the first and pass the second. |
| Confusion-matrix cells read in the wrong orientation | Directional counts are computed in code with an explicit sign convention: positive always means the first-named group was triaged **less urgently**. |
| Claims about reasoning modes with no matched comparison | Analyze any two runs of the same model side by side (e.g. `--name model-thinking` vs `--name model-plain`). |

## Install

Python 3.9+, with numpy, scipy, pandas and matplotlib. No other dependencies; model calls use the standard library and respect `HTTPS_PROXY`. LiteLLM is used if you choose `litellm:` model specs.

```bash
pip install numpy scipy pandas matplotlib
cd cftriage
python -m tests.test_basics          # quick self-checks
python -m cftriage demo --out demo   # offline, simulated models, no API keys
```

The demo builds 200 synthetic cases, runs three simulated models with known behaviour (noise only, planted female under-triage, direction-free instability) and writes `demo/report/report.md`. Read it first: it shows what each verdict looks like when the truth is known.

## Web app

```bash
python -m cftriage serve            # opens http://localhost:8765/
```

A local web app, standard library only, with four views:

- **Guide** (opens first): a plain-language, four-step walkthrough for non-technical readers. It explains the experiment, shows why one answer is not enough, tests three pretend AIs with known behaviour, then runs a real AI through a three-choice wizard and explains the result in plain words.
- **Playground**: tune simulated models (noise, planted female under-triage, direction-free instability) and see the verdicts, effect plot and noise comparison update. No API calls.
- **Power**: power for the female-vs-male test across case counts, samples and noise.
- **Studies**: build a study (synthetic, CSV or MIMIC-IV-ED demo), test one call, run a model with live progress (stop and resume at any time), analyze and write the Markdown report.

Study folders live under `--workspace` (default: the current folder), so everything the web app does also works from the CLI. API keys come only from the server's environment. Export them before `serve`; the page never asks for them. The server listens on 127.0.0.1 only and refuses cross-origin requests, so other websites cannot start paid runs.

### Hosting it

`python -m cftriage serve --public` (or the `Dockerfile`) runs the hosted version:

- Each browser gets a private workspace, deleted after 24 hours.
- Visitors paste their own API key. It is held in memory only while their test runs, never written to disk or logged, and the server's own environment keys are never used.
- Local file paths, Ollama and LiteLLM are turned off. Cases come from the built-in [MIMIC-IV-ED demo](https://physionet.org/content/mimic-iv-ed-demo/2.2/) (open data, ODbL), a CSV upload, or synthetic data.
- Study sizes and simultaneous jobs are capped, and visitors can download their results and report as a zip.

To publish on a free [Hugging Face Space](https://huggingface.co/spaces) (2 CPUs):

1. Create a Space at <https://huggingface.co/new-space> with SDK **Docker** and template **Blank**.
2. Log in once with a write token: `hf auth login`.
3. Run `deploy/hf_space.sh <your-username>/<space-name>`. It pushes the committed code plus the Space config; the Space builds in a few minutes.

Re-run step 3 after each commit you want to publish.

## Running a real study

### 1. Get cases

The MIMIC-IV-ED Demo is open access (no credentialing):
<https://physionet.org/content/mimic-iv-ed-demo/2.2/>. Download `ed/edstays.csv.gz` and `ed/triage.csv.gz` into a folder. If you also place `hosp/patients.csv.gz` from the MIMIC-IV demo next to it, age is included.

```bash
python -m cftriage build --mimic path/to/mimic-iv-ed-demo --out study
#   --nonbinary       add a "Sex: Non-binary" condition
#   --with-age        use "Patient: 45-year-old woman" instead of "Sex: Female"
```

Any other source works if you provide a CSV with `case_id, chief_complaint, heart_rate, resp_rate, spo2, sbp, dbp, temperature, pain, esi_true` (and optionally `age`): `--csv cases.csv`.

`build` drops complaints where a sex swap would change the clinical meaning (pregnancy, testicular pain, etc.) and cases with missing vitals, writes the reasons to `excluded.csv`, and asserts that the clinical text is byte-identical across conditions for every case.

The full MIMIC-IV-ED requires PhysioNet credentialing, and its data use agreement restricts sending records to third-party services. Check PhysioNet's current guidance before running the full dataset through a commercial API.

### 2. Query models

```bash
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GEMINI_API_KEY=...  OPENROUTER_API_KEY=...
python -m cftriage run --study study --model anthropic:claude-sonnet-4-6 --samples 10
python -m cftriage run --study study --model openai:gpt-5.4-mini       --samples 10
python -m cftriage run --study study --model gemini:gemini-3.5-flash   --samples 10
python -m cftriage run --study study --model openrouter:deepseek/deepseek-v3.2 --samples 10
python -m cftriage run --study study --model ollama:llama3.3 --samples 10   # local
```

Model IDs change; use whatever your provider currently lists. Each run writes `study/results/<model>.jsonl` after every call and resumes from it if interrupted. Calls per model = cases × conditions × samples (200 cases × 3 conditions × 10 samples = 6,000 short calls).

Defaults: temperature 1.0 (the models' standard sampling), 64 output tokens, JSON answer `{"esi": n}`. If you plan to deploy at temperature 0, run that too: a deterministic model still has a noise floor across serving infrastructure, and EQUITRIAGE measured it at about 0.4% for one model.

Record the date, model version and provider for every run. Hosted models change.

### 3. Analyze

```bash
python -m cftriage analyze --study study --out study/report
```

Outputs `report.md`, `contrasts.csv` and three figures.

### 4. Plan sample size first

```bash
python -m cftriage power --cases 70 200 500 --samples 10 --shift 0.10
```

`--shift` is the probability that the female label pushes a single answer one ESI level less urgent. A shift of 0.10 moves the mean by about 0.1 ESI levels, similar to the planted effect in the demo.

## The statistics

Notation: case *c*, condition *j* (male, female, unlabeled, …), sample *s*. Higher ESI = less urgent.

**1. Does the label matter at all? (omnibus, per model)**

- *Excess disagreement*: how much more often two answers disagree when they come from different labels than from the same label. Zero means the label adds nothing beyond noise.
- *SSB*: between-label sum of squares of mean ESI within each case.
- Both get an exact p-value from a **within-case label permutation**: under the null that the label is irrelevant, all samples of a case are exchangeable across labels, so shuffling labels inside each case generates the null distribution. The omnibus decision is Bonferroni over the two.

**2. Is there a direction? (per contrast, e.g. female vs male)**

- Per-case shift *d_c* = mean ESI under label *a* − mean ESI under label *b*. Sampling noise is already inside each *d_c*.
- **Primary test: case-level sign-flip permutation** on the *d_c*. It tests whether the *average* case is shifted and generalizes over cases.
- Also reported: Wilcoxon signed-rank and sign tests on *d_c*; a case-clustered t interval and a case-bootstrap interval (Miller 2024, "Adding Error Bars to Evals", arXiv:2411.00640); Cohen's *d_z*.
- Also reported: the within-case label-permutation p for the same contrast. It tests the **sharp** null that the label changes nothing for any case.

The difference between the last two matters. During validation, a simulated model whose label effect pointed in random directions on different cases was flagged as "directional" by the within-case permutation test, because the realized random offsets in a finite sample rarely cancel exactly. That test answers "did the label change these cases?", which is true for such a model. It does not answer "are women triaged less urgently on average?". The case-level sign-flip test does, so it is the primary directional test.

**3. Clinical size**

- Δ P(ESI ≤ 2): change in the share of answers that would be seen immediately (the ESI 2/3 boundary), with its own case-level test.
- Δ under-triage: among cases the nurse rated ESI 1–2, the change in the share of answers at ESI 3 or worse.

**4. Comparability with earlier studies** (secondary; less powerful)

- EQUITRIAGE-style flip rate and directional F/M ratio on modal answers, with case-bootstrap CI and binomial test; plus a split-half noise floor so the flip rate has a reference.
- Stuart–Maxwell marginal homogeneity test on the modal 5×5 table (verified against Stuart's 1955 example) and exact McNemar on ESI ≤ 2.
- Guerra-Adames Net Mean Disadvantage and Directional Triage Skew.
- What a single-sample study would have reported: the one-shot flip rate between labels and the one-shot flip rate between two answers to the identical prompt.

**5. Accuracy against nurse ESI**, per condition: exact match (case-bootstrap CI), within ±1, quadratic-weighted κ, under- and over-triage rates. Reference labels are human decisions and may carry the very bias being tested, so they are context, not truth. The counterfactual tests above do not depend on them.

**6. Subgroups and multiplicity.** Female vs male by complaint category (exploratory, BH within model). All primary directional p-values are BH-adjusted across every model and contrast in the report.

**7. Data quality.** Parse failures per condition, with a χ² test. If failures differ by condition, dropping them is itself a biased filter.

## Validation

`tests/validate_calibration.py` simulates hundreds of studies under three known truths and counts how often each test rejects at α = 0.05. `tests/test_basics.py` checks the parser, Stuart–Maxwell, κ, McNemar, the sign-flip test, BH and the vignette builder.

### Calibration (600 simulated studies per scenario)

Each study: 70 cases × 2 labels × 10 samples, noise = 25% chance an answer moves one level. Cells are the share of studies in which the test rejected at α = 0.05. Monte Carlo SE at 5% ≈ 0.009.

| Test | Null (no effect) | Planted female shift (0.10) | Unstable, no direction |
|---|---|---|---|
| Omnibus (Bonferroni of two) | 0.053 | 0.218 | 0.998 |
| **Case-level sign-flip (primary directional)** | **0.043** | **0.917** | **0.043** |
| Within-case permutation (sharp null) | 0.045 | 0.932 | 0.302 |
| Wilcoxon signed-rank | 0.058 | 0.912 | 0.052 |
| Sign test | 0.025 | 0.800 | 0.032 |
| 95% cluster CI excludes 0 | 0.050 | 0.932 | 0.050 |
| Δ P(ESI ≤ 2), case-level | 0.045 | 0.407 | 0.038 |
| Stuart–Maxwell on modal answers | 0.000 | 0.000 | 0.082 |
| Binomial on modal directional flips | 0.002 | 0.003 | 0.018 |
| Mean one-shot flip rate between labels | 38.1% | 42.6% | 45.6% |
| Mean one-shot retest flip, identical prompt | 38.2% | 45.8% | 37.2% |

What this shows:

- **The primary tests hold their error rate.** Under no effect, every case-level test rejects about 5% of the time.
- **The case-level test catches a real one-way shift** about 92% of the time with only 70 cases, because repeated sampling averages out each case's noise.
- **It does not mistake instability for bias.** For a model that moves cases in random directions, it stays at 4%, while the omnibus flags the instability 99.8% of the time. The within-case permutation test rejects 30% here, which is why it is not used for directional claims.
- **Modal-answer methods are nearly blind.** Taking each prompt's most common answer and then comparing labels (as flip-ratio and Stuart–Maxwell analyses do) detected the planted bias in essentially none of 600 studies. Most of the signal lives in how often a case lands on one level versus the next, and the mode throws that away.
- **A single-sample design cannot tell bias from noise.** With one answer per prompt, labels disagree on about 38% of cases even when the label does nothing, the same rate as asking the identical question twice. This is why a one-shot consistency figure, such as YentlBench's per-model consistency rates, cannot by itself be read as a label effect: without the retest rate for the same model, there is nothing to compare it with.

Reproduce: `python -m tests.validate_calibration --reps 600`.

### Power (150 simulated studies per cell)

Probability of detecting a female-vs-male shift with the primary test at α = 0.05. "Shift" is the chance the female label pushes one answer one ESI level less urgent; 0.05 and 0.10 move the mean by about 0.05 and 0.10 levels. Noise as above.

| Shift | Samples per prompt | 70 cases | 200 cases | 500 cases |
|---|---|---|---|---|
| 0.05 | 1 | 0.05 | 0.10 | 0.25 |
| 0.05 | 5 | 0.19 | 0.49 | 0.89 |
| 0.05 | 10 | 0.38 | 0.77 | 1.00 |
| 0.10 | 1 | 0.15 | 0.41 | 0.74 |
| 0.10 | 5 | 0.57 | 0.98 | 1.00 |
| 0.10 | 10 | 0.87 | 1.00 | 1.00 |

Repeated sampling buys more than extra cases: 70 cases with 10 samples each beats 500 cases with one sample each. A one-sample study of 70 cases, the YentlBench design, has about a 5–15% chance of detecting shifts this size, little better than chance. For small effects, aim for 200+ cases at 10 samples. Your model's real noise level changes these numbers; run `python -m cftriage power` with `--noise` set to the within-label disagreement you observe in a pilot.

## Limitations

- **A vignette is not triage.** Real nurses see the patient, gait, distress and history. Text-only ESI assignment is a controlled audit, not a measure of deployed performance.
- **Only the label line changes.** Names, pronouns in free text and age interactions are not varied unless you add conditions (see `vignettes.Condition`). EQUITRIAGE found that names and gender tokens can act differently in different models.
- **Sex-linked exclusions use keywords.** Review `excluded.csv` and adjust `SEX_LINKED_PATTERN` for your data.
- **Small datasets limit subgroups.** The MIMIC-IV-ED Demo is small. Category-level results on it are exploratory at best.
- **The simulator is simple.** It checks that the statistics behave correctly under known truths; it is not a model of how real LLMs err.
- **Statistical significance is not clinical acceptability.** A real but small shift can still matter for the patients at the ESI 2/3 boundary; report Δ P(ESI ≤ 2) and Δ under-triage alongside p-values.

## Layout

```
cftriage/
  vignettes.py   load cases, build counterfactual prompts, exclusion rules
  runner.py      providers, repeated sampling, checkpointing, ESI parsing
  stats.py       all statistics
  report.py      markdown report and figures
  simulate.py    synthetic cases and simulated models
  cli.py         command line
  web.py         web app server (python -m cftriage serve [--public])
  webui.html     web app page
Dockerfile       hosted version (public mode, built-in MIMIC-IV-ED demo)
deploy/
  hf_space.sh    publish to a Hugging Face Space
tests/
  test_basics.py
  validate_calibration.py
```

## License

Copyright (c) 2026 Vivan Gupta. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE): free to use, modify and share for research, education, personal and other noncommercial purposes, including by charities, schools and universities, public research and public health organizations, and government bodies. Commercial use needs a separate license; open an issue or contact the author.

The built-in MIMIC-IV-ED demo cases are not covered by this license; they are open data under the [ODbL](https://physionet.org/content/mimic-iv-ed-demo/2.2/).
