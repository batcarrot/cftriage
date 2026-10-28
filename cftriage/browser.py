"""Backend for the static site, running in the visitor's browser (Pyodide).

The page talks to the same /api/* routes as the server; static/backend.js
answers them by calling ``dispatch`` here inside a Web Worker. Studies, caps
and analysis reuse web.App in public mode, so results match the server.

What differs from the server:
* No threads. Work runs synchronously in the worker; progress goes back to
  the page through a JS callback.
* No network from Python. Real model calls are made by backend.js straight
  from the browser to the provider, so API keys never reach any server.
  Python prepares the prompts (run_prepare) and stores the answers
  (run_append).
* The workspace lives in the browser's IndexedDB (mounted by worker.js).
"""

from __future__ import annotations

import base64
import json
import random
import traceback
from pathlib import Path

from .runner import parse_esi
from .simulate import SimulatedClient, synthetic_cases
from .vignettes import DEFAULT_CONDITIONS, build_vignettes
from .web import LIMITS, App, _jsonable, checkpoint_path

_app: App | None = None
_emit = None
_sim: SimulatedClient | None = None


def init(workspace: str, mimic_demo: str | None, progress) -> None:
    """progress(job_id, done, total, message) is a JS function."""
    global _app, _emit
    _app = App(Path(workspace), public=True, mimic_demo=mimic_demo, janitor=False)
    _emit = progress


def _ws() -> Path:
    return _app.workspace("browser")


class _Job:
    """Stands in for web.Job: same attributes, reported as they change."""

    def __init__(self, jid: str):
        self.__dict__.update(jid=jid, progress=0, total=1, message="")

    def __setattr__(self, k, v):
        self.__dict__[k] = v
        if k in ("progress", "total", "message") and _emit is not None:
            _emit(self.jid, self.progress, self.total, self.message)

    def check_stop(self):
        pass  # cancelling a CPU job restarts the worker instead


# ------------------------------------------------------------------ model runs

def run_prepare(b: dict) -> dict:
    """Prompts still to ask for this study and model name, in random order."""
    from .cli import _load_study
    global _sim
    study = _app.study_dir(_ws(), b["study"])
    vs = _load_study(study)
    name = (b.get("name") or b["model"]).strip()[:100]
    samples = _app._cap(int(b.get("samples", 10)), "samples", "Samples per prompt")
    total = _app._cap(len(vs.vignettes) * samples, "calls", "Number of questions")
    ck = checkpoint_path(study, name)
    done = set()
    if ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["model"] == name and r.get("error") is None and r["sample"] < samples:
                    done.add((r["case_id"], r["condition"], r["sample"]))
    jobs = [{"case_id": row.case_id, "condition": row.condition, "sample": s, "prompt": row.prompt}
            for row in vs.vignettes.itertuples() for s in range(samples)
            if (row.case_id, row.condition, s) not in done]
    random.Random(int(b.get("seed", 0))).shuffle(jobs)
    if b["model"] == "simulated":
        _sim = SimulatedClient(female_shift=float(b.get("sim_shift", 0)))
    return {"system": vs.system_prompt, "jobs": jobs, "total": total, "done": len(done),
            "name": name, "n_cases": len(vs.cases), "n_conditions": len(vs.conditions)}


def run_append(b: dict) -> dict:
    """Store answers; each record has case_id, condition, sample, response, error, t."""
    study = _app.study_dir(_ws(), b["study"])
    ck = checkpoint_path(study, b["name"])
    ck.parent.mkdir(exist_ok=True)
    with ck.open("a") as fh:
        for r in b["records"]:
            text = r.get("response")
            fh.write(json.dumps({"model": b["name"], "case_id": r["case_id"],
                                 "condition": r["condition"], "sample": int(r["sample"]),
                                 "response": text, "esi": parse_esi(text) if text else None,
                                 "error": r.get("error"), "t": r.get("t")}) + "\n")
    return {"ok": True}


def simulate_answers(b: dict) -> dict:
    """Answers from the built-in practice AI for a batch of prompts."""
    client = _sim or SimulatedClient()
    return {"responses": [client(b["system"], p, 1.0, 64) for p in b["prompts"]]}


def sample_prompt(b: dict) -> dict:
    vs = build_vignettes(synthetic_cases(5), DEFAULT_CONDITIONS)
    return {"system": vs.system_prompt, "prompt": vs.vignettes.iloc[0].prompt}


# -------------------------------------------------------------------- dispatch

def _cpu(task):
    def go(b):
        fn, _ = task(b)
        return fn(_Job(b.pop("_job")))
    return go


OPS = {
    "list_studies": lambda b: _app.list_studies(_ws()),
    "study_detail": lambda b: _app.study_detail(_ws(), b["name"]),
    "build_study": lambda b: _app.build_study(_ws(), b),
    "study_zip": lambda b: {"b64": base64.b64encode(_app.study_zip(_ws(), b["name"])).decode()},
    "simulate": _cpu(lambda b: _app.simulate_task(b)),
    "power": _cpu(lambda b: _app.power_task(b)),
    "analyze": _cpu(lambda b: _app.analyze_task(_ws(), b)),
    "run_prepare": run_prepare,
    "run_append": run_append,
    "simulate_answers": simulate_answers,
    "sample_prompt": sample_prompt,
    "parse_esi": lambda b: {"esi": parse_esi(b.get("text"))},
}


def dispatch(op: str, args_json: str) -> str:
    """Returns JSON: {"ok": result} or {"error": message}."""
    try:
        out = OPS[op](json.loads(args_json))
        return json.dumps({"ok": _jsonable(out)}, allow_nan=False)
    except Exception as e:
        traceback.print_exc()
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def limits() -> str:
    return json.dumps(LIMITS)
