"""Web app: python -m cftriage serve

Standard library only (http.server). Long work (simulations, power analysis,
model runs, analysis) runs in background jobs that the page polls.

Views:
* Guide      - plain-language walkthrough and a three-choice wizard.
* Playground - simulated models with known behaviour.
* Power      - simulated power for the female-vs-male contrast.
* Studies    - build a study, query a model, analyze, write the report.

Two modes:

Local (default). Binds to 127.0.0.1. API keys come from the server's
environment. Only same-origin JSON requests to localhost are accepted, so
other web pages cannot start paid runs.

Public (--public, for hosting). Binds to 0.0.0.0. Each browser gets its own
workspace (a random session id sent in a header), deleted after a day of
inactivity. Visitors bring their own API key with each request; it is kept in
memory only while their calls run, never written to disk or logged, and the
server's own environment keys are never used. Local file paths, Ollama and
LiteLLM are disabled; cases come from the built-in MIMIC-IV-ED demo, a CSV
upload or synthetic data. Sizes and concurrent jobs are capped.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import re
import shutil
import threading
import time
import traceback
import urllib.error
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import __version__
from . import stats as S
from .report import verdict, write_report
from .runner import (AnthropicClient, GeminiClient, OpenAICompatibleClient, RunConfig,
                     client_from_spec, load_results, parse_esi, run)
from .simulate import SimulatedClient, simulate_results, synthetic_cases
from .vignettes import (AGE_CONDITIONS, DEFAULT_CONDITIONS, NONBINARY, build_vignettes,
                        load_cases_csv, load_mimic_ed_demo)

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SID_RE = re.compile(r"^[A-Za-z0-9-]{16,64}$")

# Caps that apply in public mode only.
LIMITS = {"cases": 1000, "samples": 20, "sim_models": 4, "power_points": 6, "power_sims": 200,
          "n_perm": 5000, "calls": 30000, "concurrency": 8, "csv_bytes": 5_000_000,
          "cpu_jobs": 2, "runs_total": 6, "session_ttl_h": 24}


# ------------------------------------------------------------ serialization

def _jsonable(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, pd.DataFrame):
        df = x.reset_index() if x.index.name is not None else x
        return _jsonable(df.to_dict(orient="records"))
    if isinstance(x, pd.Series):
        return _jsonable(x.to_dict())
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _results_payload(res: dict) -> dict:
    """analyze() output -> JSON, with each contrast's verdict attached."""
    contrasts = []
    for c in res["contrasts"]:
        om = res["models"][c.model]["omnibus"]
        d = _jsonable(c)
        d["omnibus_significant"] = om["p_omnibus"] < 0.05
        d["direction_significant"] = bool(c.q < 0.05)
        d["less_urgent_group"] = (c.a if c.mean_shift > 0 else c.b) if c.q < 0.05 else None
        d["verdict"] = verdict(c, om).replace("**", "")
        contrasts.append(d)
    return {"models": _jsonable(res["models"]), "contrasts": contrasts}


def _analyze_by_model(res_df, cases, contrasts, n_perm, seed, job):
    """S.analyze one model at a time (for progress), then BH across all."""
    models = sorted(res_df["model"].unique())
    job.total = len(models)
    out = {"models": {}, "contrasts": []}
    for i, m in enumerate(models):
        job.check_stop()
        job.message = f"analyzing {m}"
        r = S.analyze(res_df, cases, models=[m], contrasts=contrasts, n_perm=n_perm, seed=seed)
        out["models"].update(r["models"])
        out["contrasts"] += r["contrasts"]
        job.progress = i + 1
    q = S.benjamini_hochberg([c.p_case for c in out["contrasts"]])
    for c, qq in zip(out["contrasts"], q):
        c.q = float(qq)
    return out


_HTTP_HINTS = {
    400: "The provider rejected the request; often the model name is wrong.",
    401: "The provider didn't accept the API key. Check that you copied all of it.",
    403: "The API key isn't allowed to use this model.",
    404: "The provider doesn't know that model name. Check the spelling.",
    429: "The provider is limiting requests, or the account is out of credit.",
}


def _explain_http(e: urllib.error.HTTPError) -> str:
    """Plain-language reason for a provider error, plus its own message."""
    detail = ""
    try:
        body = json.loads(e.read()[:2000] or b"{}")
        err = body[0].get("error") if isinstance(body, list) else body.get("error", body)
        detail = (err.get("message") if isinstance(err, dict) else err) or ""
    except Exception:
        pass
    hint = _HTTP_HINTS.get(e.code, f"The provider answered with error {e.code}; try again shortly."
                           if e.code >= 500 else f"The provider answered with error {e.code}.")
    return f"{hint} ({str(detail)[:200]})" if detail else hint


def checkpoint_path(study: Path, name: str) -> Path:
    return study / "results" / (re.sub(r"[^A-Za-z0-9._-]", "_", name) + ".jsonl")


def _scrub(msg: str, secret: str | None) -> str:
    return msg.replace(secret, "[your key]") if secret else msg


# --------------------------------------------------------------------- jobs

class Cancelled(Exception):
    pass


class Job:
    def __init__(self, kind: str, fn, sid: str = "local", slot: threading.Semaphore | None = None,
                 **meta):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.sid, self.meta = kind, sid, meta
        self.status, self.message, self.error, self.result = "running", "", None, None
        self.progress, self.total = 0, 1
        self.started, self.finished = time.time(), None
        self.stop = threading.Event()
        self._fn, self._slot = fn, slot
        threading.Thread(target=self._run, daemon=True).start()

    def check_stop(self):
        if self.stop.is_set():
            raise Cancelled()

    def _run(self):
        acquired = False
        try:
            if self._slot is not None:
                if not self._slot.acquire(blocking=False):
                    self.message = "Waiting for a free slot (others are using the server)…"
                    while not self._slot.acquire(timeout=1):
                        self.check_stop()
                    self.message = ""
                acquired = True
                self.started = time.time()
            self.result = self._fn(self)
            self.status = "cancelled" if self.stop.is_set() else "done"
        except Cancelled:
            self.status = "cancelled"
        except Exception as e:
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            if self.kind != "run":  # run errors may echo provider text; keep logs clean
                traceback.print_exc()
        finally:
            if acquired:
                self._slot.release()
            self._fn = None  # drops the client, and with it any visitor key
            self.finished = time.time()

    def view(self) -> dict:
        prog = self.meta.get("progress_fn")
        if prog and self.status == "running":
            self.progress = prog()
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "message": self.message, "error": self.error,
                "progress": self.progress, "total": self.total,
                "elapsed": (self.finished or time.time()) - self.started,
                "label": self.meta.get("label"), "study": self.meta.get("study"),
                "result": self.result if self.status in ("done", "cancelled") else None}


class App:
    def __init__(self, workspace: Path, public: bool = False, mimic_demo: str | None = None,
                 janitor: bool = True):
        self.root = workspace.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.public = public
        self.mimic_demo = Path(mimic_demo) if mimic_demo else None
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.cpu_slots = threading.Semaphore(LIMITS["cpu_jobs"]) if public else None
        if public and janitor:
            threading.Thread(target=self._janitor, daemon=True).start()

    # ------------------------------------------------------- sessions

    def workspace(self, sid: str) -> Path:
        if not self.public:
            return self.root
        ws = self.root / "sessions" / sid
        ws.mkdir(parents=True, exist_ok=True)
        os.utime(ws)  # last-activity marker for the janitor
        return ws

    def _janitor(self):
        while True:
            time.sleep(1800)
            cutoff = time.time() - LIMITS["session_ttl_h"] * 3600
            busy = {j.sid for j in self.jobs.values() if j.status == "running"}
            for d in (self.root / "sessions").glob("*"):
                try:
                    if d.name not in busy and d.stat().st_mtime < cutoff:
                        shutil.rmtree(d, ignore_errors=True)
                except FileNotFoundError:
                    pass
            with self.lock:
                for k in [k for k, j in self.jobs.items()
                          if j.finished and j.finished < time.time() - 3600]:
                    del self.jobs[k]

    def _cap(self, value, name: str, label: str):
        if self.public and value > LIMITS[name]:
            raise ValueError(f"{label} is limited to {LIMITS[name]:,} on the hosted version. "
                             f"Run cftriage on your own computer for larger studies.")
        return value

    def clean(self, msg: str) -> str:
        return msg.replace(str(self.root), "") if self.public else msg

    def add(self, job: Job) -> dict:
        with self.lock:
            self.jobs[job.id] = job
        return {"job": job.id}

    def job_for(self, jid: str, sid: str) -> Job | None:
        j = self.jobs.get(jid)
        return j if j is not None and (not self.public or j.sid == sid) else None

    # ---------------------------------------------------------- studies

    def study_dir(self, ws: Path, name: str) -> Path:
        if not _NAME_RE.match(name or ""):
            raise ValueError("study names use letters, digits, '.', '_' and '-'")
        d = (ws / name).resolve()
        if d.parent != ws.resolve():
            raise ValueError("bad study name")
        return d

    def list_studies(self, ws: Path):
        out = []
        for d in sorted(ws.iterdir()):
            if d.is_dir() and (d / "conditions.json").exists() and (d / "cases.csv").exists():
                n = sum(1 for _ in open(d / "cases.csv")) - 1
                res = sorted((d / "results").glob("*.jsonl")) if (d / "results").exists() else []
                out.append({"name": d.name, "n_cases": n, "n_results": len(res),
                            "modified": d.stat().st_mtime})
        return out

    def study_detail(self, ws: Path, name: str):
        d = self.study_dir(ws, name)
        if not (d / "conditions.json").exists():
            raise ValueError(f"no study called {name!r}")
        conds = json.loads((d / "conditions.json").read_text())
        cases = pd.read_csv(d / "cases.csv", dtype={"case_id": str})
        vign = pd.read_csv(d / "vignettes.csv", dtype={"case_id": str})
        exc_path = d / "excluded.csv"
        exc = pd.read_csv(exc_path) if exc_path.exists() and exc_path.stat().st_size > 1 \
            else pd.DataFrame()
        first = vign[vign.case_id == vign.case_id.iloc[0]] if len(vign) else vign
        results = []
        for f in sorted((d / "results").glob("*.jsonl")) if (d / "results").exists() else []:
            recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            by_model: dict[str, dict] = {}
            for r in recs:
                key = (r["case_id"], r["condition"], r["sample"])
                m = by_model.setdefault(r["model"], {})
                m[key] = r
            for m, cells in by_model.items():
                vals = list(cells.values())
                results.append({
                    "file": f.name, "model": m, "n_records": len(vals),
                    "n_errors": sum(r.get("error") is not None for r in vals),
                    "n_unparsed": sum(r.get("error") is None and r.get("esi") is None for r in vals),
                    "samples": 1 + max((r["sample"] for r in vals), default=-1),
                })
        report = d / "report" / "report.md"
        return {
            "name": d.name, "path": d.name if self.public else str(d),
            "n_cases": len(cases), "n_prompts": len(vign),
            "conditions": conds,
            "excluded": len(exc),
            "exclusion_reasons": (exc["exclusion_reason"].value_counts().to_dict()
                                  if len(exc) and "exclusion_reason" in exc else {}),
            "esi_distribution": {int(k): int(v) for k, v in
                                 cases["esi_true"].value_counts().sort_index().items()},
            "system_prompt": (d / "system_prompt.txt").read_text(),
            "example": [{"condition": r.condition, "prompt": r.prompt} for r in first.itertuples()],
            "results": results,
            "report": (f"{d.name}/report/report.md" if self.public else str(report))
                      if report.exists() else None,
        }

    def study_zip(self, ws: Path, name: str) -> bytes:
        d = self.study_dir(ws, name)
        if not d.exists():
            raise ValueError("no such study")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    z.write(f, f"{d.name}/{f.relative_to(d)}")
        return buf.getvalue()

    def build_study(self, ws: Path, b: dict):
        out = self.study_dir(ws, b.get("name", ""))
        if out.exists():
            raise ValueError(f"a study called {out.name!r} already exists")
        src = b.get("source", "synthetic")
        seed = int(b.get("seed", 0))
        if src == "mimic":
            if b.get("path") and not self.public:
                folder = Path(b["path"]).expanduser()
            elif self.mimic_demo:
                folder = self.mimic_demo
            else:
                raise ValueError("give the folder that holds the MIMIC-IV-ED demo files")
            cases = load_mimic_ed_demo(folder)
        elif src == "csv":
            if b.get("csv_text"):
                text = b["csv_text"]
                if len(text.encode()) > LIMITS["csv_bytes"]:
                    raise ValueError("CSV files are limited to 5 MB")
                tmp = ws / f".upload-{uuid.uuid4().hex}.csv"
                try:
                    tmp.write_text(text)
                    cases = load_cases_csv(tmp)
                finally:
                    tmp.unlink(missing_ok=True)
            elif self.public:
                raise ValueError("choose a CSV file to upload")
            else:
                cases = load_cases_csv(Path(b["path"]).expanduser())
        else:
            cases = synthetic_cases(self._cap(int(b.get("n", 200)), "cases", "Number of cases"),
                                    seed=seed)
        conds = list(AGE_CONDITIONS if b.get("with_age") else DEFAULT_CONDITIONS)
        if b.get("nonbinary"):
            conds.append(NONBINARY)
        mx = b.get("max_cases") or (LIMITS["cases"] if self.public else None)
        if mx and len(cases) > int(mx):
            cases = cases.sample(int(mx), random_state=seed)
        vs = build_vignettes(cases, conds)
        from .cli import _save_study
        _save_study(vs, out)
        return self.study_detail(ws, out.name)

    # ------------------------------------------------------------- jobs

    # Each *_task returns (fn, label); fn(job) does the work. The server runs
    # it in a background Job; the static site (browser.py) runs it directly.

    def job_simulate(self, sid: str, b: dict):
        fn, label = self.simulate_task(b)
        return self.add(Job("simulate", fn, sid, self.cpu_slots, label=label))

    def simulate_task(self, b: dict):
        n_cases = self._cap(int(b.get("cases", 200)), "cases", "Number of cases")
        n_samples = self._cap(int(b.get("samples", 10)), "samples", "Samples per prompt")
        seed = int(b.get("seed", 0))
        n_perm = self._cap(int(b.get("n_perm", 1000)), "n_perm", "Permutations")
        models = b.get("models") or []
        if not models:
            raise ValueError("add at least one simulated model")
        self._cap(len(models), "sim_models", "Number of simulated models")
        if n_cases * n_samples * len(models) > 600_000:
            raise ValueError("too large for the playground; lower cases or samples")
        allowed = {"noise", "female_shift", "instability", "model_disagreement",
                   "unlabeled_shift", "affected_share"}

        def fn(job):
            job.message = "simulating"
            cases = synthetic_cases(n_cases, seed=seed)
            vs = build_vignettes(cases, DEFAULT_CONDITIONS)
            frames = []
            for i, m in enumerate(models):
                kw = {k: float(v) for k, v in m.items() if k in allowed}
                frames.append(simulate_results(vs.cases, model=str(m.get("name") or f"sim-{i+1}")[:64],
                                               n_samples=n_samples, seed=seed + i, **kw))
            res_df = pd.concat(frames, ignore_index=True)
            res = _analyze_by_model(res_df, vs.cases, [("female", "male"), ("unlabeled", "male")],
                                    n_perm, seed, job)
            return _results_payload(res)
        return fn, f"{len(models)} simulated models"

    def job_power(self, sid: str, b: dict):
        fn, label = self.power_task(b)
        return self.add(Job("power", fn, sid, self.cpu_slots, label=label))

    def power_task(self, b: dict):
        ns = [int(n) for n in b.get("cases", [70, 200, 500])]
        self._cap(len(ns), "power_points", "Number of case counts")
        for n in ns:
            self._cap(n, "cases", "Number of cases")
        samples = self._cap(int(b.get("samples", 10)), "samples", "Samples per prompt")
        shift = float(b.get("shift", 0.10))
        noise, affected = float(b.get("noise", 0.25)), float(b.get("affected", 1.0))
        sims = self._cap(int(b.get("sims", 100)), "power_sims", "Simulated studies per point")
        seed = int(b.get("seed", 0))

        def fn(job):
            job.total = len(ns)
            rows = []
            for i, n in enumerate(ns):
                job.check_stop()
                job.message = f"{n} cases"
                r = S.power_simulation(n, samples, shift, noise=noise,
                                       affected_share=affected, n_sims=sims, seed=seed)
                rows.append({"cases": n, "samples": samples, "shift": shift, "noise": noise,
                             "power": r["power"], "mean_effect": r["mean_effect"], "sims": sims})
                job.progress = i + 1
            return {"rows": _jsonable(rows)}
        return fn, f"power, shift {shift:.2f}"

    def _client(self, spec: str, key: str | None = None, sim_shift: float = 0.0):
        if spec == "simulated":
            return SimulatedClient(female_shift=sim_shift)
        provider, _, model = spec.partition(":")
        if not model or len(model) > 200 or re.search(r"\s", model):
            raise ValueError("enter a model name, for example claude-sonnet-4-6")
        key = (key or "").strip() or None
        if key and (len(key) > 400 or re.search(r"\s", key)):
            raise ValueError("that API key doesn't look right")
        if self.public:
            if provider not in KEY_ENV:
                raise ValueError(f"{provider!r} isn't available on the hosted version")
            if not key:
                raise ValueError("paste your API key to use a real AI")
        if key:
            if provider == "anthropic":
                return AnthropicClient(model, api_key=key)
            if provider == "openai":
                return OpenAICompatibleClient(model, api_key=key)
            if provider == "openrouter":
                return OpenAICompatibleClient(model, api_key=key, base_url="https://openrouter.ai/api/v1",
                                              key_env="OPENROUTER_API_KEY")
            if provider == "gemini":
                return GeminiClient(model, api_key=key)
        env = KEY_ENV.get(provider)
        if env and not os.environ.get(env):
            raise ValueError(f"{env} is not set in the server's environment. "
                             f"Export it and restart `python -m cftriage serve`.")
        return client_from_spec(spec)

    def test_call(self, ws: Path, b: dict):
        from .cli import _load_study
        key = b.get("api_key")
        # Without a study (the guided page checks the connection first), use
        # a synthetic case; it is only a connectivity check.
        vs = _load_study(self.study_dir(ws, b["study"])) if b.get("study") \
            else build_vignettes(synthetic_cases(5), DEFAULT_CONDITIONS)
        client = self._client(b["model"], key, float(b.get("sim_shift", 0)))
        row = vs.vignettes.iloc[0]
        t0 = time.time()
        try:
            text = client(vs.system_prompt, row.prompt, float(b.get("temperature", 1.0)),
                          int(b.get("max_tokens", 64)))
        except urllib.error.HTTPError as e:
            raise ValueError(_scrub(_explain_http(e), key)) from None
        except Exception as e:
            raise ValueError(_scrub(f"{type(e).__name__}: {e}", key)) from None
        return {"prompt": row.prompt, "response": text, "esi": parse_esi(text),
                "seconds": time.time() - t0}

    def job_run(self, sid: str, ws: Path, b: dict):
        from .cli import _load_study
        study = self.study_dir(ws, b["study"])
        vs = _load_study(study)
        spec = b["model"].strip()
        key = (b.get("api_key") or "").strip() or None
        client = self._client(spec, key, float(b.get("sim_shift", 0)))
        name = (b.get("name") or spec).strip()[:100]
        ck = checkpoint_path(study, name)
        running = [j for j in self.jobs.values() if j.kind == "run" and j.status == "running"]
        if any(j.meta.get("ck") == ck for j in running):
            raise ValueError(f"{name} is already running on this study")
        if self.public:
            if any(j.sid == sid for j in running):
                raise ValueError("you already have a test running; wait for it or stop it first")
            if len(running) >= LIMITS["runs_total"]:
                raise ValueError("the server is busy; please try again in a few minutes")
        cfg = RunConfig(name, self._cap(int(b.get("samples", 10)), "samples", "Samples per prompt"),
                        float(b.get("temperature", 1.0)), int(b.get("max_tokens", 64)),
                        self._cap(int(b.get("concurrency", 4)), "concurrency", "Concurrency"),
                        seed=int(b.get("seed", 0)))
        total = self._cap(len(vs.vignettes) * cfg.n_samples, "calls", "Number of questions")
        ck.parent.mkdir(exist_ok=True)

        def done_count():
            if not ck.exists():
                return 0
            keys = set()
            for line in ck.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["model"] == name and r.get("error") is None and r["sample"] < cfg.n_samples:
                        keys.add((r["case_id"], r["condition"], r["sample"]))
            return len(keys)

        def fn(job):
            job.total = total
            job.message = f"{spec}: {len(vs.cases)} cases × {len(vs.conditions)} conditions × {cfg.n_samples} samples"
            try:
                df = run(vs, client, cfg, checkpoint=ck, progress=False, stop=job.stop)
            except Exception as e:
                raise RuntimeError(_scrub(str(e), key)) from None
            job.progress = done_count()
            return {"calls": int(len(df)), "errors": int(df["error"].notna().sum()) if len(df) else 0,
                    "unparsed": float(df["esi"].isna().mean()) if len(df) else 0.0}
        return self.add(Job("run", fn, sid, None, label=f"{name} on {study.name}", ck=ck,
                            study=study.name, progress_fn=done_count))

    def job_analyze(self, sid: str, ws: Path, b: dict):
        fn, label = self.analyze_task(ws, b)
        return self.add(Job("analyze", fn, sid, self.cpu_slots, label=label))

    def analyze_task(self, ws: Path, b: dict):
        study = self.study_dir(ws, b["study"])
        n_perm = self._cap(int(b.get("n_perm", 2000)), "n_perm", "Permutations")
        seed = int(b.get("seed", 0))
        models = b.get("models") or None
        title = b.get("title") or "Counterfactual triage audit"
        public = self.public

        def fn(job):
            cases = pd.read_csv(study / "cases.csv", dtype={"case_id": str})
            files = sorted((study / "results").glob("*.jsonl"))
            if not files:
                raise ValueError("no results yet; run a model first")
            res_df = load_results(*files)
            if models:
                res_df = res_df[res_df.model.isin(models)]
            res = _analyze_by_model(res_df, cases, None, n_perm, seed, job)
            job.message = "writing report"
            path = write_report(res, study / "report", title=title)
            out = _results_payload(res)
            out["report"] = f"{study.name}/report/report.md" if public else str(path)
            return out
        return fn, f"analyze {study.name}"

    def info(self):
        return {"version": __version__, "public": self.public,
                "workspace": None if self.public else str(self.root),
                "keys": {k: (False if self.public else bool(os.environ.get(v)))
                         for k, v in KEY_ENV.items()},
                "key_env": KEY_ENV, "mimic_builtin": self.mimic_demo is not None,
                "limits": LIMITS if self.public else None}


# ------------------------------------------------------------------- HTTP

def _handler(app: App, port: int):
    page = (Path(__file__).with_name("webui.html")).read_bytes()
    local_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    class H(BaseHTTPRequestHandler):
        server_version = "cftriage"

        def log_message(self, fmt, *args):
            if not self.path.startswith("/api/jobs"):
                super().log_message(fmt, *args)

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("x-content-type-options", "nosniff")
            self.send_header("referrer-policy", "no-referrer")
            if app.public:
                # Hugging Face shows Spaces inside an iframe on huggingface.co.
                self.send_header("content-security-policy",
                                 "frame-ancestors 'self' https://huggingface.co https://*.hf.space")
            else:
                self.send_header("x-frame-options", "DENY")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(_jsonable(obj), allow_nan=False).encode())

        def _fail(self, e: Exception):
            msg = f"{type(e).__name__}: {e}"
            if app.public:  # never reveal server paths to visitors
                msg = msg.replace(str(app.root), "").replace(str(Path(__file__).parent), "")
            self._json({"error": msg}, 400)

        def _host_ok(self) -> bool:
            # Local: refuse DNS-rebinding hosts. Public: any host (behind a proxy).
            if app.public or self.headers.get("host") in local_hosts:
                return True
            self._json({"error": "bad host"}, 403)
            return False

        def _sid(self) -> str | None:
            if not app.public:
                return "local"
            sid = self.headers.get("x-cftriage-session") or ""
            return sid if _SID_RE.match(sid) else None

        def do_GET(self):
            if not self._host_ok():
                return
            path = urlparse(self.path).path
            try:
                if path == "/":
                    return self._send(200, page, "text/html; charset=utf-8")
                if path == "/api/info":
                    return self._json(app.info())
                sid = self._sid()
                if sid is None:
                    return self._json({"error": "missing session"}, 400)
                if path == "/api/studies":
                    return self._json(app.list_studies(app.workspace(sid)))
                m = re.fullmatch(r"/api/studies/([^/]+)/download", path)
                if m:
                    data = app.study_zip(app.workspace(sid), m.group(1))
                    return self._send(200, data, "application/zip", {
                        "content-disposition": f'attachment; filename="{m.group(1)}.zip"'})
                m = re.fullmatch(r"/api/studies/([^/]+)", path)
                if m:
                    return self._json(app.study_detail(app.workspace(sid), m.group(1)))
                if path == "/api/jobs":
                    return self._json([{k: v for k, v in j.view().items() if k != "result"}
                                       for j in list(app.jobs.values())
                                       if not app.public or j.sid == sid])
                m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", path)
                if m and app.job_for(m.group(1), sid):
                    v = app.job_for(m.group(1), sid).view()
                    if v["error"]:
                        v["error"] = app.clean(v["error"])
                    return self._json(v)
                self._json({"error": "not found"}, 404)
            except Exception as e:
                self._fail(e)

        def do_POST(self):
            if not self._host_ok():
                return
            # CSRF: same-origin JSON only (a cross-site page would need a CORS
            # preflight, which this server never approves).
            origin = self.headers.get("origin")
            if origin is not None:
                ok = local_hosts | ({self.headers.get("host"), self.headers.get("x-forwarded-host")}
                                    if app.public else set())
                if urlparse(origin).netloc not in ok:
                    return self._json({"error": "cross-origin request refused"}, 403)
            if not (self.headers.get("content-type") or "").startswith("application/json"):
                return self._json({"error": "expected application/json"}, 415)
            sid = self._sid()
            if sid is None:
                return self._json({"error": "missing session"}, 400)
            n = int(self.headers.get("content-length") or 0)
            if n > 8_000_000:
                return self._json({"error": "request too large"}, 413)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                path = urlparse(self.path).path
                ws = app.workspace(sid)
                routes = {
                    "/api/simulate": lambda: app.job_simulate(sid, body),
                    "/api/power": lambda: app.job_power(sid, body),
                    "/api/run": lambda: app.job_run(sid, ws, body),
                    "/api/analyze": lambda: app.job_analyze(sid, ws, body),
                    "/api/studies": lambda: app.build_study(ws, body),
                    "/api/test-call": lambda: app.test_call(ws, body),
                }
                if path in routes:
                    return self._json(routes[path]())
                m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/cancel", path)
                if m and app.job_for(m.group(1), sid):
                    app.job_for(m.group(1), sid).stop.set()
                    return self._json({"ok": True})
                self._json({"error": "not found"}, 404)
            except Exception as e:
                self._fail(e)

    return H


def serve(workspace: str | Path = ".", port: int = 8765, open_browser: bool = True,
          public: bool = False, mimic_demo: str | None = None):
    app = App(Path(workspace), public=public, mimic_demo=mimic_demo)
    host = "0.0.0.0" if public else "127.0.0.1"
    httpd = ThreadingHTTPServer((host, port), _handler(app, port))
    url = f"http://localhost:{port}/"
    mode = "public mode" if public else f"workspace: {app.root}"
    print(f"cftriage web app on {url}  ({mode})  Ctrl+C to stop", flush=True)
    if open_browser and not public:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
