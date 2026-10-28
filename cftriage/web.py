"""Local web app: python -m cftriage serve

Standard library only (http.server). Binds to 127.0.0.1 by default. Long work
(simulations, power analysis, model runs, analysis) runs in background jobs
that the page polls.

Three views:
* Playground - simulated models with known behaviour; see what each verdict
  looks like as you change noise, planted bias and instability.
* Power      - simulated power for the female-vs-male contrast.
* Studies    - build a study, query a model (API keys come from the server's
  environment, never from the page), analyze and write the Markdown report.

API keys are read from environment variables only. Because the server can
spend money on API calls, every POST must be JSON with a matching Origin, so
other web pages cannot drive it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import __version__
from . import stats as S
from .report import verdict, write_report
from .runner import RunConfig, client_from_spec, load_results, parse_esi, run
from .simulate import SimulatedClient, simulate_results, synthetic_cases
from .vignettes import (AGE_CONDITIONS, DEFAULT_CONDITIONS, NONBINARY, build_vignettes,
                        load_cases_csv, load_mimic_ed_demo)

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


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


# --------------------------------------------------------------------- jobs

class Cancelled(Exception):
    pass


class Job:
    def __init__(self, kind: str, fn, **meta):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.meta = kind, meta
        self.status, self.message, self.error, self.result = "running", "", None, None
        self.progress, self.total = 0, 1
        self.started, self.finished = time.time(), None
        self.stop = threading.Event()
        self._fn = fn
        threading.Thread(target=self._run, daemon=True).start()

    def check_stop(self):
        if self.stop.is_set():
            raise Cancelled()

    def _run(self):
        try:
            self.result = self._fn(self)
            self.status = "cancelled" if self.stop.is_set() else "done"
        except Cancelled:
            self.status = "cancelled"
        except Exception as e:
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            traceback.print_exc()
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
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def add(self, job: Job) -> dict:
        with self.lock:
            self.jobs[job.id] = job
        return {"job": job.id}

    # ---------------------------------------------------------- studies

    def study_dir(self, name: str) -> Path:
        if not _NAME_RE.match(name or ""):
            raise ValueError("study names use letters, digits, '.', '_' and '-'")
        d = (self.workspace / name).resolve()
        if d.parent != self.workspace:
            raise ValueError("bad study name")
        return d

    def list_studies(self):
        out = []
        for d in sorted(self.workspace.iterdir()):
            if d.is_dir() and (d / "conditions.json").exists() and (d / "cases.csv").exists():
                n = sum(1 for _ in open(d / "cases.csv")) - 1
                res = sorted((d / "results").glob("*.jsonl")) if (d / "results").exists() else []
                out.append({"name": d.name, "n_cases": n, "n_results": len(res),
                            "modified": d.stat().st_mtime})
        return out

    def study_detail(self, name: str):
        d = self.study_dir(name)
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
            "name": d.name, "path": str(d),
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
            "report": str(report) if report.exists() else None,
        }

    def build_study(self, b: dict):
        out = self.study_dir(b.get("name", ""))
        if out.exists():
            raise ValueError(f"a study called {out.name!r} already exists")
        src = b.get("source", "synthetic")
        seed = int(b.get("seed", 0))
        if src == "mimic":
            cases = load_mimic_ed_demo(Path(b["path"]).expanduser())
        elif src == "csv":
            cases = load_cases_csv(Path(b["path"]).expanduser())
        else:
            cases = synthetic_cases(int(b.get("n", 200)), seed=seed)
        conds = list(AGE_CONDITIONS if b.get("with_age") else DEFAULT_CONDITIONS)
        if b.get("nonbinary"):
            conds.append(NONBINARY)
        mx = b.get("max_cases")
        if mx and len(cases) > int(mx):
            cases = cases.sample(int(mx), random_state=seed)
        vs = build_vignettes(cases, conds)
        from .cli import _save_study
        _save_study(vs, out)
        return self.study_detail(out.name)

    # ------------------------------------------------------------- jobs

    def job_simulate(self, b: dict):
        n_cases, n_samples = int(b.get("cases", 200)), int(b.get("samples", 10))
        seed, n_perm = int(b.get("seed", 0)), int(b.get("n_perm", 1000))
        models = b.get("models") or []
        if not models:
            raise ValueError("add at least one simulated model")
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
                frames.append(simulate_results(vs.cases, model=str(m.get("name") or f"sim-{i+1}"),
                                               n_samples=n_samples, seed=seed + i, **kw))
            res_df = pd.concat(frames, ignore_index=True)
            res = _analyze_by_model(res_df, vs.cases, [("female", "male"), ("unlabeled", "male")],
                                    n_perm, seed, job)
            return _results_payload(res)
        return self.add(Job("simulate", fn, label=f"{len(models)} simulated models"))

    def job_power(self, b: dict):
        ns = [int(n) for n in b.get("cases", [70, 200, 500])]
        samples, shift = int(b.get("samples", 10)), float(b.get("shift", 0.10))
        noise, affected = float(b.get("noise", 0.25)), float(b.get("affected", 1.0))
        sims, seed = int(b.get("sims", 100)), int(b.get("seed", 0))

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
        return self.add(Job("power", fn, label=f"power, shift {shift:.2f}"))

    def _client(self, spec: str, sim_shift: float = 0.0):
        if spec == "simulated":
            return SimulatedClient(female_shift=sim_shift)
        provider = spec.partition(":")[0]
        env = KEY_ENV.get(provider)
        if env and not os.environ.get(env):
            raise ValueError(f"{env} is not set in the server's environment. "
                             f"Export it and restart `python -m cftriage serve`.")
        return client_from_spec(spec)

    def test_call(self, b: dict):
        from .cli import _load_study
        vs = _load_study(self.study_dir(b["study"]))
        client = self._client(b["model"], float(b.get("sim_shift", 0)))
        row = vs.vignettes.iloc[0]
        t0 = time.time()
        text = client(vs.system_prompt, row.prompt, float(b.get("temperature", 1.0)),
                      int(b.get("max_tokens", 64)))
        return {"prompt": row.prompt, "response": text, "esi": parse_esi(text),
                "seconds": time.time() - t0}

    def job_run(self, b: dict):
        from .cli import _load_study
        study = self.study_dir(b["study"])
        vs = _load_study(study)
        spec = b["model"].strip()
        client = self._client(spec, float(b.get("sim_shift", 0)))
        name = (b.get("name") or spec).strip()
        ck = study / "results" / (name.replace("/", "_").replace(":", "_") + ".jsonl")
        for j in self.jobs.values():
            if j.kind == "run" and j.status == "running" and j.meta.get("ck") == ck:
                raise ValueError(f"{name} is already running on this study")
        ck.parent.mkdir(exist_ok=True)
        cfg = RunConfig(name, int(b.get("samples", 10)), float(b.get("temperature", 1.0)),
                        int(b.get("max_tokens", 64)), int(b.get("concurrency", 4)),
                        seed=int(b.get("seed", 0)))
        total = len(vs.vignettes) * cfg.n_samples

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
            df = run(vs, client, cfg, checkpoint=ck, progress=False, stop=job.stop)
            job.progress = done_count()
            return {"calls": int(len(df)), "errors": int(df["error"].notna().sum()) if len(df) else 0,
                    "unparsed": float(df["esi"].isna().mean()) if len(df) else 0.0}
        return self.add(Job("run", fn, label=f"{name} on {study.name}", ck=ck, study=study.name,
                            progress_fn=done_count))

    def job_analyze(self, b: dict):
        study = self.study_dir(b["study"])
        n_perm, seed = int(b.get("n_perm", 2000)), int(b.get("seed", 0))
        models = b.get("models") or None
        title = b.get("title") or "Counterfactual triage audit"

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
            out["report"] = str(path)
            return out
        return self.add(Job("analyze", fn, label=f"analyze {study.name}"))

    def info(self):
        return {"version": __version__, "workspace": str(self.workspace),
                "keys": {k: bool(os.environ.get(v)) for k, v in KEY_ENV.items()},
                "key_env": KEY_ENV}


# ------------------------------------------------------------------- HTTP

def _handler(app: App, port: int):
    page = (Path(__file__).with_name("webui.html")).read_bytes()
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    class H(BaseHTTPRequestHandler):
        server_version = "cftriage"

        def log_message(self, fmt, *args):
            if not self.path.startswith("/api/jobs"):
                super().log_message(fmt, *args)

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("x-content-type-options", "nosniff")
            self.send_header("x-frame-options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(_jsonable(obj), allow_nan=False).encode())

        def _guard(self) -> bool:
            # DNS rebinding: only answer to our own host names.
            if self.headers.get("host") not in hosts:
                self._json({"error": "bad host"}, 403)
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            try:
                if path == "/":
                    return self._send(200, page, "text/html; charset=utf-8")
                if path == "/api/info":
                    return self._json(app.info())
                if path == "/api/studies":
                    return self._json(app.list_studies())
                m = re.fullmatch(r"/api/studies/([^/]+)", path)
                if m:
                    return self._json(app.study_detail(m.group(1)))
                if path == "/api/jobs":
                    return self._json([{k: v for k, v in j.view().items() if k != "result"}
                                       for j in app.jobs.values()])
                m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", path)
                if m and m.group(1) in app.jobs:
                    return self._json(app.jobs[m.group(1)].view())
                self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

        def do_POST(self):
            if not self._guard():
                return
            # CSRF: require a same-origin JSON request (forces a CORS preflight,
            # which this server never approves, for any cross-site page).
            origin = self.headers.get("origin")
            if origin is not None and urlparse(origin).netloc not in hosts:
                return self._json({"error": "cross-origin request refused"}, 403)
            if not (self.headers.get("content-type") or "").startswith("application/json"):
                return self._json({"error": "expected application/json"}, 415)
            n = int(self.headers.get("content-length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n < 1_000_000 else {}
                path = urlparse(self.path).path
                routes = {"/api/simulate": app.job_simulate, "/api/power": app.job_power,
                          "/api/run": app.job_run, "/api/analyze": app.job_analyze,
                          "/api/studies": app.build_study, "/api/test-call": app.test_call}
                if path in routes:
                    return self._json(routes[path](body))
                m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/cancel", path)
                if m and m.group(1) in app.jobs:
                    app.jobs[m.group(1)].stop.set()
                    return self._json({"ok": True})
                self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    return H


def serve(workspace: str | Path = ".", port: int = 8765, open_browser: bool = True):
    app = App(Path(workspace))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(app, port))
    url = f"http://localhost:{port}/"
    print(f"cftriage web app on {url}  (workspace: {app.workspace})  Ctrl+C to stop")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
