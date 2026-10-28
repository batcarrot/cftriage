"""Send vignettes to models with repeated sampling.

Design choices (and why):

* **Repeated sampling.** Every (case, condition) prompt is sent ``n_samples``
  times. Disagreement *within* a prompt is the model's own noise; the
  statistics module compares it with disagreement *between* conditions.
  Without this, a sampling coin-flip is indistinguishable from bias.
* **No output caching.** A cache that returns the same answer for the same
  prompt hides noise in one arm and not the other.
* **Randomized call order.** Conditions are interleaved in random order so
  that time-of-day drift or rate-limit degradation cannot line up with a
  condition.
* **Checkpointing.** Every completed call is appended to a JSONL file and
  skipped on rerun, so a long run can be resumed.
* **No third-party dependencies.** HTTP calls use the standard library and
  honour HTTPS_PROXY. LiteLLM is supported if installed.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from .vignettes import VignetteSet

# ------------------------------------------------------------------ parsing

_JSON_RE = re.compile(r"\{[^{}]*\"?esi\"?\s*:\s*\"?([1-5])\"?[^{}]*\}", re.I)
_ESI_RE = re.compile(r"\bESI(?:\s*(?:level|score))?\s*[:=\-]?\s*\(?([1-5])\b", re.I)
_LEVEL_RE = re.compile(r"\blevel\s*[:=]?\s*([1-5])\b", re.I)


def parse_esi(text: str | None) -> int | None:
    """Extract an ESI level from a model response. Returns None if absent or
    ambiguous (two different levels named in the final answer region)."""
    if not text:
        return None
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "esi" in {k.lower() for k in obj}:
            v = int({k.lower(): v for k, v in obj.items()}["esi"])
            return v if 1 <= v <= 5 else None
    except (ValueError, TypeError):
        pass
    m = _JSON_RE.findall(t)
    if m:
        return int(m[-1])
    m = _ESI_RE.findall(t)
    if m:
        return int(m[-1])          # last mention = final answer
    m = _LEVEL_RE.findall(t)
    if m:
        return int(m[-1])
    if re.fullmatch(r"\s*([1-5])\s*\.?\s*", t):
        return int(t.strip(" ."))
    return None


# ------------------------------------------------------------------ clients

class ModelClient(Protocol):
    def __call__(self, system: str, user: str, temperature: float,
                 max_tokens: int) -> str: ...


def _post_json(url: str, payload: dict, headers: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


@dataclass
class AnthropicClient:
    model: str
    api_key: str | None = None
    base_url: str = "https://api.anthropic.com/v1/messages"

    def __call__(self, system, user, temperature, max_tokens):
        key = self.api_key or os.environ["ANTHROPIC_API_KEY"]
        out = _post_json(self.base_url, {
            "model": self.model, "system": system, "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }, {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in out.get("content", []))


@dataclass
class OpenAICompatibleClient:
    """OpenAI, OpenRouter, Together, Groq, vLLM, Ollama (/v1), etc."""
    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    key_env: str = "OPENAI_API_KEY"
    use_max_completion_tokens: bool = False   # newer OpenAI reasoning models

    def __call__(self, system, user, temperature, max_tokens):
        key = self.api_key or os.environ.get(self.key_env, "")
        payload = {
            "model": self.model, "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        payload["max_completion_tokens" if self.use_max_completion_tokens
                else "max_tokens"] = max_tokens
        out = _post_json(self.base_url.rstrip("/") + "/chat/completions", payload,
                         {"authorization": f"Bearer {key}"} if key else {})
        return out["choices"][0]["message"].get("content") or ""


@dataclass
class GeminiClient:
    model: str
    api_key: str | None = None

    def __call__(self, system, user, temperature, max_tokens):
        key = self.api_key or os.environ["GEMINI_API_KEY"]
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={key}")
        out = _post_json(url, {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens},
        }, {})
        parts = out["candidates"][0]["content"].get("parts", [])
        return "".join(p.get("text", "") for p in parts)


@dataclass
class LiteLLMClient:
    model: str

    def __call__(self, system, user, temperature, max_tokens):
        import litellm  # optional dependency
        r = litellm.completion(model=self.model, temperature=temperature,
                               max_tokens=max_tokens,
                               messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": user}])
        return r.choices[0].message.content or ""


def client_from_spec(spec: str) -> ModelClient:
    """'anthropic:claude-sonnet-4-6', 'openai:gpt-5.4-mini',
    'openrouter:google/gemini-3-flash', 'gemini:gemini-3.5-flash',
    'ollama:llama3.3', 'litellm:<any>'."""
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAICompatibleClient(model)
    if provider == "openrouter":
        return OpenAICompatibleClient(model, base_url="https://openrouter.ai/api/v1",
                                      key_env="OPENROUTER_API_KEY")
    if provider == "ollama":
        return OpenAICompatibleClient(model, base_url="http://localhost:11434/v1",
                                      key_env="OLLAMA_API_KEY")
    if provider == "gemini":
        return GeminiClient(model)
    if provider == "litellm":
        return LiteLLMClient(model)
    raise ValueError(f"Unknown provider in spec {spec!r}")


# ------------------------------------------------------------------ runner

FATAL_HTTP = {
    401: "The provider didn't accept the API key.",
    402: "The provider account is out of credit.",
    403: "The API key isn't allowed to use this model.",
    404: "The provider doesn't know that model name.",
}
EMPTY_LIMIT = 20
EMPTY_MESSAGE = ("The model's first answers were all empty. It may be spending the whole "
                 "max-tokens budget on hidden reasoning: raise max tokens or pick another model.")


@dataclass
class RunConfig:
    model_name: str
    n_samples: int = 10
    temperature: float = 1.0
    max_tokens: int = 64
    concurrency: int = 4
    max_retries: int = 5
    seed: int = 0


def run(vset: VignetteSet, client: ModelClient | Callable, cfg: RunConfig,
        checkpoint: str | Path | None = None, progress: bool = True,
        stop: threading.Event | None = None) -> pd.DataFrame:
    """Return one row per sample:
    model, case_id, condition, sample, response, esi, error, t.

    Setting ``stop`` skips calls that have not started yet; completed calls
    stay in the checkpoint, so the run can be resumed later."""
    done: dict[tuple, dict] = {}
    ck = Path(checkpoint) if checkpoint else None
    if ck and ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None and rec.get("response"):
                    done[(rec["model"], rec["case_id"], rec["condition"], rec["sample"])] = rec

    jobs = [(row.case_id, row.condition, row.prompt, s)
            for row in vset.vignettes.itertuples()
            for s in range(cfg.n_samples)
            if (cfg.model_name, row.case_id, row.condition, s) not in done]
    random.Random(cfg.seed).shuffle(jobs)

    lock = threading.Lock()
    fh = ck.open("a") if ck else None
    results = list(done.values())
    # Errors that will hit every call (bad key, no credit, unknown model) stop
    # the run instead of being recorded thousands of times.
    halt = threading.Event()
    fatal: list[str] = []
    answered = [0, 0]  # calls answered, of which empty

    def stop_with(msg):
        with lock:
            if not fatal:
                fatal.append(msg)
        halt.set()

    def work(job):
        case_id, cond, prompt, s = job
        if halt.is_set() or (stop is not None and stop.is_set()):
            return None
        err, text = None, None
        for attempt in range(cfg.max_retries):
            try:
                text = client(vset.system_prompt, prompt, cfg.temperature, cfg.max_tokens)
                err = None
                break
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read()[:200]!r}"
                if e.code in FATAL_HTTP:
                    stop_with(f"{FATAL_HTTP[e.code]} ({err})")
                    return None
                if e.code not in (408, 409, 429, 500, 502, 503, 504, 529):
                    break
            except Exception as e:  # network errors, parse errors in client
                err = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** attempt + random.random()))
        if err is None and not (text or "").strip():
            err = "empty response"
            with lock:
                answered[1] += 1
        if err is None or err == "empty response":
            with lock:
                answered[0] += 1
                all_empty = answered[0] >= EMPTY_LIMIT and answered[1] == answered[0]
            if all_empty:
                stop_with(EMPTY_MESSAGE)
        rec = {"model": cfg.model_name, "case_id": case_id, "condition": cond,
               "sample": s, "response": text, "esi": parse_esi(text),
               "error": err, "t": time.time()}
        with lock:
            results.append(rec)
            if fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        return rec

    total = len(jobs)
    try:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futs = [pool.submit(work, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                f.result()
                if progress and (i % 50 == 0 or i == total):
                    print(f"  {cfg.model_name}: {i}/{total} calls", flush=True)
    finally:
        if fh:
            fh.close()
    if fatal:
        raise RuntimeError(fatal[0] + " Answers received so far are kept; fix this and run "
                           "again to continue.")
    if not results:
        return pd.DataFrame(columns=["model", "case_id", "condition", "sample",
                                     "response", "esi", "error", "t"])
    df = pd.DataFrame(results)
    # Keep the newest record if a failed call was later retried successfully.
    df = df.sort_values("t").drop_duplicates(
        ["model", "case_id", "condition", "sample"], keep="last")
    return df.reset_index(drop=True)


def load_results(*paths: str | Path) -> pd.DataFrame:
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    df = pd.DataFrame(recs)
    return df.sort_values("t").drop_duplicates(
        ["model", "case_id", "condition", "sample"], keep="last").reset_index(drop=True)
