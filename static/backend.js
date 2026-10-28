// Static-site backend. The page (webui.html) calls /api/* exactly as it does
// against `python -m cftriage serve`; this script answers those calls in the
// browser instead:
// * statistics and studies: the cftriage Python package in a Web Worker
//   (worker.js, Pyodide), stored in this browser's IndexedDB;
// * model runs: fetch() straight from this browser to the AI provider, so the
//   visitor's API key never reaches any server but the provider's.
"use strict";
(() => {
  const VERSION = "__VERSION__";
  const KEY_ENV = { anthropic: "ANTHROPIC_API_KEY", openai: "OPENAI_API_KEY", openrouter: "OPENROUTER_API_KEY", gemini: "GEMINI_API_KEY" };
  const realFetch = window.fetch.bind(window);

  // ------------------------------------------------------------ worker RPC
  let worker, ready, reqId = 0;
  const pending = new Map();
  const jobs = new Map();

  function startWorker() {
    worker = new Worker("static/worker.js?v=__BUILD__");
    let markReady, markFailed;
    ready = new Promise((res, rej) => { markReady = res; markFailed = rej; });
    ready.catch(() => {});
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === "ready") { markReady(); hideToast(); return; }
      if (m.type === "failed") { markFailed(new Error(m.error)); showToast(`The statistics engine failed to load: ${m.error}. Reload to try again.`, true); return; }
      if (m.type === "progress") {
        const j = jobs.get(m.job);
        if (j && j.status === "running") Object.assign(j, { progress: m.done, total: m.total, message: m.message || j.message });
        return;
      }
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      if (m.error !== undefined) p.reject(new Error(m.error)); else p.resolve(m.ok);
    };
    worker.onerror = e => showToast(`The statistics engine stopped: ${e.message || "unknown error"}. Reload the page.`, true);
  }
  function call(op, args) {
    return new Promise((resolve, reject) => {
      const id = ++reqId;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, op, args });
    });
  }
  function restartWorker(reason) {
    worker.terminate();
    for (const [, p] of pending) p.reject(new Error(reason));
    pending.clear();
    startWorker();
  }

  // ------------------------------------------------------- loading notice
  let toast, toastTimer;
  function showToast(text, isError) {
    if (!document.body) return document.addEventListener("DOMContentLoaded", () => showToast(text, isError));
    if (!toast) {
      toast = document.createElement("div");
      toast.setAttribute("role", "status");
      toast.style.cssText = "position:fixed;left:50%;bottom:20px;transform:translateX(-50%);z-index:60;max-width:min(560px,calc(100% - 32px));" +
        "background:var(--ink);color:var(--surface);border-radius:10px;padding:10px 14px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.2)";
      document.body.append(toast);
    }
    toast.textContent = text;
    toast.style.display = "block";
    if (isError) toast.style.background = "var(--crit)";
  }
  function hideToast() { clearTimeout(toastTimer); if (toast) toast.style.display = "none"; }

  startWorker();
  toastTimer = setTimeout(() => showToast("Loading the statistics engine… The first visit takes about 15 seconds."), 400);

  // ------------------------------------------------------------------ jobs
  function newJob(kind, label, extra = {}) {
    const j = { id: Math.random().toString(16).slice(2, 14).padEnd(12, "0"), kind, label, status: "running", message: "", error: null,
      progress: 0, total: 1, started: Date.now(), finished: null, result: null, study: null, stop: false, ...extra };
    jobs.set(j.id, j);
    return j;
  }
  function finish(j, status, result, error) {
    Object.assign(j, { status, result: result ?? null, error: error ?? null, finished: Date.now() });
  }
  const clean = msg => String(msg).replace(/^(ValueError|RuntimeError): /, "");
  function view(j, withResult = true) {
    const v = { id: j.id, kind: j.kind, status: j.status, message: j.message, error: j.error, progress: j.progress, total: j.total,
      elapsed: ((j.finished || Date.now()) - j.started) / 1000, label: j.label, study: j.study };
    if (withResult) v.result = j.status === "done" || j.status === "cancelled" ? j.result : null;
    return v;
  }
  function cpuJob(kind, op, body, label) {
    const j = newJob(kind, label);
    j.message = "Loading the statistics engine…";
    ready.then(() => { if (j.message === "Loading the statistics engine…") j.message = "Working…"; }).catch(() => {});
    call(op, { ...body, _job: j.id })
      .then(r => { if (j.status === "running") finish(j, "done", r); })
      .catch(e => { if (j.status === "running") finish(j, "error", null, e.message); });
    return { job: j.id };
  }

  // ------------------------------------------------------- AI providers
  const HINTS = {
    400: "The provider rejected the request; often the model name is wrong.",
    401: "The provider didn't accept the API key. Check that you copied all of it.",
    402: "Your account with this provider is out of credit. Add credit, then start again; answers so far are kept.",
    403: "The API key isn't allowed to use this model.",
    404: "The provider doesn't know that model name. Check the spelling.",
    429: "The provider is limiting requests, or the account is out of credit.",
  };
  class ProviderError extends Error {
    constructor(status, detail) {
      const hint = HINTS[status] || (status >= 500 ? `The provider answered with error ${status}; try again shortly.` : `The provider answered with error ${status}.`);
      super(detail ? `${hint} (${String(detail).slice(0, 200)})` : hint);
      this.status = status;
    }
  }
  const RETRY = new Set([408, 409, 429, 500, 502, 503, 504, 529]);
  // These will fail every call the same way, so a run stops at the first one.
  const FATAL = new Set([401, 402, 403, 404]);
  class EmptyAnswer extends Error {
    constructor(truncated, maxTokens) {
      super(truncated
        ? `The model used its whole answer limit (${maxTokens} tokens) without giving an answer, usually because it "thinks" first. Raise Max tokens in the Studies tab, or pick a model without built-in reasoning.`
        : "The model returned an empty answer.");
      this.truncated = truncated;
    }
  }
  const EMPTY_LIMIT = 20;

  async function callProvider(spec, key, system, user, temperature = 1, maxTokens = 64) {
    const i = spec.indexOf(":"), provider = spec.slice(0, i), model = spec.slice(i + 1);
    let url, headers, body, extract, truncated;
    if (provider === "anthropic") {
      url = "https://api.anthropic.com/v1/messages";
      headers = { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" };
      body = { model, system, max_tokens: maxTokens, temperature, messages: [{ role: "user", content: user }] };
      extract = j => (j.content || []).map(b => b.text || "").join("");
      truncated = j => j.stop_reason === "max_tokens";
    } else if (provider === "openai" || provider === "openrouter") {
      url = provider === "openai" ? "https://api.openai.com/v1/chat/completions" : "https://openrouter.ai/api/v1/chat/completions";
      headers = { "content-type": "application/json", authorization: `Bearer ${key}` };
      body = { model, temperature, max_tokens: maxTokens, messages: [{ role: "system", content: system }, { role: "user", content: user }] };
      extract = j => j.choices[0].message.content || "";
      truncated = j => j.choices[0].finish_reason === "length";
    } else if (provider === "gemini") {
      url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
      headers = { "content-type": "application/json", "x-goog-api-key": key };
      body = { systemInstruction: { parts: [{ text: system }] }, contents: [{ role: "user", parts: [{ text: user }] }],
        generationConfig: { temperature, maxOutputTokens: maxTokens } };
      extract = j => ((j.candidates || [])[0]?.content?.parts || []).map(p => p.text || "").join("");
      truncated = j => (j.candidates || [])[0]?.finishReason === "MAX_TOKENS";
    } else throw new Error(`${provider} isn't available on the hosted version`);
    const r = await realFetch(url, { method: "POST", headers, body: JSON.stringify(body) });
    const text = await r.text();
    let j = null; try { j = JSON.parse(text); } catch {}
    if (!r.ok) {
      const err = j && (Array.isArray(j) ? j[0]?.error : j.error);
      throw new ProviderError(r.status, err && (err.message || (typeof err === "string" ? err : "")));
    }
    const answer = extract(j);
    if (!answer.trim()) throw new EmptyAnswer(truncated(j), maxTokens);
    return answer;
  }
  function checkSpec(spec, key) {
    if (spec === "simulated") return;
    const i = spec.indexOf(":"), provider = spec.slice(0, i), model = spec.slice(i + 1);
    if (!KEY_ENV[provider]) throw new Error(`${provider} isn't available on the hosted version`);
    if (!model || /\s/.test(model)) throw new Error("enter a model name, for example claude-sonnet-4-6");
    if (!key) throw new Error("paste your API key to use a real AI");
    if (/\s/.test(key) || key.length > 400) throw new Error("that API key doesn't look right");
  }
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  async function testCall(b) {
    const key = (b.api_key || "").trim();
    checkSpec(b.model, key);
    await ready;
    const { system, prompt } = await call("sample_prompt", {});
    const t0 = performance.now();
    const response = b.model === "simulated"
      ? (await call("simulate_answers", { system, prompts: [prompt] })).responses[0]
      : await callProvider(b.model, key, system, prompt, b.temperature ?? 1, b.max_tokens ?? 1024);
    const { esi } = await call("parse_esi", { text: response });
    return { prompt, response, esi, seconds: (performance.now() - t0) / 1000 };
  }

  function startRun(b) {
    const key = (b.api_key || "").trim(), spec = (b.model || "").trim();
    checkSpec(spec, key);
    for (const j of jobs.values()) if (j.kind === "run" && j.status === "running") throw new Error("you already have a test running; wait for it or stop it first");
    const name = (b.name || spec).trim().slice(0, 100);
    const j = newJob("run", `${name} on ${b.study}`, { study: b.study });
    const concurrency = Math.max(1, Math.min(8, +b.concurrency || 4));
    (async () => {
      j.message = "Loading the statistics engine…";
      await ready;
      const prep = await call("run_prepare", { ...b, model: spec, name });
      Object.assign(j, { total: prep.total, progress: prep.done,
        message: `${spec}: ${prep.n_cases} cases × ${prep.n_conditions} conditions × ${+b.samples || 10} samples` });
      const queue = prep.jobs.slice();
      let buf = [], saving = Promise.resolve(), calls = 0, errors = 0, unparsed = 0, fatal = null, answered = 0, empty = 0;
      const save = () => {
        if (!buf.length) return saving;
        const records = buf; buf = [];
        saving = saving.then(() => call("run_append", { study: b.study, name: prep.name, records }));
        return saving;
      };
      const record = (q, response, error) => {
        calls++; if (error) errors++; else j.progress++;
        buf.push({ case_id: q.case_id, condition: q.condition, sample: q.sample, response, error, t: Date.now() / 1000 });
        if (buf.length >= 25) save();
      };
      if (spec === "simulated") {
        while (queue.length && !j.stop) {
          const batch = queue.splice(0, 300);
          const { responses } = await call("simulate_answers", { system: prep.system, prompts: batch.map(q => q.prompt) });
          batch.forEach((q, i) => record(q, responses[i], null));
          await save();
        }
      } else {
        const worker = async () => {
          while (queue.length && !j.stop && !fatal) {
            const q = queue.shift();
            let text = null, err = null;
            for (let attempt = 0; attempt < 5; attempt++) {
              try { text = await callProvider(spec, key, prep.system, q.prompt, b.temperature ?? 1, b.max_tokens ?? 1024); err = null; answered++; break; }
              catch (e) {
                err = e.message;
                if (e instanceof EmptyAnswer) {
                  answered++; empty++;
                  if (e.truncated) fatal = e.message;
                  else if (answered >= EMPTY_LIMIT && empty === answered) fatal = "The model's first answers were all empty. Try another model, or raise Max tokens in the Studies tab.";
                  err = "empty response";
                  break;
                }
                const retry = !(e instanceof ProviderError) || RETRY.has(e.status);
                if (!retry) { if (FATAL.has(e.status)) fatal = e.message; break; }
                await sleep(Math.min(60000, 1000 * (2 ** attempt) + Math.random() * 1000));
                if (j.stop) break;
              }
            }
            if (!fatal) record(q, text, err);
          }
        };
        await Promise.all(Array.from({ length: concurrency }, worker));
      }
      await save();
      if (fatal) throw new Error(`${fatal} The run has stopped.`);
      finish(j, j.stop ? "cancelled" : "done", { calls, errors, unparsed });
    })().catch(e => { if (j.status === "running") finish(j, "error", null, clean(e.message)); });
    return { job: j.id };
  }

  // --------------------------------------------------------------- routes
  async function route(method, path, body) {
    let m;
    if (method === "GET") {
      if (path === "/api/info") return { version: VERSION, public: true, static: true, workspace: null,
        keys: Object.fromEntries(Object.keys(KEY_ENV).map(k => [k, false])), key_env: KEY_ENV, mimic_builtin: true, limits: null };
      if (path === "/api/jobs") return [...jobs.values()].map(j => view(j, false));
      if ((m = path.match(/^\/api\/jobs\/([0-9a-f]+)$/))) {
        const j = jobs.get(m[1]); if (!j) throw Object.assign(new Error("not found"), { status: 404 });
        return view(j);
      }
      await ready;
      if (path === "/api/studies") return call("list_studies", {});
      if ((m = path.match(/^\/api\/studies\/([^/]+)\/download$/))) {
        const { b64 } = await call("study_zip", { name: decodeURIComponent(m[1]) });
        return new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], { type: "application/zip" });
      }
      if ((m = path.match(/^\/api\/studies\/([^/]+)$/))) return call("study_detail", { name: decodeURIComponent(m[1]) });
    } else {
      if (path === "/api/simulate") return cpuJob("simulate", "simulate", body, `${(body.models || []).length} simulated models`);
      if (path === "/api/power") return cpuJob("power", "power", body, "power");
      if (path === "/api/analyze") return cpuJob("analyze", "analyze", body, `analyze ${body.study}`);
      if (path === "/api/run") return startRun(body);
      if (path === "/api/test-call") return testCall(body);
      if (path === "/api/studies") { await ready; return call("build_study", body); }
      if ((m = path.match(/^\/api\/jobs\/([0-9a-f]+)\/cancel$/))) {
        const j = jobs.get(m[1]);
        if (j && j.status === "running") {
          j.stop = true;
          if (j.kind !== "run") {  // Python can't be interrupted mid-calculation: restart it
            finish(j, "cancelled");
            restartWorker("Stopped.");
            for (const o of jobs.values()) if (o.status === "running" && o.kind !== "run") finish(o, "cancelled");
          }
        }
        return { ok: true };
      }
    }
    throw Object.assign(new Error("not found"), { status: 404 });
  }

  window.fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : input.url;
    if (!url.startsWith("/api/")) return realFetch(input, init);
    const path = url.split("?")[0], method = (init.method || "GET").toUpperCase();
    try {
      const out = await route(method, path, init.body ? JSON.parse(init.body) : {});
      if (out instanceof Blob) return new Response(out, { status: 200, headers: { "content-type": "application/zip" } });
      return new Response(JSON.stringify(out), { status: 200, headers: { "content-type": "application/json" } });
    } catch (e) {
      return new Response(JSON.stringify({ error: clean(e.message) }), { status: e.status || 400, headers: { "content-type": "application/json" } });
    }
  };
})();
