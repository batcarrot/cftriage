// Web Worker for the static site: runs the cftriage Python package in
// Pyodide. backend.js sends {id, op, args}; we answer {id, ok} or {id, error},
// and send {type: "progress", job, done, total, message} while a job runs.
"use strict";
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
importScripts(PYODIDE + "pyodide.js");

let py, dispatch, matplotlibLoaded = false;

function syncfs(populate) {
  return new Promise((resolve, reject) => py.FS.syncfs(populate, err => err ? reject(err) : resolve()));
}

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not load ${url} (HTTP ${r.status})`);
  return r;
}

const ready = (async () => {
  const manifest = await (await fetchOk("manifest.json")).json();
  py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage(["numpy", "pandas", "scipy"]);

  py.FS.mkdirTree("/app/cftriage");
  await Promise.all(manifest.python.map(async f => {
    py.FS.writeFile(`/app/cftriage/${f}`, await (await fetchOk(`py/cftriage/${f}?v=${manifest.build}`)).text());
  }));
  py.FS.mkdirTree("/app/data/mimic-iv-ed-demo/ed");
  await Promise.all(manifest.mimic.map(async f => {
    const buf = new Uint8Array(await (await fetchOk(`data/mimic-iv-ed-demo/ed/${f}`)).arrayBuffer());
    py.FS.writeFile(`/app/data/mimic-iv-ed-demo/ed/${f}`, buf);
  }));

  // Studies persist in this browser's IndexedDB.
  py.FS.mkdirTree("/work");
  py.FS.mount(py.FS.filesystems.IDBFS, {}, "/work");
  await syncfs(true);

  self.cfProgress = (job, done, total, message) => postMessage({ type: "progress", job, done, total, message });
  py.runPython(`
import sys
sys.path.insert(0, "/app")
from cftriage import browser
from js import cfProgress
browser.init("/work", "/app/data/mimic-iv-ed-demo", cfProgress)
`);
  dispatch = py.globals.get("browser").dispatch;
})();

// One request at a time: Python is single-threaded and the IndexedDB sync
// must not interleave with writes.
let queue = Promise.resolve();
const MUTATES = new Set(["build_study", "analyze", "run_append"]);

onmessage = e => {
  const { id, op, args } = e.data;
  queue = queue.then(async () => {
    try {
      await ready;
      if (op === "analyze" && !matplotlibLoaded) {
        postMessage({ type: "progress", job: args._job, done: 0, total: 1, message: "Loading the chart library (first time only)…" });
        await py.loadPackage(["matplotlib"]);
        matplotlibLoaded = true;
      }
      const out = JSON.parse(dispatch(op, JSON.stringify(args || {})));
      if (MUTATES.has(op)) await syncfs(false);
      postMessage(out.error !== undefined ? { id, error: out.error } : { id, ok: out.ok });
    } catch (err) {
      postMessage({ id, error: String(err && err.message || err) });
    }
  });
};

ready.then(() => postMessage({ type: "ready" }), err => postMessage({ type: "failed", error: String(err && err.message || err) }));
