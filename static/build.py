"""Build the static site into _site/ (or the folder given as the first argument).

    python static/build.py [out_dir]

The site is cftriage/webui.html plus static/backend.js, which answers the
page's /api/* calls in the browser (see cftriage/browser.py). Python sources
and the MIMIC-IV-ED demo (open data, ODbL) are copied in for Pyodide.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIMIC_URL = "https://physionet.org/files/mimic-iv-ed-demo/2.2/ed/{}"
MIMIC_FILES = ["edstays.csv.gz", "triage.csv.gz"]


def main(out: Path) -> None:
    # Read the version without importing the package (no numpy needed here).
    __version__ = re.search(r'__version__ = "([^"]+)"', (ROOT / "cftriage" / "__init__.py").read_text())[1]
    build = time.strftime("%Y%m%d%H%M%S")
    if out.exists():
        shutil.rmtree(out)
    st = out / "static"
    (st / "py" / "cftriage").mkdir(parents=True)

    page = (ROOT / "cftriage" / "webui.html").read_text()
    anchor = '<script>\n"use strict";'
    assert page.count(anchor) == 1, "webui.html main script tag not found"
    page = page.replace(anchor, f'<script src="static/backend.js?v={build}"></script>\n{anchor}')
    (out / "index.html").write_text(page)

    backend = (ROOT / "static" / "backend.js").read_text()
    (st / "backend.js").write_text(backend.replace("__VERSION__", __version__).replace("__BUILD__", build))
    shutil.copy(ROOT / "static" / "worker.js", st / "worker.js")

    sources = sorted(p.name for p in (ROOT / "cftriage").glob("*.py"))
    for name in sources:
        shutil.copy(ROOT / "cftriage" / name, st / "py" / "cftriage" / name)

    data = st / "data" / "mimic-iv-ed-demo" / "ed"
    data.mkdir(parents=True)
    cache = ROOT / ".cache" / "mimic-iv-ed-demo" / "ed"
    for f in MIMIC_FILES:
        if not (cache / f).exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MIMIC_URL.format(f), cache / f)
        shutil.copy(cache / f, data / f)

    (st / "manifest.json").write_text(json.dumps({"build": build, "python": sources, "mimic": MIMIC_FILES}))
    (out / ".nojekyll").write_text("")
    print(f"static site written to {out} (build {build})")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "_site").resolve())
