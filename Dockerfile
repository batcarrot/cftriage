# Hosted (public) version of the cftriage web app, e.g. a Hugging Face Space.
# Visitors bring their own API keys; see cftriage/web.py for the public mode.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Built-in real cases: the open-access MIMIC-IV-ED demo (ODbL).
RUN python -c "import urllib.request, os; os.makedirs('/app/data/mimic-iv-ed-demo/ed'); \
[urllib.request.urlretrieve(f'https://physionet.org/files/mimic-iv-ed-demo/2.2/ed/{f}.csv.gz', \
f'/app/data/mimic-iv-ed-demo/ed/{f}.csv.gz') for f in ('edstays', 'triage')]"

COPY cftriage ./cftriage

USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 \
    CFTRIAGE_PUBLIC=1 CFTRIAGE_MIMIC_DEMO=/app/data/mimic-iv-ed-demo PORT=7860
EXPOSE 7860
CMD ["python", "-m", "cftriage", "serve", "--workspace", "/tmp/cftriage", "--no-browser"]
