"""Synthetic cases and simulated models with KNOWN behaviour.

Use these to check that the statistics do what they claim before trusting
them on real models: a noise-only model should be flagged about 5% of the
time at alpha = 0.05, and a model with a planted bias should be caught.

Nothing here is real patient data.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

_COMPLAINTS = {
    "chest_pain": ["Chest pain", "Chest pain, diaphoresis", "Chest pressure", "Palpitations"],
    "neuro": ["Headache", "Dizziness", "Left sided weakness", "Syncope", "Blurred vision, headache"],
    "respiratory": ["Shortness of breath", "Cough, fever", "Wheezing", "Dyspnea on exertion"],
    "abdominal": ["Abd pain", "Nausea/vomiting", "RLQ abdominal pain", "Diarrhea"],
    "trauma": ["Fall", "MVC", "S/p fall, head strike", "Hand laceration"],
    "psych": ["Anxiety", "Suicidal ideation", "Etoh intoxication"],
    "extremity_pain": ["Back pain", "Knee pain", "Ankle injury", "Leg swelling"],
    "general": ["Fever", "Weakness", "Fatigue", "Rash"],
}
_ESI_P = np.array([0.06, 0.33, 0.53, 0.075, 0.005])


def synthetic_cases(n: int, seed: int = 0, esi_probs=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    esi = rng.choice(np.arange(1, 6), size=n, p=esi_probs if esi_probs is not None else _ESI_P)
    cats = rng.choice(list(_COMPLAINTS), size=n)
    sev = (3 - esi)  # >0 sicker
    rows = []
    for i in range(n):
        s = sev[i]
        rows.append({
            "case_id": f"syn{seed}_{i:05d}",
            "chief_complaint": rng.choice(_COMPLAINTS[cats[i]]),
            "heart_rate": int(np.clip(rng.normal(85 + 12 * s, 12), 40, 180)),
            "resp_rate": int(np.clip(rng.normal(17 + 2.5 * s, 2), 8, 40)),
            "spo2": int(np.clip(rng.normal(98 - 2 * max(s, 0), 1.5), 70, 100)),
            "sbp": int(np.clip(rng.normal(132 - 6 * max(s, 0), 18), 60, 230)),
            "dbp": int(np.clip(rng.normal(78, 12), 30, 130)),
            "temperature": round(float(np.clip(rng.normal(98.4 + 0.4 * max(s, 0), 0.6), 95, 105)), 1),
            "pain": int(rng.integers(0, 11)),
            "esi_true": int(esi[i]),
            "age": int(rng.integers(18, 90)),
            "category": cats[i],
        })
    return pd.DataFrame(rows)


def simulate_results(cases: pd.DataFrame, model: str = "sim", n_samples: int = 10,
                     noise: float = 0.25, female_shift: float = 0.0,
                     affected_share: float = 1.0, nonbinary_shift: float = 0.0,
                     unlabeled_shift: float = 0.0, instability: float = 0.0,
                     model_disagreement: float = 0.4,
                     conditions=("male", "female", "unlabeled"),
                     reference: str = "male", seed: int = 0) -> pd.DataFrame:
    """Generate results in the same format as runner.run().

    noise            P(a sample moves one level up or down from the case's level)
    female_shift     P(a female-labelled sample moves one level LESS urgent)
    affected_share   share of cases the female shift applies to
    instability      P(a (case, non-reference label) cell gets a fixed random
                     +/-1 offset) - label sensitivity with no direction
    model_disagreement  P(the model's base level for a case differs by one
                     from the nurse ESI) - unrelated to labels
    """
    rng = np.random.default_rng(seed)
    n = len(cases)
    base = cases["esi_true"].to_numpy().astype(int)
    off = rng.choice([-1, 0, 1], size=n,
                     p=[model_disagreement / 2, 1 - model_disagreement, model_disagreement / 2])
    base = np.clip(base + off, 1, 5)
    affected = rng.random(n) < affected_share
    shifts = {"female": female_shift, "nonbinary": nonbinary_shift,
              "unlabeled": unlabeled_shift}
    rows = []
    t0 = time.time()
    for i, cid in enumerate(cases["case_id"].astype(str)):
        for cond in conditions:
            cell_off = 0
            if cond != reference and instability > 0 and rng.random() < instability:
                cell_off = rng.choice([-1, 1])
            for s in range(n_samples):
                y = base[i] + cell_off
                if rng.random() < noise:
                    y += rng.choice([-1, 1])
                sp = shifts.get(cond, 0.0)
                if sp and (cond != "female" or affected[i]) and rng.random() < sp:
                    y += 1
                y = int(np.clip(y, 1, 5))
                rows.append({"model": model, "case_id": cid, "condition": cond,
                             "sample": s, "response": json.dumps({"esi": y}),
                             "esi": y, "error": None, "t": t0})
    return pd.DataFrame(rows)


class SimulatedClient:
    """Drop-in client for runner.run(), for testing the full pipeline offline.
    The base level comes from a hash of the clinical text, so it is identical
    across conditions; the female label adds a configurable shift."""

    def __init__(self, noise: float = 0.25, female_shift: float = 0.0, seed: int = 0):
        self.noise, self.female_shift = noise, female_shift
        self.rng = np.random.default_rng(seed)

    def __call__(self, system, user, temperature, max_tokens):
        clinical = "\n".join(l for l in user.split("\n")
                             if not re.match(r"(Sex|Patient):", l))
        h = int(hashlib.sha256(clinical.encode()).hexdigest(), 16)
        y = 1 + h % 5
        if self.rng.random() < self.noise:
            y += int(self.rng.choice([-1, 1]))
        if "Female" in user or "woman" in user:
            if self.rng.random() < self.female_shift:
                y += 1
        return json.dumps({"esi": int(np.clip(y, 1, 5))})
