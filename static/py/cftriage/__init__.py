"""cftriage: counterfactual demographic testing of LLM emergency triage."""

from .stats import analyze, contrast, omnibus_test, power_simulation, prepare  # noqa: F401
from .vignettes import build_vignettes, load_mimic_ed_demo  # noqa: F401

__version__ = "0.1.0"
