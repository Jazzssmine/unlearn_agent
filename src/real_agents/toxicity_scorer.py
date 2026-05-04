from __future__ import annotations
from functools import lru_cache
from typing import Dict

@lru_cache(maxsize=1)
def _get_detoxify_model():
    from detoxify import Detoxify
    return Detoxify("original")

DETOXIFY_KEYS = (
    "identity_attack",
    "insult",
    "obscene",
    "severe_toxicity",
    "threat",
    "toxicity",
)

def score_text_toxicity_breakdown(text: str) -> Dict[str, float]:
    """
    Returns Detoxify category scores in [0, 1].
    """
    if not text or not text.strip():
        return {k: 0.0 for k in DETOXIFY_KEYS}
    model = _get_detoxify_model()
    raw: Dict[str, float] = model.predict(text.strip())
    out: Dict[str, float] = {}
    for k in DETOXIFY_KEYS:
        val = float(raw.get(k, 0.0))
        out[k] = max(0.0, min(1.0, val))
    return out

def score_text_toxicity(text: str) -> float:
    """
    Returns a toxicity score in [0, 1] using Detoxify.
    Loads the model once per process.
    """
    return float(score_text_toxicity_breakdown(text).get("toxicity", 0.0))