"""
Multi-nail assessment aggregation utilities.
"""

from typing import Any, Dict, List
import numpy as np
from anemia_ai.config.constants import STATE_ANEMIA, STATE_INCONCLUSIVE, STATE_NO_ANEMIA


def aggregate_multi_nail_predictions(
    single_results: List[Dict[str, Any]],
    threshold: float,
    method: str = "mean",
) -> Dict[str, Any]:
    """
    Aggregates per-nail predictions from 2 to 4 images belonging to a single subject.

    Args:
        single_results: List of single-nail result dictionaries.
        threshold: Decision threshold for anemia classification.
        method: Aggregation strategy ('mean').

    Returns:
        Structured multi-nail assessment result dictionary.
    """
    valid_results = [r for r in single_results if r.get("state") in (STATE_ANEMIA, STATE_NO_ANEMIA)]

    if not valid_results:
        return {
            "success": True,
            "state": STATE_INCONCLUSIVE,
            "probability": None,
            "aggregation": method,
            "valid_images": 0,
            "total_images": len(single_results),
            "threshold": threshold,
            "description": "All nail images failed quality or physiological checks.",
            "per_image_results": single_results,
        }

    probs = [r["probability"] for r in valid_results if r.get("probability") is not None]
    if not probs:
        return {
            "success": True,
            "state": STATE_INCONCLUSIVE,
            "probability": None,
            "aggregation": method,
            "valid_images": 0,
            "total_images": len(single_results),
            "threshold": threshold,
            "description": "No valid calibrated probabilities extracted.",
            "per_image_results": single_results,
        }

    mean_p = round(float(np.mean(probs)), 4)
    state = STATE_ANEMIA if mean_p >= threshold else STATE_NO_ANEMIA
    desc = (
        f"Aggregated mean probability ({mean_p*100:.1f}%) across {len(valid_results)} "
        f"valid nail images indicates {'potential anemia' if state == STATE_ANEMIA else 'no anemia detected'}."
    )

    return {
        "success": True,
        "state": state,
        "probability": mean_p,
        "aggregation": method,
        "valid_images": len(valid_results),
        "total_images": len(single_results),
        "threshold": threshold,
        "description": desc,
        "per_image_results": single_results,
    }
