"""A deterministic scoring tool.

Letting the LLM eyeball a weighted average is unreliable and hard to audit.
Instead the Comparator agent calls this tool with the scores it assigned per
criterion, and the tool does the arithmetic. This is a good example of
"use a tool for anything that should not depend on model arithmetic."
"""

from typing import Dict, List


def compute_weighted_score(
    criterion_scores: Dict[str, float],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Compute a normalized weighted score.

    Args:
        criterion_scores: e.g. {"price": 7, "reliability": 9}
        weights: e.g. {"price": 0.4, "reliability": 0.6} — need not sum to 1,
            they are normalized here.

    Returns:
        A dict with the weighted total and the normalized weights actually used,
        so the agent can cite exactly how the number was derived.
    """
    total_weight = sum(weights.get(k, 0) for k in criterion_scores) or 1.0
    normalized = {k: weights.get(k, 0) / total_weight for k in criterion_scores}
    weighted_total = sum(
        criterion_scores[k] * normalized[k] for k in criterion_scores
    )
    return {
        "weighted_total": round(weighted_total, 2),
        "normalized_weights": {k: round(v, 3) for k, v in normalized.items()},
    }
