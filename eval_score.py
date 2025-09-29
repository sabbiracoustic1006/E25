from collections import Counter, defaultdict
from typing import Dict, Iterable, Tuple

Entity = Tuple[str, str, str]


def _build_counters(
    entities: Iterable[Entity],
    exclude_aspect: str,
) -> Dict[Tuple[str, str], Counter]:
    counter: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for category, aspect, value in entities:
        if aspect == exclude_aspect:
            continue
        counter[(category, aspect)][value] += 1
    return counter


def compute_competition_score(
    true_entities,
    pred_entities,
    beta: float = 0.2,
    exclude_aspect: str = "O",
):
    """Compute per-category F-beta scores.

    Parameters
    ----------
    true_entities, pred_entities:
        Iterable of ``(category, aspect_name, aspect_value)`` tuples.
    beta:
        Weighting for recall inside the F-score (defaults to 0.2).
    exclude_aspect:
        Aspect name to ignore during scoring, defaults to ``"O"``.

    Returns
    -------
    dict
        ``{"per_category": {cat: score, ...}, "overall_score": mean}``
    """

    true_counter = _build_counters(true_entities, exclude_aspect)
    pred_counter = _build_counters(pred_entities, exclude_aspect)

    categories = {
        cat for cat, _ in true_counter.keys()
    } | {
        cat for cat, _ in pred_counter.keys()
    }

    cat_scores: Dict[str, float] = {}
    for cat in sorted(categories):
        tp = fp = fn = 0

        aspects = {
            aspect
            for c, aspect in true_counter.keys()
            if c == cat
        } | {
            aspect
            for c, aspect in pred_counter.keys()
            if c == cat
        }

        if not aspects:
            cat_scores[cat] = 0.0
            continue

        for aspect in aspects:
            true_counts = true_counter.get((cat, aspect), Counter())
            pred_counts = pred_counter.get((cat, aspect), Counter())

            values = set(true_counts.keys()) | set(pred_counts.keys())
            for value in values:
                t_val = true_counts.get(value, 0)
                p_val = pred_counts.get(value, 0)
                matched = min(t_val, p_val)
                tp += matched
                fn += t_val - matched
                fp += p_val - matched

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = (beta ** 2) * precision + recall
        f_beta = (
            (1 + beta ** 2) * precision * recall / denom
            if denom > 0
            else 0.0
        )

        cat_scores[cat] = f_beta

    overall = sum(cat_scores.values()) / len(cat_scores) if cat_scores else 0.0

    return {"per_category": cat_scores, "overall_score": overall}


if __name__ == "__main__":
    true_ents = [
        ("0", "material", "ceramic"),
        ("0", "material", "ceramic"),
        ("0", "color", "red"),
        ("1", "type", "timing belt"),
        ("1", "type", "chain"),
    ]

    pred_ents = [
        ("0", "material", "ceramic"),
        ("0", "color", "blue"),
        ("0", "material", "fiber"),
        ("1", "type", "timing belt"),
    ]

    scores = compute_competition_score(true_ents, pred_ents, beta=0.2)
    print("Per-category scores:", scores["per_category"])
    print("Overall score:", scores["overall_score"])
