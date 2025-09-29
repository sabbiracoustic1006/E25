"""Competition scoring helper with per-record accounting."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple, Union


_GLOBAL_RECORD_ID = "__GLOBAL__"


EntityLike = Union[
    Tuple[str, str, str],
    Tuple[str, str, str, str],
    Dict[str, str],
]


def _normalise_entities(
    entities: Iterable[EntityLike],
    exclude_aspect: str,
) -> List[Tuple[str, str, str, str]]:
    """Normalise heterogeneous entity records to (record, category, aspect, value)."""

    normalised: List[Tuple[str, str, str, str]] = []
    for item in entities:
        record_id: str | None
        category: str | None
        aspect: str | None
        value: str

        if isinstance(item, dict):
            record_id = item.get("record_id")
            category = item.get("category")
            aspect = item.get("aspect_name") or item.get("aspect")
            value = (
                item.get("aspect_value")
                or item.get("value")
                or item.get("span")
                or ""
            )
        elif isinstance(item, (tuple, list)):
            if len(item) == 4:
                record_id, category, aspect, value = item  # type: ignore[misc]
            elif len(item) == 3:
                category, aspect, value = item  # type: ignore[misc]
                record_id = None
            else:
                raise ValueError(
                    "Entity tuples must contain either 3 (category, aspect, value) "
                    "or 4 (record_id, category, aspect, value) items."
                )
        else:
            raise TypeError(
                "Entities must be dictionaries or tuples/lists describing the span."
            )

        if aspect is None or category is None:
            continue
        if aspect == exclude_aspect:
            continue

        record_key = record_id if record_id is not None else _GLOBAL_RECORD_ID
        normalised.append((str(record_key), str(category), str(aspect), str(value).strip()))

    return normalised


def compute_competition_score(
    true_entities: Sequence[EntityLike],
    pred_entities: Sequence[EntityLike],
    beta: float = 0.2,
    exclude_aspect: str = "O",
) -> Dict[str, object]:
    """
    Compute the competition score with per-record matching.

    Parameters
    ----------
    true_entities / pred_entities
        Iterable collections describing extracted spans. Supported element shapes:
          * (record_id, category, aspect_name, value/span)
          * (category, aspect_name, value/span)
          * dict with keys including ``category`` / ``aspect_name`` / ``span`` and
            optional ``record_id``.
    beta
        F-beta weight (default 0.2).
    exclude_aspect
        Aspect name to ignore (default "O").

    Returns
    -------
    dict
        Contains ``per_category`` scores, the ``overall_score`` and aggregate
        ``global_counts`` for tp/fp/fn.
    """

    beta_sq = beta ** 2

    true_normalised = _normalise_entities(true_entities, exclude_aspect)
    pred_normalised = _normalise_entities(pred_entities, exclude_aspect)

    true_by_record: Dict[str, Dict[Tuple[str, str], Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    pred_by_record: Dict[str, Dict[Tuple[str, str], Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    categories = set()

    for record_id, category, aspect, value in true_normalised:
        true_by_record[record_id][(category, aspect)][value] += 1
        categories.add(category)
    for record_id, category, aspect, value in pred_normalised:
        pred_by_record[record_id][(category, aspect)][value] += 1
        categories.add(category)

    record_ids = set(true_by_record.keys()) | set(pred_by_record.keys())

    tp_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    true_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    pred_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    global_tp = 0
    global_fp = 0
    global_fn = 0

    for r_idx, record_id in enumerate(record_ids):
        true_map = true_by_record.get(record_id, {})
        pred_map = pred_by_record.get(record_id, {})
        aspect_keys = set(true_map.keys()) | set(pred_map.keys())
        if r_idx < 5:  # Print debug info for first few records
            print("----" * 10)
            print("Record ID:", record_id)
            print(true_map, pred_map, aspect_keys)

        for key in aspect_keys:
            t_counter = true_map.get(key, Counter())
            p_counter = pred_map.get(key, Counter())
            value_keys = set(t_counter.keys()) | set(p_counter.keys())

            tp = sum(min(t_counter.get(val, 0), p_counter.get(val, 0)) for val in value_keys)
            true_total = sum(t_counter.values())
            pred_total = sum(p_counter.values())

            fp = pred_total - tp
            fn = true_total - tp

            tp_counts[key] += tp
            true_counts[key] += true_total
            pred_counts[key] += pred_total

            global_tp += tp
            global_fp += fp
            global_fn += fn

    total_true_by_cat = defaultdict(int)
    for (category, _aspect), total in true_counts.items():
        total_true_by_cat[category] += total

    cat_scores: Dict[str, float] = {}
    for category in sorted(categories):
        cat_true_total = total_true_by_cat.get(category, 0)
        if cat_true_total == 0:
            cat_scores[category] = 0.0
            continue

        weighted_f_sum = 0.0
        for (cat_key, aspect) in (
            (cat, asp) for (cat, asp) in true_counts if cat == category
        ):
            tp = tp_counts.get((cat_key, aspect), 0)
            true_total = true_counts.get((cat_key, aspect), 0)
            pred_total = pred_counts.get((cat_key, aspect), 0)

            precision = tp / pred_total if pred_total else 0.0
            recall = tp / true_total if true_total else 0.0
            denom = beta_sq * precision + recall
            f_beta = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0

            weight = true_total / cat_true_total if cat_true_total else 0.0
            weighted_f_sum += weight * f_beta

        cat_scores[category] = weighted_f_sum

    overall = sum(cat_scores.values()) / len(cat_scores) if cat_scores else 0.0

    return {
        "per_category": cat_scores,
        "overall_score": overall,
        "global_counts": {"tp": global_tp, "fp": global_fp, "fn": global_fn},
    }


if __name__ == "__main__":
    # ————————————————————————————
    # Example 1 (basic smoke-test)
    # ————————————————————————————
    true_ents = [
        ("rec-1", "0", "material", "ceramic"),
        ("rec-1", "0", "material", "ceramic"),
        ("rec-1", "0", "color", "red"),
        ("rec-2", "1", "type", "timing belt"),
        ("rec-2", "1", "type", "chain"),
    ]
    pred_ents = [
        ("rec-1", "0", "material", "ceramic"),
        ("rec-1", "0", "color", "blue"),
        ("rec-1", "0", "material", "fiber"),
        ("rec-2", "1", "type", "timing belt"),
    ]

    scores = compute_competition_score(true_ents, pred_ents, beta=0.2)
    print("=== Example 1 (smoke-test) ===")
    print("Per-category scores:", scores["per_category"])
    print("Overall score:", scores["overall_score"])
    print("Global counts:", scores["global_counts"])
    print()

    # ————————————————————————————
    # Example 2 (BMW/Mercedes/Audi case)
    # ————————————————————————————
    true_ents2 = [
        # Sample 1
        ("rec-1", "2", "Produktart", "Bremsbeläge"),
        ("rec-1", "2", "Produktart", "Satz"),
        ("rec-1", "2", "Einbauposition", "vorne"),
        ("rec-1", "2", "Einbauposition", "hinten"),
        ("rec-1", "2", "Hersteller", "ATE"),
        # Sample 2
        ("rec-2", "1", "Produktart", "Bremsbeläge"),
        ("rec-2", "1", "Produktart", "Satz"),
        ("rec-2", "1", "Einbauposition", "vorne"),
        ("rec-2", "1", "Hersteller", "Bosch"),
        # Sample 3
        ("rec-3", "1", "Produktart", "Bremsbeläge"),
        ("rec-3", "1", "Produktart", "Satz"),
        ("rec-3", "1", "Einbauposition", "hinten"),
        ("rec-3", "1", "Hersteller", "ATE"),
    ]

    pred_ents2 = [
        # Sample 1
        ("rec-1", "2", "Produktart", "Bremsbeläge"),
        ("rec-1", "2", "Produktart", "Satz"),
        ("rec-1", "2", "Hersteller", "Bosch"),   # wrong (should be ATE)
        # Sample 2
        ("rec-2", "1", "Produktart", "Bremsbeläge"),
        ("rec-2", "1", "Produktart", "Satz"),
        ("rec-2", "1", "Hersteller", "ATE"),    # wrong (should be Bosch)
        # Sample 3
        ("rec-3", "1", "Produktart", "Bremsbeläge"),
        ("rec-3", "1", "Einbauposition", "vorne"),  # wrong (should be hinten)
        ("rec-3", "1", "Hersteller", "Bosch"),      # wrong (should be ATE)
    ]

    scores2 = compute_competition_score(true_ents2, pred_ents2, beta=0.2)
    print("=== Example 2 (cross-sample BMW/Mercedes/Audi) ===")
    print("Per-category scores:", scores2["per_category"])
    print("Overall score:", scores2["overall_score"])
    print("Global counts:", scores2["global_counts"])