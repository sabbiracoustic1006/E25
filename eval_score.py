from collections import Counter, defaultdict

def compute_competition_score(true_entities,
                              pred_entities,
                              beta: float = 0.2,
                              exclude_aspect: str = "O"):
    """
    true_entities, pred_entities: lists of (category, aspect_name, aspect_value)
    beta: weight for F_beta
    exclude_aspect: aspect_name to ignore (default "O")
    Returns: dict with per-category score and overall mean score
    """
    # 1) build counters: for each (cat,asp) count of values
    true_counter = defaultdict(Counter)
    pred_counter = defaultdict(Counter)
    categories = set()
    for cat, asp, val in true_entities:
        if asp == exclude_aspect: continue
        true_counter[(cat, asp)][val] += 1
        categories.add(cat)
    for cat, asp, val in pred_entities:
        if asp == exclude_aspect: continue
        pred_counter[(cat, asp)][val] += 1
        categories.add(cat)

    # 2) for each category compute total true spans (for weighting)
    total_true_by_cat = defaultdict(int)
    for (cat, asp), ctr in true_counter.items():
        total_true_by_cat[cat] += sum(ctr.values())

    # 3) compute per-category weighted F
    cat_scores = {}
    for cat in sorted(categories):
        # collect all aspects for this category
        asps = {asp for (c, asp) in true_counter.keys() if c == cat}
        if not asps:
            cat_scores[cat] = 0.0
            continue

        cat_true_total = total_true_by_cat[cat]
        weighted_f_sum = 0.0

        for asp in asps:
            tcnts = true_counter[(cat, asp)]
            pcnts = pred_counter.get((cat, asp), Counter())

            true_total = sum(tcnts.values())
            pred_total = sum(pcnts.values())

            # true positives = sum of min(count_true[val], count_pred[val])
            tp = sum(min(tcnts[v], pcnts.get(v, 0)) for v in tcnts)

            if true_total == 0 or pred_total == 0:
                f_beta = 0.0
            else:
                precision = tp / pred_total
                recall    = tp / true_total
                denom = (beta**2) * precision + recall
                f_beta = ((1 + beta**2) * precision * recall / denom) if denom>0 else 0.0

            weight = true_total / cat_true_total
            weighted_f_sum += weight * f_beta

        cat_scores[cat] = weighted_f_sum

    # 4) mean across categories
    if cat_scores:
        overall = sum(cat_scores.values()) / len(cat_scores)
    else:
        overall = 0.0

    return {"per_category": cat_scores, "overall_score": overall}


if __name__ == "__main__":
    # ————————————————
    # EXAMPLE USAGE
    # ————————————————
    # Suppose your gold file contained:
    true_ents = [
        ("0", "material", "ceramic"),
        ("0", "material", "ceramic"),
        ("0", "color",    "red"),
        ("1","type",     "timing belt"),
        ("1","type",     "chain"),
    ]

    # And your model predicted:
    pred_ents = [
        ("0", "material", "ceramic"),
        ("0", "color",    "blue"),
        ("0", "material", "fiber"),
        ("1","type",     "timing belt"),
    ]

    scores = compute_competition_score(true_ents, pred_ents, beta=0.2)
    print("Per-category scores:", scores["per_category"])
    print("Overall score:    ", scores["overall_score"])
