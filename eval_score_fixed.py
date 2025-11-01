def calculate_fbeta(pred, target, beta=0.2):
    # pred, target are sets of tuples like {(r_id, span), ...}
    tp = pred & target  # true positives
    fp = pred - target  # false positives
    fn = target - tp    # false negatives

    beta2 = beta ** 2
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0

    if precision == 0 and recall == 0:
        fbeta = 0.0
    else:
        fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall)

    return fbeta


def compute_competition_score(all_targets, all_preds, beta=0.2):
    """
    Compute per-category and overall weighted F-beta scores.

    Args:
        all_targets (list[dict]): Ground truth records with columns like ['category', 'aspect_name', ...].
        all_preds (list[dict]): Prediction records with same structure as targets.
        beta (float): Beta value for F-beta computation. Default is 0.2 (precision-focused).

    Returns:
        dict: {
            'per_category': {'1': float, '2': float},
            'overall_score': float
        }
    """
    import pandas as pd

    # Convert to DataFrames
    df_target = pd.DataFrame(all_targets)
    df_pred = pd.DataFrame(all_preds)

    results = {'per_category': {}}
    category_scores = []

    # Iterate over both categories
    for cat in ['1', '2']:
        df_target_cat = df_target[df_target['category'] == cat]
        df_pred_cat = df_pred[df_pred['category'] == cat]

        cat_score = 0.0
        total = len(df_target_cat)

        # Iterate over all aspect names within this category
        for aspect_name in df_target_cat['aspect_name'].unique():
            target_rows = df_target_cat[df_target_cat['aspect_name'] == aspect_name]
            pred_rows = df_pred_cat[df_pred_cat['aspect_name'] == aspect_name]

            # Convert rows into sets of tuples (excluding metadata columns)
            target = set(target_rows.drop(columns=['category', 'aspect_name']).apply(tuple, axis=1))
            pred = set(pred_rows.drop(columns=['category', 'aspect_name']).apply(tuple, axis=1))

            # Weighted F-beta for this aspect
            weight = len(target_rows) / total
            cat_score += weight * calculate_fbeta(pred, target, beta=beta)

        results['per_category'][cat] = cat_score
        category_scores.append(cat_score)

    # Compute overall mean across categories
    overall = sum(category_scores) / len(category_scores)
    results['overall_score'] = overall

    return results
