import pandas as pd
from sklearn.model_selection import StratifiedKFold

def convert_tagged_to_aspect(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    rows, cur = [], (None,None,None, None)  # (rid,cat,title,asp)
    buf = []
    for _,r in df.iterrows():
        rid,cat,title,tok,tag = r["Record Number"],r["Category"],r["Title"],r["Token"],r["Tag"]
        if rid != cur[0]:
            if cur[3] and buf:
                rows.append([*cur, " ".join(buf)])
            cur = (rid,cat,title,None); buf=[]
        if tag:
            if cur[3] and buf:
                rows.append([*cur, " ".join(buf)])
            cur = (cur[0],cur[1],cur[2], tag); buf=[tok]
        else:
            if cur[3]:
                buf.append(tok)
    if cur[3] and buf:
        rows.append([*cur, " ".join(buf)])
    return pd.DataFrame(rows, columns=[
        "Record Number","Category","Title","Aspect Name","Aspect Value"
    ])

# 2) Stratified k-fold with “rare” bucket (as before)
def stratified_kfold_split(df, n_splits=5, random_state=42):
    sumdf = (
        df.groupby("Record Number")
          .agg({"Category":"first",
                "Aspect Name": lambda x: x.mode()[0] if not x.mode().empty else "UNKNOWN"})
          .reset_index()
    )
    sumdf["key"] = sumdf["Category"] + "_" + sumdf["Aspect Name"]
    vc = sumdf["key"].value_counts()
    rare = vc[vc < n_splits].index
    sumdf.loc[sumdf["key"].isin(rare), "key"] = "rare"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sumdf["fold"] = -1
    for fold, (_,vidx) in enumerate(skf.split(sumdf, sumdf["key"])):
        sumdf.loc[vidx,"fold"] = fold
    return df.merge(sumdf[["Record Number","fold"]], on="Record Number", how="left")

def resolve_word_labels(tokens, labels, offset_mapping):
    word_preds = []
    cur_labels = []
    cur_offsets = []

    for i, (tok, label, (start, end)) in enumerate(zip(tokens, labels, offset_mapping)):
        if tok in ['[CLS]', '[SEP]'] or (start == 0 and end == 0):
            continue  # Skip special tokens

        is_new_word = tok.startswith('▁')

        if is_new_word and cur_offsets:
            # Finalize previous word
            final_label = next((l for l in cur_labels if l != 'O'), 'O')
            if 'B-O' in cur_labels or 'I-O' in cur_labels:
                final_label = 'B-O'
            # print(cur_labels)
            word_preds.append({
                'entity': final_label,
                'start': cur_offsets[0][0],
                'end': cur_offsets[-1][1]
            })
            cur_labels, cur_offsets = [], []

        cur_labels.append(label)
        cur_offsets.append((start, end))

    # Handle last word
    if cur_offsets:
        # final_label = next((l for l in cur_labels if l != 'O'), 'O')
        final_label = next((l for l in cur_labels if l != 'O'), 'O')
        # print(cur_labels)
        if 'B-O' in cur_labels or 'I-O' in cur_labels:
            final_label = 'B-O'
        # if 'O' in cur_labels and final_label != 'O':
        #     final_label = 'O'
        word_preds.append({
            'entity': final_label,
            'start': cur_offsets[0][0],
            'end': cur_offsets[-1][1]
        })

    return word_preds


def merge_spans(preds):
    merged = []
    cur = None
    for tok in preds:
        label = tok["entity"]
        if label.startswith("B-"):
            if cur: merged.append(cur)
            cur = {
                "aspect_name": label[2:],
                "start": tok["start"],
                "end": tok["end"]
            }
        elif label.startswith("I-") and cur and label[2:]==cur["aspect_name"]:
            cur["end"] = tok["end"]
        else:
            if cur:
                merged.append(cur)
                cur = None
    if cur: merged.append(cur)
    return merged