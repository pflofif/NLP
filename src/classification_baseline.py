
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
def build_pair_dataset(
    df: pd.DataFrame,
    queries_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    n_neg: int = 15,
    seed: int = 42,
    text_col: str = "text_v2",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    doc_lookup = df.set_index("doc_id")[text_col].fillna("").astype(str).to_dict()
    all_doc_ids = list(doc_lookup.keys())

    rows = []
    for _, qrow in queries_df.iterrows():
        qid = qrow["query_id"]
        qtext = qrow["query_text"]

        rel_doc_ids = set(
            labels_df[labels_df["query_id"] == qid]["doc_id"].tolist()
        )

        for did in rel_doc_ids:
            if did in doc_lookup:
                rows.append({
                    "query_id":   qid,
                    "doc_id":     did,
                    "query_text": qtext,
                    "doc_text":   doc_lookup[did],
                    "pair_text":  qtext + " " + doc_lookup[did],
                    "label":      1,
                })

        neg_pool = [d for d in all_doc_ids if d not in rel_doc_ids]
        n_sample = min(n_neg, len(neg_pool))
        neg_ids = rng.choice(neg_pool, size=n_sample, replace=False)
        for did in neg_ids:
            rows.append({
                "query_id":   qid,
                "doc_id":     did,
                "query_text": qtext,
                "doc_text":   doc_lookup[did],
                "pair_text":  qtext + " " + doc_lookup[did],
                "label":      0,
            })

    return pd.DataFrame(rows).reset_index(drop=True)


def split_pairs_by_query(
    pairs_df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    query_ids = pairs_df["query_id"].unique().tolist()
    rng.shuffle(query_ids)

    n = len(query_ids)
    n_val  = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))

    test_qs  = set(query_ids[:n_test])
    val_qs   = set(query_ids[n_test: n_test + n_val])
    train_qs = set(query_ids[n_test + n_val:])

    train = pairs_df[pairs_df["query_id"].isin(train_qs)].reset_index(drop=True)
    val   = pairs_df[pairs_df["query_id"].isin(val_qs)].reset_index(drop=True)
    test  = pairs_df[pairs_df["query_id"].isin(test_qs)].reset_index(drop=True)
    return train, val, test


def split_pairs_by_doc(
    pairs_df: pd.DataFrame,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def assign(doc_id: str) -> str:
        if doc_id in test_ids:
            return "test"
        if doc_id in val_ids:
            return "val"
        return "train"

    pairs_df = pairs_df.copy()
    pairs_df["split"] = pairs_df["doc_id"].apply(assign)

    train = pairs_df[pairs_df["split"] == "train"].reset_index(drop=True)
    val   = pairs_df[pairs_df["split"] == "val"].reset_index(drop=True)
    test  = pairs_df[pairs_df["split"] == "test"].reset_index(drop=True)
    return train, val, test

def make_pipeline(
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int = 30000,
    class_weight: str | None = None,
    max_iter: int = 500,
) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            sublinear_tf=True,
            max_features=max_features,
        )),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42,
            solver="lbfgs",
        )),
    ])

def evaluate_pipeline(
    pipe: Pipeline,
    X: list[str],
    y: list[int],
    label_names: list[str] | None = None,
) -> dict:
    y_pred = pipe.predict(X)
    return {
        "accuracy":    round(accuracy_score(y, y_pred), 4),
        "macro_f1":    round(f1_score(y, y_pred, average="macro", labels=[0, 1], zero_division=0), 4),
        "report":      classification_report(y, y_pred, labels=[0, 1], target_names=label_names, zero_division=0),
        "conf_matrix": confusion_matrix(y, y_pred),
        "y_pred":      y_pred,
    }

def get_top_features(
    pipe: Pipeline,
    n: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    tfidf = pipe.named_steps["tfidf"]
    clf   = pipe.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    classes = clf.classes_

    result = {}
    for i, cls in enumerate(classes):
        coefs = clf.coef_[i] if len(classes) > 2 else clf.coef_[0]
        if len(classes) == 2 and i == 0:
            coefs = -coefs  

        top_idx  = np.argsort(coefs)[-n:][::-1]
        result[f"class_{cls}_top"] = [
            (feature_names[j], round(float(coefs[j]), 4)) for j in top_idx
        ]
        bot_idx  = np.argsort(coefs)[:n]
        result[f"class_{cls}_bot"] = [
            (feature_names[j], round(float(coefs[j]), 4)) for j in bot_idx
        ]
    return result
