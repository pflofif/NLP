from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

def make_svm_word_pipeline(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 30_000,
    class_weight: str | None = None,
    C: float = 1.0,
    sublinear_tf: bool = True,
) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
        )),
        ("clf", LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=42,
            max_iter=2000,
        )),
    ])


def make_svm_char_pipeline(
    char_ngram_range: tuple[int, int] = (3, 5),
    max_features: int = 50_000,
    class_weight: str | None = None,
    C: float = 1.0,
    analyzer: str = "char_wb",
) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=char_ngram_range,
            sublinear_tf=True,
            max_features=max_features,
        )),
        ("clf", LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=42,
            max_iter=2000,
        )),
    ])


def make_svm_word_char_pipeline(
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    word_max_features: int = 30_000,
    char_max_features: int = 50_000,
    class_weight: str | None = None,
    C: float = 1.0,
    char_analyzer: str = "char_wb",
) -> Pipeline:
    feature_union = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            sublinear_tf=True,
            max_features=word_max_features,
        )),
        ("char", TfidfVectorizer(
            analyzer=char_analyzer,
            ngram_range=char_ngram_range,
            sublinear_tf=True,
            max_features=char_max_features,
        )),
    ])
    return Pipeline([
        ("features", feature_union),
        ("clf", LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=42,
            max_iter=2000,
        )),
    ])

def run_logreg_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: list[str],
    text_col: str = "pair_text",
    label_col: str = "label",
    ngram_range: tuple[int, int] = (1, 2),
    class_weight: str | None = "balanced",
    max_features: int = 30_000,
) -> dict:
    from classification_baseline import make_pipeline, evaluate_pipeline

    pipe = make_pipeline(
        ngram_range=ngram_range,
        max_features=max_features,
        class_weight=class_weight,
    )
    pipe.fit(train_df[text_col].tolist(), train_df[label_col].tolist())

    return {
        "pipe": pipe,
        "val":  evaluate_pipeline(pipe, val_df[text_col].tolist(),  val_df[label_col].tolist(),  label_names),
        "test": evaluate_pipeline(pipe, test_df[text_col].tolist(), test_df[label_col].tolist(), label_names),
    }


def run_linear_svc(
    pipe: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: list[str],
    text_col: str = "pair_text",
    label_col: str = "label",
) -> dict:
    pipe.fit(train_df[text_col].tolist(), train_df[label_col].tolist())

    def _eval(df: pd.DataFrame) -> dict:
        X = df[text_col].tolist()
        y = df[label_col].tolist()
        y_pred = pipe.predict(X)
        return {
            "accuracy":    round(accuracy_score(y, y_pred), 4),
            "macro_f1":    round(f1_score(y, y_pred, average="macro", labels=[0, 1], zero_division=0), 4),
            "report":      classification_report(y, y_pred, labels=[0, 1], target_names=label_names, zero_division=0),
            "conf_matrix": confusion_matrix(y, y_pred),
            "y_pred":      y_pred,
            "y_true":      np.array(y),
            "scores":      pipe.decision_function(X),
        }

    return {
        "pipe": pipe,
        "val":  _eval(val_df),
        "test": _eval(test_df),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    title: str = "Confusion matrix",
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
