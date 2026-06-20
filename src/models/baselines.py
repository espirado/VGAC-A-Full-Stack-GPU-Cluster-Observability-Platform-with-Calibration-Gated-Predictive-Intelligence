from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Tuple

from src.features.text import build_tfidf_vectorizer


def build_tfidf_logreg() -> Pipeline:
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")),
    ])


