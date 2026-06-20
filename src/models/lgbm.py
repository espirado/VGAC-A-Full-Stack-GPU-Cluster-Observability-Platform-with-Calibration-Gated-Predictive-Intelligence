from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.features.text import build_tfidf_vectorizer


def build_tfidf_lgbm() -> Pipeline:
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        (
            "clf",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.1,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ])


