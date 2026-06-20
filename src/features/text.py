from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )


