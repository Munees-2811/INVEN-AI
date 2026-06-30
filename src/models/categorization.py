"""
AI product categorization.

Trains a TF-IDF + Logistic Regression text classifier on product names so new /
uncategorized products can be auto-assigned to a category. This is a genuine ML
model (persisted via the MLOps registry) with an LLM fallback for zero-shot
labelling when training data is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline


@dataclass
class CategorizationModel:
    pipeline: Pipeline
    classes: list[str]
    cv_accuracy: float

    def predict(self, names: list[str]) -> list[dict]:
        proba = self.pipeline.predict_proba(names)
        out = []
        for name, row in zip(names, proba):
            idx = int(np.argmax(row))
            out.append(
                {
                    "product_name": name,
                    "predicted_category": self.classes[idx],
                    "confidence": round(float(row[idx]), 3),
                }
            )
        return out


def train_categorizer(products: pd.DataFrame) -> CategorizationModel:
    """Train the text categorizer on the existing labelled catalogue."""
    X = products["product_name"].astype(str).tolist()
    y = products["category"].astype(str).tolist()

    # Combine word n-grams (capture the head noun, e.g. "Cola" → Beverages) with
    # character n-grams (robust to spelling / brand prefixes / unit suffixes).
    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                                 sublinear_tf=True, min_df=1)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                                 sublinear_tf=True, min_df=1)),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=10.0)),
        ]
    )

    n_splits = min(5, max(2, int(min(pd.Series(y).value_counts()))))
    try:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv)
        acc = float(scores.mean())
    except Exception:
        acc = float("nan")

    pipe.fit(X, y)
    return CategorizationModel(pipeline=pipe, classes=list(pipe.named_steps["clf"].classes_), cv_accuracy=acc)


def suggest_category(model: CategorizationModel, product_name: str) -> dict:
    return model.predict([product_name])[0]
