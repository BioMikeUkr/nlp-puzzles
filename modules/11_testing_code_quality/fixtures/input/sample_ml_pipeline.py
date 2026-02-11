"""
Sample ML pipeline for testing exercises.
Simple text classification using sklearn.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def create_pipeline(max_features: int = 1000, C: float = 1.0) -> Pipeline:
    """Create a text classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features)),
        ("clf", LogisticRegression(C=C, max_iter=200, random_state=42)),
    ])


def train_pipeline(
    pipeline: Pipeline,
    texts: list[str],
    labels: list[int],
) -> Pipeline:
    """Train the pipeline on texts and labels."""
    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels must have same length: {len(texts)} != {len(labels)}"
        )
    if len(texts) == 0:
        raise ValueError("Cannot train on empty data")
    pipeline.fit(texts, labels)
    return pipeline


def predict(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    """Predict labels for texts."""
    return pipeline.predict(texts)


def evaluate(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    """Evaluate predictions and return metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def preprocess_texts(texts: list[str]) -> list[str]:
    """Basic text preprocessing."""
    processed = []
    for text in texts:
        text = text.lower().strip()
        # Remove URLs
        import re
        text = re.sub(r"http\S+", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        processed.append(text)
    return processed


# Sample data for testing
SAMPLE_TEXTS = [
    "This product is amazing, I love it!",
    "Terrible quality, waste of money.",
    "Great customer service, very helpful.",
    "The item broke after one day, very disappointed.",
    "Absolutely fantastic, exceeded expectations!",
    "Worst purchase ever, do not buy.",
    "Good value for the price, recommended.",
    "Poor packaging, item arrived damaged.",
]

SAMPLE_LABELS = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
