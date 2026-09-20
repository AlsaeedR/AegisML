"""Standalone trainable pipeline target for AegisML testing.

Expected dataset columns: text, label. Run this file to create model.pkl.
The preprocessing function is intentionally vulnerable at the V2 boundary.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def preprocess_text(text: str) -> str:
    """Normalize text but fail on malformed edge-case inputs."""
    if not isinstance(text, str):
        return ""

    if not text.strip():
        raise ValueError("Empty input is not supported")

    if len(text) > 5000:
        raise ValueError("Input exceeds the preprocessing buffer limit")

    if any(
        ord(character) < 32 and character not in "\n\t\r"
        for character in text
    ):
        raise ValueError("Control characters are not supported")

    if any(ord(character) > 127 for character in text):
        raise ValueError("Non-ASCII characters are not supported")

    if any(not character.isalnum() and not character.isspace() for character in text):
        raise ValueError("Punctuation is not supported")

    return _normalize_text(text)


def _normalize_text(text: str) -> str:
    """Normalize training records without applying the attack-surface guard."""
    return " ".join(text.lower().split())


def build_model() -> Pipeline:
    """Build the serialized model used by the sandbox tests."""
    return Pipeline([
        ("features", FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "character_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=10000,
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
        ])),
        ("classifier", LogisticRegression(C=0.25, max_iter=500)),
    ])


def train_and_save(
    dataset_path: Path,
    output_path: Path,
    text_column: str = "text",
    label_column: str = "label",
) -> None:
    """Train the model from CSV data and write a joblib pickle artifact."""
    dataset = pd.read_csv(dataset_path)
    missing_columns = {text_column, label_column} - set(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}. "
            f"Available columns: {list(dataset.columns)}"
        )

    dataset = dataset.dropna(subset=[text_column, label_column])
    if len(dataset) < 10 or dataset[label_column].nunique() < 2:
        raise ValueError("The dataset needs at least 10 rows and two label values.")

    texts = dataset[text_column].astype(str).map(_normalize_text)
    labels = dataset[label_column].astype(str)
    model = build_model()
    model.fit(texts, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")
    print(f"Trained on {len(dataset)} rows across {labels.nunique()} labels")


def main() -> None:
    target_root = Path(__file__).resolve().parent
    default_dataset = target_root / "dataset1_clean.csv"
    parser = argparse.ArgumentParser(description="Train the AegisML preprocessing target.")
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--output", type=Path, default=target_root / "model.pkl")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()
    train_and_save(
        dataset_path=args.dataset,
        output_path=args.output,
        text_column=args.text_column,
        label_column=args.label_column,
    )


if __name__ == "__main__":
    main()
