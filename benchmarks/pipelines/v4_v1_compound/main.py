from __future__ import annotations

import argparse
import os
import pickle
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class DataValidationError(ValueError):
    """Raised when dataset schema, format, or values fail validation checks."""
    pass


# Domain & Boundary Constraints
ALLOWED_LABELS = {"positive", "negative"}
REQUIRED_COLUMNS = ["text", "label"]

MIN_TEXT_LENGTH = 3
MAX_INPUT_LENGTH = 1000
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MIN_TOTAL_SAMPLES = 4
MIN_SAMPLES_PER_CLASS = 2
MAX_IMBALANCE_RATIO = 10.0


# ---------------------------------------------------------------------------
# 1. Data Cleaning & Preprocessing (Protected against Preprocessing Attacks)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize text safely against preprocessing attacks (homoglyphs, control chars, ReDoS)."""
    if not isinstance(text, str):
        return ""

    # Bound length against ReDoS and memory exhaustion
    text = text[:MAX_INPUT_LENGTH]

    # Unicode NFKC normalization to neutralize homoglyphs and compatibility spoofing
    text = unicodedata.normalize("NFKC", text)

    # Strip invisible, control, and zero-width characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\ufeff]", "", text)

    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 2. Strict Data Validation & Poisoning Mitigation
# ---------------------------------------------------------------------------

def is_suspicious_poison(text_clean: str) -> bool:
    """Detect repetitive token stuffing or anomalous backdoor triggers (defends Data Poisoning)."""
    tokens = text_clean.split()
    if not tokens:
        return False
    if len(tokens) >= 5:
        most_common_count = Counter(tokens).most_common(1)[0][1]
        if most_common_count / len(tokens) > 0.4:
            return True
    return False


def resolve_path(path: str | Path) -> Path:
    """Resolve file path checking cwd and script directory."""
    p = Path(path)
    if p.exists():
        return p
    base_dir = Path(__file__).resolve().parent
    if (base_dir / path).exists():
        return base_dir / path
    return p


def validate_file(path: str | Path) -> Path:
    """Validate file existence, extension, and bounds."""
    p = resolve_path(path)
    if not p.exists():
        raise DataValidationError(f"File not found: {p.resolve()}")
    if not p.is_file():
        raise DataValidationError(f"Expected a regular file path, got: {p.resolve()}")
    if p.suffix.lower() != ".csv":
        raise DataValidationError(f"Invalid file extension '{p.suffix}'. Expected a '.csv' file.")
    file_size = p.stat().st_size
    if file_size == 0:
        raise DataValidationError(f"File is empty (0 bytes): {p.resolve()}")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise DataValidationError(f"File size ({file_size} bytes) exceeds limit ({MAX_FILE_SIZE_BYTES} bytes).")
    return p


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load, sanitize, and validate dataset with robust schema validation and deduplication."""
    p = validate_file(path)

    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise DataValidationError(f"Failed to parse CSV file: {e}") from e

    # 1. Robust Schema Validation
    df.columns = [str(c).strip().lower() for c in df.columns]
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise DataValidationError(f"Schema violation: missing required columns {sorted(missing)}.")

    # Restrict strictly to required schema columns (drops injected/extra columns)
    df = df[["text", "label"]].copy()

    # 2. Missing Values & Corrupted Data Sanitization
    initial_rows = len(df)
    df = df.dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Filter out empty or whitespace-only texts
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH].copy()
    df["text"] = df["text"].str.slice(0, MAX_INPUT_LENGTH)

    # Filter strictly to allowed classes
    df = df[df["label"].isin(ALLOWED_LABELS)].copy()

    dropped_missing = initial_rows - len(df)
    if dropped_missing > 0:
        print(f"Schema Validation: Filtered out {dropped_missing} invalid/missing/corrupted rows.")

    # 3. Clean text normalization
    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() > 0].copy()

    # 4. Defenses against Data Poisoning & Injected Duplicates BEFORE Training:
    # 4a. Detect and remove label-flipped / conflicting duplicates
    label_counts_per_text = df.groupby("text_clean")["label"].nunique()
    unconflicted_texts = label_counts_per_text[label_counts_per_text == 1].index
    conflicts_count = len(label_counts_per_text) - len(unconflicted_texts)
    if conflicts_count > 0:
        print(f"Poisoning Defense: Purged {conflicts_count} label-conflicted/flipped text groups.")
    df = df[df["text_clean"].isin(unconflicted_texts)].copy()

    # 4b. Robust Deduplication: Collapse all repeated samples to single instances
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text_clean"], keep="first").copy()
    duplicates_removed = before_dedup - len(df)
    if duplicates_removed > 0:
        print(f"Deduplication Defense: Successfully purged {duplicates_removed} duplicate training samples.")

    # 4c. Filter out anomalous repeated token stuffing triggers
    df = df[~df["text_clean"].map(is_suspicious_poison)].copy()

    if df.empty:
        raise DataValidationError("No valid data remaining after schema validation and deduplication.")

    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[int, int]:
    """Validate dataset row counts, class balance, and distribution bounds."""
    if df.empty:
        raise DataValidationError("Dataset is empty after validation and filtering.")

    total_samples = len(df)
    if total_samples < MIN_TOTAL_SAMPLES:
        raise DataValidationError(
            f"Insufficient dataset size: has {total_samples} samples; minimum required is {MIN_TOTAL_SAMPLES}."
        )

    counts = df["label"].value_counts()
    class_count = len(counts)
    if class_count < 2:
        raise DataValidationError(
            f"Dataset has only {class_count} class ({counts.index.tolist()}). At least 2 classes are required."
        )

    underrepresented = counts[counts < MIN_SAMPLES_PER_CLASS]
    if not underrepresented.empty:
        raise DataValidationError(
            f"Classes underrepresented (fewer than {MIN_SAMPLES_PER_CLASS} samples): {underrepresented.to_dict()}."
        )

    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio > MAX_IMBALANCE_RATIO:
        raise DataValidationError(
            f"Severe class imbalance detected: max/min ratio is {imbalance_ratio:.1f}x (limit is {MAX_IMBALANCE_RATIO:.1f}x)."
        )

    return total_samples, class_count


# ---------------------------------------------------------------------------
# 3. Model Training (INTENTIONALLY VULNERABLE to Adversarial Robustness)
# ---------------------------------------------------------------------------

def train_model(
    data_path: str | Path = "sample_reviews_200.csv",
    output_path: str | Path = "model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train pipeline.

    VULNERABILITY NOTE:
    This pipeline deliberately uses standard TF-IDF and Logistic Regression without
    adversarial training, synonym substitution defense, or robust smoothing.
    It remains VULNERABLE against Adversarial Robustness (e.g. test-time evasion
    attacks and adversarial word perturbations).
    """
    data_file = resolve_path(data_path)

    print("=" * 60)
    print(" TRAINING SENTIMENT MODEL (ad2)")
    print("=" * 60)

    df = load_dataset(data_file)
    row_count, class_count = validate_dataset(df)
    print(f"Loaded {row_count} validated samples across {class_count} classes from '{data_file.name}'.")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())

    # Ensure pre-training deduplication is guaranteed
    df = df.drop_duplicates(subset=["text_clean"], keep="first").copy()

    x_train, x_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    # Standard model pipeline (intentionally susceptible to adversarial perturbations)
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
            ),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
    rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print("\nValidation Metrics (Holdout Test Split):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    labels = sorted(np.unique(df["label"]))
    cm = confusion_matrix(y_test, predictions, labels=labels)

    # Save model as .pkl in ad2
    base_dir = Path(__file__).resolve().parent
    out_file = Path(output_path)
    if not out_file.is_absolute() and not str(out_file).replace("\\", "/").startswith("ad2/"):
        out_file = base_dir / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "wb") as f:
        pickle.dump(model, f)
    print(f"\nTrained model exported successfully as pkl to:\n  -> {out_file.resolve()}")

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "labels": labels,
    }
    return model, metrics


# ---------------------------------------------------------------------------
# 4. Model Inference & Input Validation
# ---------------------------------------------------------------------------

def load_model(model_path: str | Path = "model.pkl") -> Pipeline:
    """Load trained pipeline from .pkl (or .joblib fallback)."""
    p = resolve_path(model_path)
    if not p.exists():
        base_dir = Path(__file__).resolve().parent
        if (base_dir / "model.pkl").exists():
            p = base_dir / "model.pkl"

    if not p.exists():
        raise FileNotFoundError(f"Model not found at '{p.resolve()}'. Train the model first.")

    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(p)


def validate_input_text(text: Any) -> str:
    """Validate single text input for prediction / inference (defends Data Validation Weakness)."""
    if text is None:
        raise DataValidationError("Input text is None. A valid non-empty string is required.")
    if not isinstance(text, str):
        raise DataValidationError(f"Invalid input type: expected str, got {type(text).__name__}.")
    if not text.strip():
        raise DataValidationError("Input text cannot be empty or contain only whitespace.")
    if len(text) > MAX_INPUT_LENGTH:
        raise DataValidationError(f"Input text length ({len(text)}) exceeds maximum allowed limit ({MAX_INPUT_LENGTH} characters).")
    cleaned = clean_text(text)
    if not cleaned:
        raise DataValidationError("Input text contains no recognizable words after cleaning.")
    return cleaned


def predict_sentiment(
    text: str,
    model: Pipeline | str | Path = "model.pkl",
) -> str:
    """Predict sentiment ('positive' or 'negative') for input text with strict validation."""
    if isinstance(model, (str, Path)):
        model = load_model(model)

    cleaned = validate_input_text(text)
    prediction = model.predict([cleaned])[0]
    return str(prediction)


def predict_text(
    model: Pipeline | str | Path,
    text: str,
) -> Tuple[str, float]:
    """Predict label and confidence for a single input string."""
    if isinstance(model, (str, Path)):
        model = load_model(model)

    cleaned = validate_input_text(text)
    prediction = model.predict([cleaned])[0]

    confidence = 1.0
    classifier = model.named_steps.get("classifier")
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(max(probabilities))

    return str(prediction), confidence


# ---------------------------------------------------------------------------
# 5. Evaluation on External Dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model: Pipeline | str | Path,
    eval_csv_path: str | Path,
) -> Dict[str, Any]:
    """Evaluate a trained model against an evaluation CSV dataset."""
    p = resolve_path(eval_csv_path)

    print("=" * 60)
    print(f" EVALUATING ON: {p.name}")
    print("=" * 60)

    if isinstance(model, (str, Path)):
        model = load_model(model)

    df = load_dataset(p)
    row_count, class_count = validate_dataset(df)
    print(f"Loaded {row_count} evaluation rows across {class_count} classes.")

    predictions = model.predict(df["text_clean"])
    acc = accuracy_score(df["label"], predictions)
    prec = precision_score(df["label"], predictions, average="weighted", zero_division=0)
    rec = recall_score(df["label"], predictions, average="weighted", zero_division=0)
    f1 = f1_score(df["label"], predictions, average="weighted", zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(df["label"], predictions, zero_division=0))

    labels = sorted(np.unique(df["label"]))
    cm = confusion_matrix(df["label"], predictions, labels=labels)
    print("Confusion Matrix:")
    print(f"Labels: {labels}")
    print(cm)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "predictions": predictions,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# 6. CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified NLP Pipeline for ad2 (Hardened against Poisoning, Preprocessing, and Validation Weaknesses; Vulnerable to Adversarial Perturbations)."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    train_parser = subparsers.add_parser("train", help="Train the classification model.")
    train_parser.add_argument(
        "--data",
        default="sample_reviews_200.csv",
        help="Path to training CSV file.",
    )
    train_parser.add_argument(
        "--output",
        default="model.pkl",
        help="Path to save trained model file (.pkl).",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict sentiment for input text.")
    predict_parser.add_argument(
        "--model",
        default="model.pkl",
        help="Path to saved model file (.pkl).",
    )
    predict_parser.add_argument("--text", required=True, help="Input text to classify.")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on a CSV dataset.")
    eval_parser.add_argument(
        "--model",
        default="model.pkl",
        help="Path to saved model file (.pkl).",
    )
    eval_parser.add_argument(
        "--data",
        default="evaluation_dataset.csv",
        help="Path to evaluation CSV dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        train_model(data_path=args.data, output_path=args.output, test_size=args.test_size)
    elif args.command == "predict":
        label, conf = predict_text(model=args.model, text=args.text)
        print(f"Input text     : {args.text}")
        print(f"Predicted label: {label}")
        print(f"Confidence     : {conf:.4f}")
    elif args.command == "evaluate":
        evaluate_dataset(model=args.model, eval_csv_path=args.data)
    else:
        # Default behavior if executed directly: run training
        train_model(data_path="sample_reviews_200.csv", output_path="model.pkl")


if __name__ == "__main__":
    main()
