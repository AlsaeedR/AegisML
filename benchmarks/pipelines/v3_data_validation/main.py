"""
AegisML V3-only controlled fixture
Adapted from the Delecis text-classification workflow.

TARGET PROFILE
==============
V1 Data Poisoning
    DEFENDED / LOW

V2 Preprocessing Attack Surface
    DEFENDED / LOW

V3 Data Validation Weaknesses
    INTENTIONALLY VULNERABLE
    This is the ONLY intended vulnerability.

V4 Adversarial Robustness
    DEFENDED / LOW

V4 DESIGN NOTE
==============
The final estimator is a DIRECT sklearn NearestCentroid classifier rather than
a nested pipeline. This keeps AegisML dynamic scheduling compatible.

The model uses:
- a small security-reviewed fixed feature vocabulary,
- L2-normalized TF-IDF features,
- prototype / nearest-centroid classification,
- explicit large inter-class feature-space separation,
- adversarial-style text augmentation,
- inference rejection for out-of-vocabulary and low-margin inputs.

The class prototypes are deliberately separated so the nearest decision
boundary lies beyond AegisML's 0.5 relative perturbation budget on the clean
evaluation samples.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline


DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"
TRUSTED_DATASET_SHA256 = "b91ba5e712f583d0eda24be7540ce2c19fa2f334053f4674609a9c7b68f7aded"

MAX_INPUT_LENGTH = 1000
MIN_RELATIVE_CENTROID_MARGIN = 0.15

# Security-reviewed, intentionally narrow feature surface.
ROBUST_FEATURE_VOCABULARY = {
    "alpha": 0,
    "beta": 1,
}


# ---------------------------------------------------------------------------
# V1 - DATA POISONING DEFENSE
# ---------------------------------------------------------------------------

def verify_dataset_integrity(path: str | Path) -> None:
    """
    Refuse to train unless the approved dataset artifact is present and its
    SHA-256 digest exactly matches the trusted value.
    """
    p = Path(path)

    if not p.exists() or not p.is_file():
        raise ValueError("Trusted training dataset is missing.")

    actual = hashlib.sha256(p.read_bytes()).hexdigest()

    if actual != TRUSTED_DATASET_SHA256:
        raise ValueError(
            "Training dataset integrity verification failed. "
            "Refusing to train on modified or untrusted data."
        )


# ---------------------------------------------------------------------------
# V2 - PREPROCESSING DEFENSE
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """
    Harden untrusted text before inference.

    Controls:
    - strict string handling
    - maximum length cap
    - Unicode NFKC normalization
    - control / zero-width / bidi character removal
    - URL removal
    - lowercase normalization
    - character allow-listing
    - whitespace normalization
    """
    if not isinstance(value, str):
        return ""

    text = value[:MAX_INPUT_LENGTH]
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(
        r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\ufeff]",
        " ",
        text,
    )
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# V3 - INTENTIONALLY WEAK DATA VALIDATION
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """
    V3 INTENTIONALLY VULNERABLE.

    File integrity is verified for V1, but row-level training validation is
    deliberately absent.

    There is intentionally:
    - no null rejection
    - no duplicate rejection
    - no allowed-label/domain validation
    - no dataframe dtype validation
    - no class balance/cardinality validation
    """
    verify_dataset_integrity(path)

    # INTENTIONAL V3 WEAKNESS:
    return pd.read_csv(path, encoding="utf-8")


# ---------------------------------------------------------------------------
# V4 - ADVERSARIAL ROBUSTNESS DEFENSE
# ---------------------------------------------------------------------------

def adversarial_training_variants(text: Any) -> list[str]:
    """
    Create bounded text perturbations while preserving the security-reviewed
    class marker. These are used as adversarial-style training augmentation.
    """
    base = clean_text(text)

    if not base:
        return [base]

    tokens = base.split()
    variants = [base]

    if len(tokens) > 3:
        variants.append(" ".join(tokens[1:] + tokens[:1]))

    if len(tokens) > 5:
        variants.append(" ".join(tokens[:-1]))

    variants.append(re.sub(r"\s+", " ", base))

    return list(dict.fromkeys(variants))


def build_model() -> Pipeline:
    """
    Direct-estimator architecture compatible with AegisML dynamic testing.

    Robustness properties:
    - fixed allow-listed feature vocabulary
    - L2-normalized feature vectors
    - prototype-based nearest-centroid decision rule
    - no fragile single linear coefficient vector
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    vocabulary=ROBUST_FEATURE_VOCABULARY,
                    lowercase=True,
                    norm="l2",
                ),
            ),
            (
                "classifier",
                NearestCentroid(metric="euclidean"),
            ),
        ]
    )


def train_model(
    dataset_path: str | Path = DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
) -> Pipeline:
    """
    Train while intentionally preserving the V3 weakness.

    No row-level schema/null/duplicate/domain validation is added here.
    """
    df = load_dataset(dataset_path)

    # INTENTIONAL V3 WEAKNESS:
    raw_texts = df["text"].tolist()
    labels = df["label"].tolist()

    augmented_texts = []
    augmented_labels = []

    for text, label in zip(raw_texts, labels):
        for variant in adversarial_training_variants(text):
            augmented_texts.append(variant)
            augmented_labels.append(label)

    model = build_model()
    model.fit(augmented_texts, augmented_labels)
    joblib.dump(model, model_path)

    return model


def load_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)


def _relative_centroid_margin(model: Pipeline, cleaned_text: str) -> float:
    """
    Measure relative distance to the nearest prototype decision boundary.
    Used as an inference-time adversarial/OOD rejection gate.
    """
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    x = vectorizer.transform([cleaned_text]).toarray()[0]
    norm = float(np.linalg.norm(x))

    if norm <= 1e-12:
        return 0.0

    predicted = classifier.predict([x])[0]
    classes = list(classifier.classes_)

    if len(classes) != 2:
        return 0.0

    i = classes.index(predicted)
    j = 1 - i

    ci = classifier.centroids_[i]
    cj = classifier.centroids_[j]

    direction = cj - ci
    denominator = float(np.linalg.norm(direction))

    if denominator <= 1e-12:
        return 0.0

    numerator = abs(
        float(np.dot(x, direction))
        - 0.5 * (
            float(np.dot(cj, cj))
            - float(np.dot(ci, ci))
        )
    )

    return (numerator / denominator) / norm


def predict_text(model: Pipeline, text: Any):
    """
    Hardened inference boundary for V2 and V4.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError("Input is empty after normalization.")

    vector = model.named_steps["tfidf"].transform([cleaned])

    if vector.nnz == 0:
        raise ValueError(
            "Out-of-distribution input rejected: no approved robust features."
        )

    margin = _relative_centroid_margin(model, cleaned)

    if margin < MIN_RELATIVE_CENTROID_MARGIN:
        raise ValueError(
            "Low-margin input rejected by adversarial robustness gate."
        )

    return model.predict([cleaned])[0]


if __name__ == "__main__":
    trained = train_model()
    print("Saved:", MODEL_PATH)
    print("Example:", predict_text(trained, "alpha alpha robust secure marker"))
