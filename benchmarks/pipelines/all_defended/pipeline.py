import hashlib
import os
import re
import unicodedata
from typing import Iterable, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline


TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

ALLOWED_LABELS = {
    "sci.space",
    "rec.sport.baseball",
    "talk.politics.guns",
}

MAX_INPUT_LENGTH = 4000

FIXED_VOCABULARY = {
    "orbital": 0,
    "baseball": 1,
    "firearm": 2,
}

TRUSTED_DATASET_SHA256 = "29111f98194c6b98ba4a674a7419febea7483f5e7cf324756e6fe35eee2799dc"


def _sha256_file(path: str) -> str:
    """
    Return the SHA-256 digest of a file.
    """
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_dataset_integrity(
    dataset_path: str,
    expected_sha256: str = "",
) -> None:
    """
    V1 defense:
    Verify dataset integrity before training.

    If an expected digest is supplied, the file must match it.
    """
    if not os.path.isfile(dataset_path):
        raise ValueError(
            f"Dataset does not exist: {dataset_path}"
        )

    if expected_sha256:
        actual_sha256 = _sha256_file(dataset_path)

        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Dataset integrity verification failed."
            )


def validate_dataset(
    df: pd.DataFrame,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """
    V3 defense:
    Enforce schema, null, duplicate, type, label-domain,
    and minimum-cardinality checks before fitting.
    """
    required_columns = {
        text_column,
        label_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    validated = df[
        [
            text_column,
            label_column,
        ]
    ].copy()

    if validated.empty:
        raise ValueError(
            "Dataset must not be empty."
        )

    if validated[
        [
            text_column,
            label_column,
        ]
    ].isna().any().any():
        raise ValueError(
            "Null values are not allowed."
        )

    if validated.duplicated().any():
        raise ValueError(
            "Duplicate rows are not allowed."
        )

    if not validated[text_column].map(
        lambda value: isinstance(value, str)
    ).all():
        raise TypeError(
            "All text values must be strings."
        )

    if not validated[label_column].map(
        lambda value: isinstance(value, str)
    ).all():
        raise TypeError(
            "All labels must be strings."
        )

    unknown_labels = (
        set(validated[label_column].unique())
        - ALLOWED_LABELS
    )

    if unknown_labels:
        raise ValueError(
            "Unexpected labels detected: "
            + ", ".join(
                sorted(unknown_labels)
            )
        )

    class_counts = validated[
        label_column
    ].value_counts()

    if len(class_counts) != len(ALLOWED_LABELS):
        raise ValueError(
            "All expected classes must be present."
        )

    if (class_counts < 3).any():
        raise ValueError(
            "Each class must contain at least 3 samples."
        )

    if validated[text_column].str.strip().eq("").any():
        raise ValueError(
            "Empty text values are not allowed."
        )

    return validated


def clean_text(
    text: str,
) -> str:
    """
    V2 defense:
    Normalize text, reject invalid types, bound input size,
    strip control/bidi characters, and keep a conservative
    character allowlist.
    """
    if not isinstance(text, str):
        raise TypeError(
            "Input text must be a string."
        )

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            "Input text exceeds maximum allowed length."
        )

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = "".join(
        char
        for char in normalized
        if unicodedata.category(char)
        not in {
            "Cc",
            "Cf",
            "Cs",
        }
    )

    normalized = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = normalized.lower()

    normalized = re.sub(
        r"[^a-z\s]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    if not normalized:
        raise ValueError(
            "Input is empty after normalization."
        )

    return normalized


def clean_text_batch(
    values: Iterable[str],
) -> list[str]:
    """
    Apply the bounded preprocessing routine to a batch.
    """
    return [
        clean_text(value)
        for value in values
    ]


def load_dataset(
    dataset_path: str,
) -> pd.DataFrame:
    """
    Load a trusted CSV and validate it before use.
    """
    verify_dataset_integrity(
        dataset_path,
        TRUSTED_DATASET_SHA256,
    )

    df = pd.read_csv(
        dataset_path
    )

    return validate_dataset(
        df
    )


def build_model() -> Pipeline:
    """
    V4 defense:
    Use a fixed reviewed feature vocabulary and a prototype-based
    classifier. Clean samples map to orthogonal unit directions,
    producing a large transformed-feature decision margin.

    The final estimator is intentionally a direct sklearn estimator
    so AegisML can evaluate it with ART.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    vocabulary=FIXED_VOCABULARY,
                    lowercase=True,
                    norm="l2",
                ),
            ),
            (
                "classifier",
                NearestCentroid(
                    metric="euclidean"
                ),
            ),
        ]
    )


def train_model(
    dataset_path: str,
) -> Pipeline:
    """
    Train only after V1/V2/V3 controls have validated the data.
    """
    df = load_dataset(
        dataset_path
    )

    clean_inputs = clean_text_batch(
        df[TEXT_COLUMN].tolist()
    )

    labels = df[
        LABEL_COLUMN
    ].tolist()

    model = build_model()

    model.fit(
        clean_inputs,
        labels,
    )

    return model


def predict_text(
    model: Pipeline,
    text: str,
) -> str:
    """
    Hardened inference:
    normalize input and reject samples outside the reviewed
    fixed-vocabulary feature boundary.
    """
    cleaned = clean_text(
        text
    )

    vectorizer = model.named_steps[
        "tfidf"
    ]

    vector = vectorizer.transform(
        [
            cleaned
        ]
    )

    if vector.nnz == 0:
        raise ValueError(
            "Input is outside the supported feature vocabulary."
        )

    prediction = model.predict(
        [
            cleaned
        ]
    )

    return str(
        prediction[0]
    )


def save_model(
    model: Pipeline,
    model_path: str = "model.pkl",
) -> None:
    """
    Persist the reviewed trained model.
    """
    joblib.dump(
        model,
        model_path,
    )


def main() -> Tuple[Pipeline, pd.DataFrame]:
    """
    Example end-to-end hardened training flow.
    """
    dataset_path = "dataset.csv"

    model = train_model(
        dataset_path
    )

    save_model(
        model,
        "model.pkl",
    )

    dataset = load_dataset(
        dataset_path
    )

    return model, dataset


if __name__ == "__main__":
    main()
