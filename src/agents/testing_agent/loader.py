import os
import joblib
import pickle
import pandas as pd
from typing import Any, Tuple, List, Optional, Dict


def load_trained_model(model_path: str) -> Any:
    """
    Loads a serialized model artifact using joblib with a pickle fallback.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def load_vectorizer(vectorizer_path: Optional[str]) -> Optional[Any]:
    """
    Loads a standalone vectorizer artifact if one was saved separately from the model.
    """
    if not vectorizer_path or not os.path.exists(vectorizer_path):
        return None

    try:
        return joblib.load(vectorizer_path)
    except Exception:
        try:
            with open(vectorizer_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


def resolve_vectorizer_from_agent1(agent_1_results: Optional[Dict[str, Any]], base_dir: str = "data") -> Optional[str]:
    """
    Inspects Agent 1 pipeline graph nodes to locate any vectorizer artifacts
    serialized during pipeline execution (such as tfidf_vectorizer.pkl).
    """
    if not agent_1_results:
        return None

    pipeline_graph = agent_1_results.get("pipeline_graph", {})
    nodes = pipeline_graph.get("nodes", [])

    candidate_names = ["tfidf_vectorizer.pkl", "vectorizer.pkl", "tfidf.pkl"]

    # First check if AST discovered specific serialization filenames
    for node in nodes:
        desc = node.get("description", "").lower()
        for candidate in candidate_names:
            if candidate in desc:
                full_path = os.path.join(base_dir, candidate)
                if os.path.exists(full_path):
                    return full_path

    # Fallback: check if candidate vectorizer exists directly in the data directory
    for candidate in candidate_names:
        candidate_path = os.path.join(base_dir, candidate)
        if os.path.exists(candidate_path):
            return candidate_path

    return None


def load_dataset(
    dataset_path: str,
    text_column: str,
    label_column: str,
    sample_size: int = 200,
) -> Tuple[List[str], List[Any]]:
    """
    Loads and samples evaluation data from CSV, confirming required columns exist.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Dataset must contain '{text_column}' and '{label_column}' columns. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_column, label_column])

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()

    return texts, labels
