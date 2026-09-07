import joblib
import pickle
import pandas as pd
from typing import Any, Tuple, List

def load_trained_model(model_path: str) -> Any:
    try:
        return joblib.load(model_path)
    except Exception:
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def load_dataset(dataset_path: str, text_column: str, label_column: str, sample_size: int=200) -> Tuple[List[str], List[Any]]:
    df = pd.read_csv(dataset_path)
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Dataset must contain '{text_column}' and '{label_column}' columns. Found columns: {list(df.columns)}")
    df = df.dropna(subset=[text_column, label_column])
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()
    return (texts, labels)
