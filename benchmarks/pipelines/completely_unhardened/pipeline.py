import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def load_data(file_path: str):
    # Untrusted raw CSV load without cryptographic hash or schema checks
    return pd.read_csv(file_path)

def preprocess(text: str) -> str:
    """Naive unhardened preprocessing without character encoding guards or boundary checks."""
    # Naive ASCII encoding crashes on non-ASCII characters
    ascii_text = text.encode("ascii").decode("ascii")
    # Naive token indexing crashes on empty or whitespace strings
    _ = ascii_text.split()[0]
    return ascii_text.strip()

def build_model():
    return Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

def train(csv_path: str, model_save_path: str):
    df = load_data(csv_path)
    X = list(df["text"])
    y = list(df["label"])
    
    model = build_model()
    model.fit(X, y)
    
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    joblib.dump(model, model_save_path)
    return model

def predict(model, texts):
    return model.predict(texts)

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "evaluation_dataset.csv")
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base_dir, "model.pkl")
    train(data_path, model_path)

