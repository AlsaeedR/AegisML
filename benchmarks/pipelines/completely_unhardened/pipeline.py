import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def load_data(file_path: str):
    # Untrusted raw CSV load without cryptographic hash or schema checks
    return pd.read_csv(file_path)

def preprocess(text: str) -> str:
    # Basic lowercasing without character whitelisting or boundary guards
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

def train(csv_path: str, model_save_path: str, vectorizer_save_path: str):
    df = load_data(csv_path)
    # Direct column access without validation
    df["clean_text"] = df["text"].apply(preprocess)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    joblib.dump(vectorizer, vectorizer_save_path)
    return model, vectorizer

def predict(model, vectorizer, texts):
    cleaned = [preprocess(t) for t in texts]
    X = vectorizer.transform(cleaned)
    return model.predict(X)

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    train(data_path, "outputs/model.pkl", "outputs/tfidf_vectorizer.pkl")

