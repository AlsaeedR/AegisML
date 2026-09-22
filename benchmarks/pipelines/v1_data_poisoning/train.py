"""
Fake News Detection
--------------------
Trains a TF-IDF + PassiveAggressiveClassifier model to classify news
articles as REAL or FAKE, following the classic approach popularized by
the DataFlair "Detecting Fake News" tutorial.

Usage:
    python train.py --data ../data/news.csv

The CSV is expected to have at least these columns:
    title, text, label   (label values: REAL / FAKE)

If your dataset uses different column names, pass --text-col and
--label-col to override the defaults.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    return df


def build_pipeline(max_df: float = 0.7):
    vectorizer = TfidfVectorizer(stop_words="english", max_df=max_df)
    classifier = PassiveAggressiveClassifier(max_iter=50)
    return vectorizer, classifier


def train_and_evaluate(df: pd.DataFrame, text_col: str, label_col: str,
                        test_size: float = 0.2, random_state: int = 7):
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=test_size, random_state=random_state
    )

    vectorizer, classifier = build_pipeline()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier.fit(X_train_tfidf, y_train)
    y_pred = classifier.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {round(acc * 100, 2)}%")
    print("Confusion Matrix (labels = {}):".format(list(classifier.classes_)))
    print(cm)
    print("\nClassification Report:")
    print(report)

    return vectorizer, classifier, acc, cm


def save_model(vectorizer, classifier, out_dir: str = "."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(out_dir / "classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)
    print(f"Saved vectorizer.pkl and classifier.pkl to {out_dir.resolve()}")


def predict_single(vectorizer, classifier, text: str) -> str:
    vec = vectorizer.transform([text])
    return classifier.predict(vec)[0]


def main():
    parser = argparse.ArgumentParser(description="Train a fake news detection model")
    parser.add_argument("--data", default="../data/news.csv", help="Path to the news CSV file")
    parser.add_argument("--text-col", default="text", help="Name of the text column")
    parser.add_argument("--label-col", default="label", help="Name of the label column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--out-dir", default=".", help="Directory to save the trained model files")
    args = parser.parse_args()

    df = load_data(args.data, args.text_col, args.label_col)
    print(f"Loaded {len(df)} rows from {args.data}")

    vectorizer, classifier, acc, cm = train_and_evaluate(
        df, args.text_col, args.label_col, test_size=args.test_size
    )
    save_model(vectorizer, classifier, args.out_dir)


if __name__ == "__main__":
    main()
