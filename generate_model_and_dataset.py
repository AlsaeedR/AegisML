import joblib
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CATEGORIES = ["rec.sport.baseball", "sci.space", "talk.politics.guns"]

print("Downloading a small text classification dataset (20 Newsgroups subset)...")
data = fetch_20newsgroups(
    subset="train",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
)

texts = data.data
labels = [data.target_names[i] for i in data.target]

print(f"Loaded {len(texts)} samples across {len(CATEGORIES)} categories.")

print("Training TF-IDF + Logistic Regression pipeline...")
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000)),
    ("clf", LogisticRegression(max_iter=200)),
])
model.fit(texts, labels)

joblib.dump(model, "data/model.pkl")
print("Saved trained model to data/model.pkl")

df = pd.DataFrame({"text": texts, "label": labels})
df.to_csv("data/dataset.csv", index=False)
print(f"Saved dataset to data/dataset.csv ({len(df)} rows)")

print("\nDone! You can now run Agent 2 with:")
print('  model_path="data/model.pkl"')
print('  dataset_path="data/dataset.csv"')
print('  text_column="text"')
print('  label_column="label"')
