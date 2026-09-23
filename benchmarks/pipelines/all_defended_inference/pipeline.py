import os
import re
from typing import List
import joblib
from pydantic import BaseModel, Field

class InferencePayload(BaseModel):
    items: List[str] = Field(..., min_items=1)

def load_inference_artifacts(model_path: str, vectorizer_path: str):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def preprocess_batch(text: str) -> str:
    # No SQL/command injection filtering, only simple strip
    if not isinstance(text, str):
        return ""
    return text.strip()

def run_batch_inference(model, vectorizer, raw_payload: dict) -> list:
    # Protected by Pydantic schema
    validated = InferencePayload(**raw_payload)
    
    cleaned = [preprocess_batch(item) for item in validated.items]
    features = vectorizer.transform(cleaned)
    predictions = model.predict(features)
    return predictions.tolist()

