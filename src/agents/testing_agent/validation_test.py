import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Any, Dict


def run_validation_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = state.get("dataset_path")
    text_col = state.get("text_column")
    label_col = state.get("label_column")
    model = state.get("model")
    
    if not dataset_path or not model:
        return {"validation_evidence": {
            "vulnerability_id": "V3", "vulnerability_name": "Data Validation Weaknesses",
            "status": "inconclusive", "severity": "low",
            "evidence": {"status": "skipped", "reason": "Missing model or dataset"}
        }}

    # 1. Load data and split
    
    df = pd.read_csv(dataset_path)
    
    df[text_col] = df[text_col].fillna("")
    
    X_text = df[text_col].tolist()
    y_true = df[label_col].tolist()

    # Standard 70/30 split like V1
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_true, test_size=0.3, random_state=42
    )

    # 2. Get Clean Baseline Accuracy
    clean_model = clone(model)
    clean_model.fit(X_train, y_train)
    clean_preds = clean_model.predict(X_test)
    clean_acc = accuracy_score(y_test, clean_preds)

    # 3. Corrupt the training data (add duplicates and missing values)
    # We create a new corrupted training set
    X_corrupt = list(X_train)
    y_corrupt = list(y_train)

    # Inject 20% duplicates
    n_dup = int(len(X_train) * 0.2)
    for _ in range(n_dup):
        idx = np.random.randint(0, len(X_train))
        X_corrupt.append(X_train[idx])
        y_corrupt.append(y_train[idx])

    # Inject missing values (NaN) into first 20 strings
    # Note: Since it's text, we can use None or empty strings, or try to force an error
    # For the model to see a "missing value", we'll insert empty strings
        # Inject missing values (empty strings) into first 20 strings
    for i in range(min(20, len(X_corrupt))):
        X_corrupt[i] = ""   # <--- Use an empty string, NOT np.nan!

    # 4. Retrain on corrupted data
    corrupted_model = clone(model)
    corrupted_model.fit(X_corrupt, y_corrupt)

    # 5. Test on clean test set
    corrupted_preds = corrupted_model.predict(X_test)
    corrupted_acc = accuracy_score(y_test, corrupted_preds)

    accuracy_drop = round(clean_acc - corrupted_acc, 4)

    # 6. Decide status based on drop (same threshold as V1)
    status = "vulnerable" if accuracy_drop > 0.05 else "not_vulnerable"
    severity = "high" if accuracy_drop > 0.15 else "medium" if accuracy_drop > 0.05 else "low"

    return {"validation_evidence": {
        "vulnerability_id": "V3",
        "vulnerability_name": "Data Validation Weaknesses",
        "status": status,
        "severity": severity,
        "evidence": {
            "method": "data_corruption_retrain",
            "clean_accuracy": round(clean_acc, 4),
            "corrupted_accuracy": round(corrupted_acc, 4),
            "accuracy_drop": accuracy_drop,
            "n_duplicates_injected": n_dup,
            "n_missing_values_injected": min(20, len(X_train)),
            "interpretation": (
                "Significant accuracy degradation observed after injecting duplicates and missing "
                "values into the training set. Confirms absence of robust schema validation and "
                "deduplication before training."
                if accuracy_drop > 0.05
                else "Model demonstrated empirical resilience to data quality issues (duplicates/missing)."
            )
        },
        "summary": "vulnerable" if accuracy_drop > 0.05 else "all_clean"
    }}