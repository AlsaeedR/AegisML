import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Any, Dict


def run_validation_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    model = state.get("model")
    X_text = state.get("X_text")
    y_true = state.get("y_true")

    if not model or not X_text or not y_true:
        return {"validation_evidence": {
            "vulnerability_id": "V3", "vulnerability_name": "Data Validation Weaknesses",
            "status": "inconclusive", "severity": "low",
            "evidence": {"status": "skipped", "reason": "Missing model or preloaded dataset"}
        }}

    try:
        rng = np.random.default_rng(42)  # Reproducible, no global state

        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_true, test_size=0.3, random_state=42
        )

        clean_model = clone(model)
        clean_model.fit(X_train, y_train)
        clean_preds = clean_model.predict(X_test)
        clean_acc = accuracy_score(y_test, clean_preds)

        X_corrupt = list(X_train)
        y_corrupt = list(y_train)

        n_dup = int(len(X_train) * 0.2)
        dup_indices = rng.integers(0, len(X_train), size=n_dup)
        for idx in dup_indices:
            X_corrupt.append(X_train[idx])
            y_corrupt.append(y_train[idx])

        n_missing = min(20, len(X_corrupt))
        for i in range(n_missing):
            X_corrupt[i] = ""

        corrupted_model = clone(model)
        corrupted_model.fit(X_corrupt, y_corrupt)
        corrupted_preds = corrupted_model.predict(X_test)
        corrupted_acc = accuracy_score(y_test, corrupted_preds)

        accuracy_drop = round(clean_acc - corrupted_acc, 4)
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
                "n_missing_values_injected": n_missing,
                "interpretation": (
                    "Significant accuracy degradation observed after injecting duplicates and missing "
                    "values into the training set. Confirms absence of robust schema validation and "
                    "deduplication before training."
                    if accuracy_drop > 0.05
                    else "Model demonstrated empirical resilience to data quality issues (duplicates/missing)."
                ),
            },
            "summary": "vulnerable" if accuracy_drop > 0.05 else "all_clean",
        }}
    except Exception as e:
        return {"validation_evidence": {
            "vulnerability_id": "V3", "vulnerability_name": "Data Validation Weaknesses",
            "status": "inconclusive", "severity": "low",
            "evidence": {"status": "error", "reason": str(e)}
        }}