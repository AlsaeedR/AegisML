import numpy as np
from typing import Any, Dict

def run_preprocess_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    model = state.get("model") # The full pipeline handles vectorization!
    
    if not model:
        return {"preprocessing_evidence": {
            "vulnerability_id": "V2", "vulnerability_name": "Preprocessing Attack Surface",
            "status": "inconclusive", "severity": "low",
            "evidence": {"status": "skipped", "reason": "Missing model"}
        }}

    # Craft malicious inputs to break the preprocessing pipeline
    malicious_inputs = [
        "a" * 10000,           # Extremely long string
        "\x00\x01\x02",        # Null bytes
        "😀" * 500,            # Complex unicode
        "",                    # Empty string
        " " * 1000,            # Whitespace only
        "DROP TABLE users;",   # SQL injection attempt
        "<script>alert(1)</script>", # XSS attempt
    ]

    try:
        # This will run the internal vectorizer and classifier
        preds = model.predict(malicious_inputs)
        status = "not_vulnerable"
        severity = "low"
        evidence = {"status": "ok", "details": "Model handled malformed inputs without crashing", "predictions": preds.tolist()}
    except Exception as e:
        status = "vulnerable"
        severity = "high"
        evidence = {"status": "error", "reason": str(e)}

    return {"preprocessing_evidence": {
        "vulnerability_id": "V2",
        "vulnerability_name": "Preprocessing Attack Surface",
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "summary": "1 issue(s)" if status == "vulnerable" else "All clean"
    }}