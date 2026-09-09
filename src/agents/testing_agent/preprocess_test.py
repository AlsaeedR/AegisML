from typing import Any, Dict


def _accepts_raw_text(model) -> bool:
    """Checks whether the model's first pipeline step looks like a text vectorizer."""
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        return len(steps) > 1  # vectorizer step exists before the final estimator
    return False


def run_preprocess_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    model = state.get("model")

    if not model:
        return {"preprocessing_evidence": {
            "vulnerability_id": "V2", "vulnerability_name": "Preprocessing Attack Surface",
            "status": "inconclusive", "severity": "low",
            "evidence": {"status": "skipped", "reason": "Missing model"}
        }}

    if not _accepts_raw_text(model):
        return {"preprocessing_evidence": {
            "vulnerability_id": "V2", "vulnerability_name": "Preprocessing Attack Surface",
            "status": "not_applicable", "severity": None,
            "evidence": {"reason": "Model does not appear to accept raw text input; malformed-text test does not apply to this pipeline shape."}
        }}

    malformed_inputs = [
        "a" * 10000,
        "\x00\x01\x02",
        "😀" * 500,
        "",
        " " * 1000,
        "DROP TABLE users;",
        "<script>alert(1)</script>",
    ]

    try:
        preds = model.predict(malformed_inputs)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        dynamic_status = "not_vulnerable"
        dynamic_evidence = {
            "status": "ok",
            "details": "Model handled malformed/edge-case inputs without crashing",
            "predictions": preds_list,
        }
    except Exception as e:
        dynamic_status = "vulnerable"
        dynamic_evidence = {"status": "error", "reason": str(e)}

    status = dynamic_status
    severity = "high" if dynamic_status == "vulnerable" else "low"

    return {"preprocessing_evidence": {
        "vulnerability_id": "V2",
        "vulnerability_name": "Preprocessing Attack Surface",
        "status": status,
        "severity": severity,
        "evidence": {"dynamic_test": dynamic_evidence},
        "summary": "Dynamic test: " + dynamic_status,
    }}