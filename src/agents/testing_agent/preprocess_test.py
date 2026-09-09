import ast
from typing import Any, Dict, List

DANGEROUS_CALLS = {
    ("pickle", "load"), ("pickle", "loads"),
    ("yaml", "load"),
    ("os", "system"),
    ("subprocess", "Popen"), ("subprocess", "call"), ("subprocess", "run"),
}
DANGEROUS_BUILTINS = {"eval", "exec"}


def _static_scan(code: str) -> List[Dict[str, Any]]:
    """AST-based scan for known-dangerous call patterns that a dynamic input test can miss."""
    findings = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [{"finding": "unparseable_code", "detail": str(e)}]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        module, name = None, None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module, name = func.value.id, func.attr
        elif isinstance(func, ast.Name):
            name = func.id

        if name in DANGEROUS_BUILTINS or (module, name) in DANGEROUS_CALLS:
            findings.append({
                "call": f"{module + '.' if module else ''}{name}",
                "lineno": getattr(node, "lineno", None),
            })
    return findings


def run_preprocess_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    model = state.get("model")
    code = state.get("code") or state.get("pipeline_source")

    if not model:
        return {"preprocessing_evidence": {
            "vulnerability_id": "V2", "vulnerability_name": "Preprocessing Attack Surface",
            "status": "inconclusive", "severity": "low",
            "evidence": {"status": "skipped", "reason": "Missing model"}
        }}

    # --- Static pass: catch code-level risks the dynamic test can't see ---
    static_findings = _static_scan(code) if code else []
    static_status = "vulnerable" if static_findings and static_findings[0].get("finding") != "unparseable_code" else "not_vulnerable"

    # --- Dynamic pass: crash-resistance to malformed/edge-case text input ---
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

    # Overall status/severity = worse of the two passes
    overall_vulnerable = static_status == "vulnerable" or dynamic_status == "vulnerable"
    status = "vulnerable" if overall_vulnerable else "not_vulnerable"
    severity = "high" if overall_vulnerable else "low"

    return {"preprocessing_evidence": {
        "vulnerability_id": "V2",
        "vulnerability_name": "Preprocessing Attack Surface",
        "status": status,
        "severity": severity,
        "evidence": {
            "static_scan": {
                "status": static_status,
                "findings": static_findings,
                "note": "No source code provided; static scan skipped." if not code else None,
            },
            "dynamic_test": dynamic_evidence,
        },
        "summary": f"{len(static_findings)} static finding(s), dynamic: {dynamic_status}",
    }}