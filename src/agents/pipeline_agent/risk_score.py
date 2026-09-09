def _severity_from_score(score: float) -> str:
    """Derives severity label from the final adjusted risk score, not the static rule."""
    if score >= 8.0:
        return "Critical"
    if score >= 6.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    return "Low"


def calculate_risk_score(vulnerabilities, deployment_context=None):
    """
    Rule-based scoring based on the NIST AI Risk Management Framework.
    """
    rules = {
        "Data Poisoning": {"impact": 9, "likelihood": 7},
        "Preprocessing Attack Surface": {"impact": 6, "likelihood": 8},
        "Data Validation Weaknesses": {"impact": 5, "likelihood": 9},
        "Adversarial Robustness": {"impact": 10, "likelihood": 9},
        "default": {"impact": 3, "likelihood": 3},
    }

    category_control_keywords = {
        "Data Poisoning": ["provenance", "outlier", "sanitiz", "dedup"],
        "Preprocessing Attack Surface": ["input validation", "sanitiz", "schema"],
        "Data Validation Weaknesses": ["schema", "validation", "dedup"],
        "Adversarial Robustness": ["adversarial training", "robust", "detection"],
    }

    existing_controls = []
    if deployment_context:
        existing_controls = [c.lower() for c in deployment_context.get("existing_controls", [])]

    for v in vulnerabilities:
        cat = v.get("category", "default")
        rule = rules.get(cat, rules["default"])
        impact = rule["impact"]
        likelihood = rule["likelihood"]

        matched_controls = [
            c for c in existing_controls
            if any(kw in c for kw in category_control_keywords.get(cat, []))
        ]
        if matched_controls:
            likelihood = max(1, likelihood - 3)

        n_components = len(v.get("affected_components", []))
        if n_components >= 3:
            impact = min(10, impact + 1)

        risk_score = round((impact * likelihood) / 10, 1)
        severity = _severity_from_score(risk_score)

        rationale = f"Base impact {rule['impact']}/likelihood {rule['likelihood']} for {cat}"
        if matched_controls:
            rationale += f"; likelihood reduced due to existing control(s): {matched_controls}"
        if n_components >= 3:
            rationale += f"; impact increased ({n_components} affected components)"
        rationale += f". Final: impact {impact}, likelihood {likelihood}, score {risk_score}, severity {severity}."

        v["severity"] = severity
        v["risk_score"] = risk_score
        v["score_rationale"] = rationale

    return vulnerabilities