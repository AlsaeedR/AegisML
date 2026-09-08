def calculate_risk_score(vulnerabilities):
    """
    Rule-based scoring based on the NIST AI Risk Management Framework.
    Returns a dictionary with 'severity' and 'risk_score'.
    """
    # (Impact * Likelihood) / 10 = Risk Score (0 to 10)
    # Define base rules for the categories found in your Agent 1
    rules = {
        "Data Poisoning": {"severity": "High", "impact": 9, "likelihood": 7},   # 6.3 score
        "Preprocessing Attack Surface": {"severity": "Medium", "impact": 6, "likelihood": 8}, # 4.8 score
        "Data Validation Weaknesses": {"severity": "Medium", "impact": 5, "likelihood": 9},    # 4.5 score
        "Adversarial Robustness": {"severity": "Critical", "impact": 10, "likelihood": 9},    # 9.0 score
        # Default fallback if LLM names a weird category:
        "default": {"severity": "Low", "impact": 3, "likelihood": 3}
    }

    for v in vulnerabilities:
        # Get the category, or use default
        cat = v.get("category", "default")
        rule = rules.get(cat, rules["default"])

        # Calculate score (Impact * Likelihood / 10)
        risk_score = round((rule["impact"] * rule["likelihood"]) / 10, 1)
        
        
        v["severity"] = rule["severity"]
        v["risk_score"] = risk_score
        v["score_rationale"] = f"Calculated based on Impact {rule['impact']} and Likelihood {rule['likelihood']} for {cat}."

    return vulnerabilities