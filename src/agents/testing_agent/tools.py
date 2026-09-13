import os
from typing import Any, Dict, List, Optional
import pandas as pd
from langchain_core.tools import tool


@tool
def inspect_dataset_profile(
    dataset_path: str,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Safely inspect the target dataset structure and metadata without executing untrusted code.
    Returns row counts, column names, class distributions, and text length statistics.
    """
    if not os.path.exists(dataset_path):
        return {"error": f"Dataset file not found at: {dataset_path}"}

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return {"error": f"Failed to parse CSV dataset: {str(e)}"}

    total_rows = len(df)
    columns = list(df.columns)

    # Detect text and label columns if not provided
    detected_text_col = text_column
    if not detected_text_col:
        for candidate in ["clean_text", "Comment", "text", "content", "abstract", "sentence"]:
            if candidate in columns:
                detected_text_col = candidate
                break
        if not detected_text_col and columns:
            detected_text_col = columns[0]

    detected_label_col = label_column
    if not detected_label_col:
        for candidate in ["Topic", "label", "target", "category", "class", "sentiment"]:
            if candidate in columns:
                detected_label_col = candidate
                break
        if not detected_label_col and len(columns) > 1:
            detected_label_col = columns[-1]

    profile: Dict[str, Any] = {
        "total_rows": total_rows,
        "columns": columns,
        "text_column": detected_text_col,
        "label_column": detected_label_col,
        "missing_values": {col: int(df[col].isna().sum()) for col in columns},
    }

    if detected_label_col and detected_label_col in df.columns:
        class_counts = df[detected_label_col].value_counts().to_dict()
        profile["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}
        profile["num_classes"] = len(class_counts)

    if detected_text_col and detected_text_col in df.columns:
        text_series = df[detected_text_col].dropna().astype(str)
        word_counts = text_series.apply(lambda s: len(s.split()))
        profile["text_stats"] = {
            "avg_word_count": round(float(word_counts.mean()), 1) if not word_counts.empty else 0,
            "max_word_count": int(word_counts.max()) if not word_counts.empty else 0,
            "min_word_count": int(word_counts.min()) if not word_counts.empty else 0,
        }

    return profile


@tool
def calculate_perturbation_budget(
    avg_word_count: float = 100.0,
    num_classes: int = 3,
    high_sparsity: bool = True,
) -> Dict[str, Any]:
    """
    Computes mathematically sound perturbation budgets and test sample sizes
    for adversarial evasion attacks (HopSkipJump) based on input feature properties.
    """
    # High-dimensional sparse text vectors (e.g. TF-IDF 5000 features) are sensitive
    # to small L2 shifts across many dimensions; lower budget ensures realistic perturbations.
    if high_sparsity or avg_word_count < 50:
        recommended_budget = 0.35
        sample_size = 50
        max_iter = 40
        rationale = (
            "High-dimensional sparse text representation detected. A conservative relative "
            "perturbation budget (<= 0.35) avoids adding artificial tokens outside natural syntax."
        )
    elif avg_word_count > 250:
        recommended_budget = 0.50
        sample_size = 35
        max_iter = 50
        rationale = (
            "Long-form text corpus detected. Higher word volume permits a slightly larger "
            "perturbation budget (<= 0.50) across the token distribution."
        )
    else:
        recommended_budget = 0.40
        sample_size = 50
        max_iter = 45
        rationale = "Balanced text length profile. Standard moderate perturbation budget (0.40) applied."

    return {
        "recommended_max_relative_budget": recommended_budget,
        "recommended_sample_size": sample_size,
        "recommended_max_iter": max_iter,
        "num_classes": num_classes,
        "rationale": rationale,
    }


@tool
def resolve_threat_surface(
    agent1_findings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Maps Agent 1 static threat evaluation findings and statuses into corresponding
    dynamic testing modules (V1_poisoning, V2_preprocessing, V3_validation, V4_adversarial).
    Threats marked 'vulnerable' are targeted for exploit verification.
    Threats marked 'mitigated' are included for defense-resilience verification.
    Threats marked 'not_applicable' are skipped.
    """
    category_map = {
        "data poisoning": "V1_poisoning",
        "preprocessing attack surface": "V2_preprocessing",
        "data validation weaknesses": "V3_validation",
        "adversarial robustness": "V4_adversarial",
    }

    id_map = {
        "v1": "V1_poisoning",
        "v2": "V2_preprocessing",
        "v3": "V3_validation",
        "v4": "V4_adversarial",
    }

    planned_tests = set()
    mappings = []
    findings_list = agent1_findings or []

    for finding in findings_list:
        v_id = str(finding.get("vulnerability_id", "")).strip().lower()
        cat = str(finding.get("category", "")).strip().lower()
        status = str(finding.get("status", "vulnerable")).strip().lower()

        # Skip testing if the threat class is not applicable to the pipeline
        if status == "not_applicable":
            continue

        matched_test = id_map.get(v_id) or category_map.get(cat)
        if matched_test:
            planned_tests.add(matched_test)
            mappings.append({
                "vulnerability_id": finding.get("vulnerability_id"),
                "category": finding.get("category"),
                "status": status,
                "matched_dynamic_test": matched_test,
                "test_purpose": "exploit_verification" if status == "vulnerable" else "defense_bypass_check",
            })

    # Default to all standard tests only if no findings were supplied
    if not planned_tests and not findings_list:
        planned_tests = {"V1_poisoning", "V4_adversarial", "V2_preprocessing", "V3_validation"}

    # Maintain canonical ordering
    test_order = ["V1_poisoning", "V4_adversarial", "V2_preprocessing", "V3_validation"]
    ordered_planned = [t for t in test_order if t in planned_tests]

    return {
        "planned_tests": ordered_planned,
        "mappings": mappings,
        "total_tests_planned": len(ordered_planned),
    }


@tool
def inspect_agent1_hypotheses(
    vulnerability_id: Optional[str] = None,
    agent1_findings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Inspect specific static vulnerability claims and affected components reported by Agent 1.
    Call this to examine what flaws Agent 1 claimed exist before planning tests.

    Args:
        vulnerability_id: Optional filter (e.g. 'V1', 'V2', 'V3', 'V4'). If omitted, returns all findings.
        agent1_findings: Optional list of findings.
    """
    findings = agent1_findings or []
    if vulnerability_id:
        v_target = vulnerability_id.strip().upper()
        matched = [
            f for f in findings
            if str(f.get("vulnerability_id", "")).strip().upper() == v_target
        ]
        return {
            "vulnerability_id": vulnerability_id,
            "matched_findings": matched,
            "found": len(matched) > 0,
        }

    return {
        "all_findings": findings,
        "total": len(findings),
    }


COGNITIVE_PLANNING_TOOLS = [
    inspect_dataset_profile,
    calculate_perturbation_budget,
    resolve_threat_surface,
    inspect_agent1_hypotheses,
]


def make_cognitive_planning_tools(
    agent_1_results: Optional[Dict[str, Any]],
    dataset_path: str,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> List[Any]:
    """
    Factory creating cognitive planning tools pre-bound via closure to
    the current execution state artifacts.
    """
    agent1 = agent_1_results or {}
    vuln_findings = agent1.get("vulnerability_findings", {}).get("vulnerabilities", [])

    @tool
    def bound_inspect_dataset_profile(
        path: Optional[str] = None,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Inspect dataset properties (row counts, class distributions, text lengths).
        If arguments are omitted, defaults to the state dataset configuration.
        """
        target_path = path or dataset_path
        target_text = text_col or text_column
        target_label = label_col or label_column
        return inspect_dataset_profile.invoke({
            "dataset_path": target_path,
            "text_column": target_text,
            "label_column": target_label,
        })

    @tool
    def bound_resolve_threat_surface(
        custom_findings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Map Agent 1 static vulnerability findings to dynamic tests.
        Defaults to evaluating the actual findings received from Agent 1.
        """
        findings_to_use = custom_findings if custom_findings is not None else vuln_findings
        return resolve_threat_surface.invoke({"agent1_findings": findings_to_use})

    @tool
    def bound_inspect_agent1_hypotheses(
        vulnerability_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query Agent 1's qualitative vulnerability findings by ID (e.g. 'V1', 'V4') or retrieve all.
        """
        return inspect_agent1_hypotheses.invoke({
            "vulnerability_id": vulnerability_id,
            "agent1_findings": vuln_findings,
        })

    return [
        bound_inspect_dataset_profile,
        calculate_perturbation_budget,
        bound_resolve_threat_surface,
        bound_inspect_agent1_hypotheses,
    ]


@tool
def query_attack_telemetry(
    test_id: str,
    evidence_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Retrieve granular empirical test telemetry, degradation measurements,
    and attack logs for a specific dynamic test (V1, V2, V3, V4).
    """
    bundle = evidence_bundle or {}
    canonical_key = None
    for k in ["V1_poisoning", "V4_adversarial", "V2_preprocessing", "V3_validation"]:
        if test_id.lower() in k.lower():
            canonical_key = k
            break

    if not canonical_key or canonical_key not in bundle:
        return {
            "error": f"No empirical evidence found for test identifier '{test_id}'.",
            "available_tests": list(bundle.keys()),
        }

    evidence = bundle[canonical_key] or {}
    return {
        "test_id": canonical_key,
        "status": evidence.get("status", "unverified"),
        "severity": evidence.get("severity"),
        "summary": evidence.get("summary", ""),
        "raw_evidence": evidence.get("evidence", {}),
    }


@tool
def evaluate_hypothesis_correlation(
    vulnerability_id: str,
    static_status: str,
    empirical_status: str,
    empirical_severity: Optional[str] = None,
    accuracy_drop: Optional[float] = None,
    attack_success_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Mathematically correlate Agent 1's qualitative hypothesis with empirical test telemetry.
    Deduces whether the finding is a 'Confirmed Risk', 'False Positive (Mitigated)',
    'Hidden Risk', or 'Unverified'.
    """
    v_stat = static_status.strip().lower()
    e_stat = empirical_status.strip().lower()

    if e_stat == "unverified" or e_stat == "skipped":
        correlation = "Unverified"
        rationale = "Empirical test was inconclusive or skipped under Zero-Trust policy."
    elif v_stat == "vulnerable" and e_stat == "vulnerable":
        correlation = "Confirmed Risk"
        rationale = (
            f"Static vulnerability hypothesis empirically confirmed. "
            f"Severity: {empirical_severity or 'detected'}."
        )
    elif v_stat == "vulnerable" and e_stat == "not_vulnerable":
        correlation = "False Positive (Mitigated)"
        rationale = (
            "Empirical testing refuted the static hypothesis: pipeline successfully resisted "
            "or sanitized attack inputs under empirical evaluation."
        )
    elif v_stat in ("mitigated", "not_applicable") and e_stat == "vulnerable":
        correlation = "Hidden Risk"
        rationale = (
            "Pipeline was assumed defended or not applicable, but empirical tests demonstrated "
            "concrete susceptibility."
        )
    elif e_stat == "not_applicable":
        correlation = "Not Applicable"
        rationale = "Test confirmed to be not applicable to target pipeline structure."
    else:
        correlation = "False Positive (Mitigated)"
        rationale = "Model resisted attack conditions; no empirical degradation observed."

    metrics: Dict[str, Any] = {}
    if accuracy_drop is not None:
        metrics["accuracy_drop"] = accuracy_drop
    if attack_success_rate is not None:
        metrics["attack_success_rate"] = attack_success_rate

    return {
        "vulnerability_id": vulnerability_id,
        "correlation_status": correlation,
        "test_status": e_stat,
        "dynamic_severity": empirical_severity,
        "correlation_rationale": rationale,
        "metrics": metrics,
    }


FORENSIC_DIAGNOSTIC_TOOLS = [
    query_attack_telemetry,
    evaluate_hypothesis_correlation,
]


def make_forensic_diagnostic_tools(
    evidence_bundle: Dict[str, Any],
    agent_1_results: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Factory producing post-attack forensic diagnostic tools pre-bound to
    empirical test telemetry and upstream Agent 1 findings.
    """
    agent1 = agent_1_results or {}
    static_findings = agent1.get("vulnerability_findings", {}).get("vulnerabilities", [])
    static_by_id = {
        str(f.get("vulnerability_id", "")).strip().upper(): f
        for f in static_findings
    }

    @tool
    def bound_query_attack_telemetry(test_id: str) -> Dict[str, Any]:
        """
        Query detailed telemetry, degradation metrics, and logs for a specific dynamic test (V1, V2, V3, V4).
        """
        return query_attack_telemetry.invoke({
            "test_id": test_id,
            "evidence_bundle": evidence_bundle,
        })

    @tool
    def bound_evaluate_hypothesis_correlation(
        vulnerability_id: str,
    ) -> Dict[str, Any]:
        """
        Correlate Agent 1's qualitative claim for a vulnerability ID against empirical telemetry.
        Automatically retrieves Agent 1's static status and compares it with empirical measurements.
        """
        v_upper = vulnerability_id.strip().upper()
        static_f = static_by_id.get(v_upper, {})
        static_status = static_f.get("status", "vulnerable")

        canonical_key = next(
            (k for k in evidence_bundle if v_upper in k.upper()),
            None
        )
        ev = evidence_bundle.get(canonical_key, {}) if canonical_key else {}
        emp_status = ev.get("status", "unverified")
        emp_sev = ev.get("severity")

        raw_ev = ev.get("evidence", {})
        acc_drop = raw_ev.get("accuracy_drop") or raw_ev.get("generic_test", {}).get("max_accuracy_drop")
        asr = raw_ev.get("attack_success_rate_within_budget") or raw_ev.get("attack_success_rate")

        return evaluate_hypothesis_correlation.invoke({
            "vulnerability_id": vulnerability_id,
            "static_status": static_status,
            "empirical_status": emp_status,
            "empirical_severity": emp_sev,
            "accuracy_drop": acc_drop,
            "attack_success_rate": asr,
        })

    return [
        bound_query_attack_telemetry,
        bound_evaluate_hypothesis_correlation,
    ]
