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
    Safely inspect the target dataset structure and metadata without executing code.
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
    agent1_findings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Maps Agent 1 static vulnerability categories and IDs into corresponding
    dynamic testing modules (V1_poisoning, V2_preprocessing, V3_validation, V4_adversarial).
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

    for finding in agent1_findings:
        v_id = str(finding.get("vulnerability_id", "")).strip().lower()
        cat = str(finding.get("category", "")).strip().lower()

        matched_test = id_map.get(v_id) or category_map.get(cat)
        if matched_test:
            planned_tests.add(matched_test)
            mappings.append({
                "vulnerability_id": finding.get("vulnerability_id"),
                "category": finding.get("category"),
                "matched_dynamic_test": matched_test,
            })

    # Default to all standard tests if no explicit mappings could be deduced
    if not planned_tests:
        planned_tests = {"V1_poisoning", "V4_adversarial", "V2_preprocessing", "V3_validation"}

    # Maintain canonical ordering
    test_order = ["V1_poisoning", "V4_adversarial", "V2_preprocessing", "V3_validation"]
    ordered_planned = [t for t in test_order if t in planned_tests]

    return {
        "planned_tests": ordered_planned,
        "mappings": mappings,
        "total_tests_planned": len(ordered_planned),
    }


COGNITIVE_PLANNING_TOOLS = [
    inspect_dataset_profile,
    calculate_perturbation_budget,
    resolve_threat_surface,
]

