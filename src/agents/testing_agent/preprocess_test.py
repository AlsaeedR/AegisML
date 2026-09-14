import os
import importlib.util
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

TEXT_VECTORIZER_TYPES = (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

PREPROCESSOR_NAMES = (
    "preprocess_text",
    "preprocess",
    "clean_text",
    "clean",
    "normalize_text",
    "normalize",
)


def _load_module_from_path(path: str):
    """Dynamically load a python module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location("dynamic_pipeline_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def _get_external_preprocessor(
    state: Dict[str, Any],
) -> Tuple[Optional[Callable[[str], Any]], Optional[str]]:
    candidate_keys = (
        "preprocess_function",
        "preprocessing_function",
        "text_preprocessor",
        "preprocessor",
    )

    for key in candidate_keys:
        candidate = state.get(key)
        if callable(candidate):
            name = getattr(candidate, "__name__", key)
            return candidate, name

    module = state.get("pipeline_module")

    # FIX: If module is not in state, dynamically load it from pipeline_path
    if module is None:
        pipeline_path = state.get("pipeline_path")
        if pipeline_path and os.path.exists(pipeline_path):
            module = _load_module_from_path(pipeline_path)

    if module is not None:
        for name in PREPROCESSOR_NAMES:
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate, name

    return None, None


def _apply_external_preprocessor(
    preprocessor: Optional[Callable[[str], Any]],
    value: str,
) -> Any:
    if preprocessor is None:
        return value
    return preprocessor(value)


def _contains_text_vectorizer(transformer) -> bool:
    if transformer is None:
        return False

    if isinstance(transformer, TEXT_VECTORIZER_TYPES):
        return True

    transformer_name = transformer.__class__.__name__.lower()
    text_indicators = (
        "vectorizer",
        "tokenizer",
        "text",
        "tfidf",
        "countvectorizer",
        "hashingvectorizer",
    )

    if any(indicator in transformer_name for indicator in text_indicators):
        return True

    if hasattr(transformer, "transformer_list"):
        try:
            for _, nested_transformer in transformer.transformer_list:
                if _contains_text_vectorizer(nested_transformer):
                    return True
        except Exception:
            pass

    if hasattr(transformer, "transformers"):
        try:
            for item in transformer.transformers:
                if len(item) >= 2:
                    nested_transformer = item[1]
                    if nested_transformer == "drop" or nested_transformer == "passthrough":
                        continue
                    if _contains_text_vectorizer(nested_transformer):
                        return True
        except Exception:
            pass

    if hasattr(transformer, "steps"):
        try:
            for _, nested_transformer in transformer.steps:
                if _contains_text_vectorizer(nested_transformer):
                    return True
        except Exception:
            pass

    return False


def _accepts_raw_text(model) -> bool:
    if not hasattr(model, "steps"):
        return False

    steps = list(model.steps)
    if len(steps) < 2:
        return False

    preprocessing_steps = steps[:-1]
    for _, transformer in preprocessing_steps:
        if _contains_text_vectorizer(transformer):
            return True

    return False


def _to_serializable_prediction(prediction: Any) -> Any:
    if hasattr(prediction, "item"):
        try:
            return prediction.item()
        except Exception:
            pass
    return prediction


def _build_test_cases() -> List[Dict[str, str]]:
    return [
        {"name": "empty_input", "input": ""},
        {"name": "whitespace_only", "input": " " * 1000},
        {"name": "very_long_input", "input": "a" * 10000},
        {"name": "unicode_input", "input": "😀" * 500},
        {"name": "control_characters", "input": "\x00\x01\x02"},
        {"name": "repeated_punctuation", "input": "!@#$%^&*()_" * 500},
        {"name": "mixed_unicode_text", "input": "Hello مرحبا 你好 😀 security test"},
    ]


def _run_individual_tests(
    model,
    test_cases: List[Dict[str, str]],
    preprocessor: Optional[Callable[[str], Any]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    semantic_case_names = {
        "very_long_input",
        "unicode_input",
        "control_characters",
        "repeated_punctuation",
        "mixed_unicode_text",
    }

    for case in test_cases:
        case_name = case["name"]
        test_input = case["input"]

        try:
            processed_input = _apply_external_preprocessor(preprocessor, test_input)
            semantic_flags: List[str] = []

            if (
                preprocessor is not None
                and isinstance(processed_input, str)
                and case_name in semantic_case_names
            ):
                raw_compact_length = len("".join(str(test_input).split()))
                processed_compact_length = len("".join(processed_input.split()))

                if raw_compact_length > 0 and processed_compact_length == 0:
                    semantic_flags.append("nonempty_input_collapsed_to_empty")
                elif raw_compact_length > 0:
                    retention_ratio = processed_compact_length / raw_compact_length
                    if (
                        case_name in {"unicode_input", "mixed_unicode_text"}
                        and retention_ratio < 0.5
                    ):
                        semantic_flags.append("substantial_character_loss")

            predictions = model.predict([processed_input])
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()
            else:
                predictions = list(predictions)

            prediction = predictions[0] if predictions else None

            result: Dict[str, Any] = {
                "case": case_name,
                "status": "passed",
                "prediction": _to_serializable_prediction(prediction),
                "semantic_flags": semantic_flags,
            }

            if preprocessor is not None and isinstance(processed_input, str):
                result.update({
                    "preprocessed_length": len(processed_input),
                    "preprocessed_preview": processed_input[:120],
                })

            results.append(result)

        except Exception as exc:
            results.append({
                "case": case_name,
                "status": "failed",
                "error": str(exc),
                "semantic_flags": [],
            })

    return results


def _run_batch_test(
    model,
    test_cases: List[Dict[str, str]],
    preprocessor: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    malformed_inputs = [case["input"] for case in test_cases]

    try:
        processed_inputs = [
            _apply_external_preprocessor(preprocessor, value) for value in malformed_inputs
        ]

        predictions = model.predict(processed_inputs)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        else:
            predictions = list(predictions)

        predictions = [_to_serializable_prediction(prediction) for prediction in predictions]

        return {"status": "passed", "prediction_count": len(predictions)}

    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def run_preprocess_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    model = state.get("model")
    external_preprocessor, external_preprocessor_name = _get_external_preprocessor(state)

    if model is None:
        return {
            "preprocessing_evidence": {
                "vulnerability_id": "V2",
                "vulnerability_name": "Preprocessing Attack Surface",
                "status": "inconclusive",
                "severity": None,
                "evidence": {"status": "skipped", "reason": "Model was not available."},
                "summary": "Dynamic preprocessing test was inconclusive.",
            }
        }

    model_accepts_raw_text = _accepts_raw_text(model)

    if not model_accepts_raw_text and external_preprocessor is None:
        return {
            "preprocessing_evidence": {
                "vulnerability_id": "V2",
                "vulnerability_name": "Preprocessing Attack Surface",
                "status": "not_applicable",
                "severity": None,
                "evidence": {"status": "not_applicable", "reason": "No raw text acceptance."},
                "summary": "Dynamic malformed-text preprocessing test was not applicable.",
            }
        }

    test_cases = _build_test_cases()
    individual_results = _run_individual_tests(model, test_cases, external_preprocessor)
    batch_result = _run_batch_test(model, test_cases, external_preprocessor)

    failed_cases = [r for r in individual_results if r.get("status") == "failed"]
    total_cases = len(individual_results)
    failure_count = len(failed_cases)
    failure_rate = failure_count / total_cases if total_cases else 0.0

    semantic_results = [r for r in individual_results if r.get("semantic_flags")]
    semantic_issue_count = len(semantic_results)

    semantic_evaluable_cases = [
        r for r in individual_results 
        if r.get("case") in {
            "very_long_input", "unicode_input", "control_characters",
            "repeated_punctuation", "mixed_unicode_text",
        }
    ]
    semantic_evaluable_count = len(semantic_evaluable_cases)
    semantic_issue_rate = (
        semantic_issue_count / semantic_evaluable_count
        if semantic_evaluable_count else 0.0
    )

    batch_failed = batch_result.get("status") == "failed"

    if failure_count == 0 and not batch_failed and semantic_issue_count == 0:
        status = "not_vulnerable"
        severity = "low"
        interpretation = "Handled all malformed edge-case inputs without runtime exceptions or significant character loss."
    elif failure_count > 0 or batch_failed:
        status = "vulnerable"
        if failure_rate >= 0.5 or (batch_failed and failure_count > 1):
            severity = "high"
        else:
            severity = "medium"
        interpretation = f"{failure_count}/{total_cases} malformed edge-case input(s) triggered unhandled runtime exceptions."
    else:
        # Handled without runtime crashes, but semantic normalization flags were triggered
        status = "vulnerable"
        if semantic_issue_rate >= 0.5:
            severity = "medium"
        else:
            severity = "low"
        interpretation = "Pipeline executed without runtime crashes, but input normalization loss or character collapse was observed across non-Latin/edge-case inputs."

    evidence = {
        "method": "Malformed and edge-case raw-text preprocessing robustness test",
        "execution_path": (
            "external_preprocessor_then_model" 
            if external_preprocessor is not None else "serialized_model_pipeline"
        ),
        "external_preprocessor": external_preprocessor_name,
        "total_test_cases": total_cases,
        "failed_test_cases": failure_count,
        "failure_rate": round(failure_rate, 4),
        "semantic_issue_count": semantic_issue_count,
        "semantic_evaluable_cases": semantic_evaluable_count,
        "semantic_issue_rate": round(semantic_issue_rate, 4),
        "semantic_issue_cases": [
            {"case": r.get("case"), "semantic_flags": r.get("semantic_flags", [])}
            for r in semantic_results
        ],
        "individual_tests": individual_results,
        "batch_test": batch_result,
        "interpretation": interpretation,
        "scope_note": "Evaluates malformed raw-text handling at the ML preprocessing boundary.",
    }

    return {
        "preprocessing_evidence": {
            "vulnerability_id": "V2",
            "vulnerability_name": "Preprocessing Attack Surface",
            "status": status,
            "severity": severity,
            "evidence": evidence,
            "summary": (
                f"Dynamic preprocessing robustness test: {status}; "
                f"{failure_count}/{total_cases} individual cases failed; "
                f"{semantic_issue_count}/{semantic_evaluable_count} semantic checks triggered."
            ),
        }
    }