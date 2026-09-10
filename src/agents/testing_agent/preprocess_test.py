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


def _get_external_preprocessor(
    state: Dict[str, Any],
) -> Tuple[
    Optional[Callable[[str], Any]],
    Optional[str],
]:
    """
    Resolve an external text-preprocessing callable
    supplied by the surrounding AegisML execution
    state.

    The testing agent may pass a callable directly
    through one of the supported state keys. This
    allows V2 to evaluate projects where text
    cleaning happens before model.predict() rather
    than inside the serialized sklearn Pipeline.

    The model itself is not modified.
    """

    candidate_keys = (
        "preprocess_function",
        "preprocessing_function",
        "text_preprocessor",
        "preprocessor",
    )

    for key in candidate_keys:
        candidate = state.get(key)

        if callable(candidate):
            name = getattr(
                candidate,
                "__name__",
                key,
            )
            return candidate, name

    module = state.get("pipeline_module")

    if module is not None:
        for name in PREPROCESSOR_NAMES:
            candidate = getattr(
                module,
                name,
                None,
            )

            if callable(candidate):
                return candidate, name

    return None, None


def _apply_external_preprocessor(
    preprocessor: Optional[Callable[[str], Any]],
    value: str,
) -> Any:
    """
    Apply an external preprocessing callable when
    available. Otherwise return the original value.
    """

    if preprocessor is None:
        return value

    return preprocessor(value)


def _contains_text_vectorizer(transformer) -> bool:
    """
    Recursively inspect sklearn-style preprocessing
    containers to determine whether they contain a
    raw-text vectorizer or text-oriented transformer.

    This supports nested structures such as:

        Pipeline
            -> FeatureUnion
                -> TfidfVectorizer
                -> TfidfVectorizer
                -> TfidfVectorizer

    It also keeps the previous behavior for ordinary
    pipelines whose first preprocessing step is a
    direct text vectorizer.
    """

    if transformer is None:
        return False

    if isinstance(
        transformer,
        TEXT_VECTORIZER_TYPES,
    ):
        return True

    transformer_name = (
        transformer.__class__.__name__.lower()
    )

    text_indicators = (
        "vectorizer",
        "tokenizer",
        "text",
        "tfidf",
        "countvectorizer",
        "hashingvectorizer",
    )

    if any(
        indicator in transformer_name
        for indicator in text_indicators
    ):
        return True

    if hasattr(
        transformer,
        "transformer_list",
    ):
        try:
            for _, nested_transformer in (
                transformer.transformer_list
            ):
                if _contains_text_vectorizer(
                    nested_transformer
                ):
                    return True
        except Exception:
            pass

    if hasattr(
        transformer,
        "transformers",
    ):
        try:
            for item in transformer.transformers:
                if len(item) >= 2:
                    nested_transformer = item[1]

                    if (
                        nested_transformer == "drop"
                        or nested_transformer
                        == "passthrough"
                    ):
                        continue

                    if _contains_text_vectorizer(
                        nested_transformer
                    ):
                        return True
        except Exception:
            pass

    if hasattr(
        transformer,
        "steps",
    ):
        try:
            for _, nested_transformer in (
                transformer.steps
            ):
                if _contains_text_vectorizer(
                    nested_transformer
                ):
                    return True
        except Exception:
            pass

    return False


def _accepts_raw_text(model) -> bool:
    """
    Determine whether the uploaded model appears to
    accept raw text directly.

    V2 is intended to test malformed and edge-case
    text handling at the preprocessing boundary.
    Numeric-only pipelines are therefore treated as
    not applicable for this specific dynamic test.

    The detection supports both direct text
    vectorizers and nested sklearn preprocessing
    structures such as FeatureUnion, ColumnTransformer,
    and nested Pipeline objects.
    """

    if not hasattr(model, "steps"):
        return False

    steps = list(model.steps)

    if len(steps) < 2:
        return False

    preprocessing_steps = steps[:-1]

    for _, transformer in preprocessing_steps:
        if _contains_text_vectorizer(
            transformer
        ):
            return True

    return False


def _to_serializable_prediction(
    prediction: Any,
) -> Any:
    """
    Convert NumPy/scikit-learn prediction values
    into JSON-serializable Python values.
    """

    if hasattr(
        prediction,
        "item",
    ):
        try:
            return prediction.item()
        except Exception:
            pass

    return prediction


def _build_test_cases() -> List[Dict[str, str]]:
    """
    Define malformed and edge-case raw-text inputs.

    These cases test preprocessing robustness only.
    They are not intended to claim SQL injection,
    XSS, or application-layer exploitability.
    """

    return [
        {
            "name": "empty_input",
            "input": "",
        },
        {
            "name": "whitespace_only",
            "input": " " * 1000,
        },
        {
            "name": "very_long_input",
            "input": "a" * 10000,
        },
        {
            "name": "unicode_input",
            "input": "😀" * 500,
        },
        {
            "name": "control_characters",
            "input": "\x00\x01\x02",
        },
        {
            "name": "repeated_punctuation",
            "input": "!@#$%^&*()_" * 500,
        },
        {
            "name": "mixed_unicode_text",
            "input": (
                "Hello مرحبا 你好 😀 "
                "security test"
            ),
        },
    ]


def _run_individual_tests(
    model,
    test_cases: List[Dict[str, str]],
    preprocessor: Optional[
        Callable[[str], Any]
    ] = None,
) -> List[Dict[str, Any]]:
    """
    Run each malformed input independently.

    Besides prediction failures, V2 records observable
    preprocessing degradation when an external text
    preprocessor is available.

    The semantic checks intentionally focus on
    measurable transformations rather than claiming
    application-layer injection vulnerabilities.
    """

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
            processed_input = (
                _apply_external_preprocessor(
                    preprocessor,
                    test_input,
                )
            )

            semantic_flags: List[str] = []

            if (
                preprocessor is not None
                and isinstance(
                    processed_input,
                    str,
                )
                and case_name in semantic_case_names
            ):
                raw_compact_length = len(
                    "".join(
                        str(test_input).split()
                    )
                )

                processed_compact_length = len(
                    "".join(
                        processed_input.split()
                    )
                )

                if (
                    raw_compact_length > 0
                    and processed_compact_length == 0
                ):
                    semantic_flags.append(
                        "nonempty_input_collapsed_to_empty"
                    )

                elif raw_compact_length > 0:
                    retention_ratio = (
                        processed_compact_length
                        / raw_compact_length
                    )

                    if (
                        case_name
                        in {
                            "unicode_input",
                            "mixed_unicode_text",
                        }
                        and retention_ratio < 0.5
                    ):
                        semantic_flags.append(
                            "substantial_character_loss"
                        )

            predictions = model.predict(
                [processed_input]
            )

            if hasattr(
                predictions,
                "tolist",
            ):
                predictions = (
                    predictions.tolist()
                )
            else:
                predictions = list(
                    predictions
                )

            prediction = (
                predictions[0]
                if predictions
                else None
            )

            result: Dict[str, Any] = {
                "case": case_name,
                "status": "passed",
                "prediction": (
                    _to_serializable_prediction(
                        prediction
                    )
                ),
                "semantic_flags": semantic_flags,
            }

            if (
                preprocessor is not None
                and isinstance(
                    processed_input,
                    str,
                )
            ):
                result.update(
                    {
                        "preprocessed_length": len(
                            processed_input
                        ),
                        "preprocessed_preview": (
                            processed_input[:120]
                        ),
                    }
                )

            results.append(
                result
            )

        except Exception as exc:
            results.append(
                {
                    "case": case_name,
                    "status": "failed",
                    "error": str(exc),
                    "semantic_flags": [],
                }
            )

    return results

def _run_batch_test(
    model,
    test_cases: List[Dict[str, str]],
    preprocessor: Optional[
        Callable[[str], Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Test whether a batch containing several
    malformed inputs can disrupt preprocessing
    for the entire prediction request.

    External preprocessing is applied to each item
    before the batch is passed to model.predict().
    """

    malformed_inputs = [
        case["input"]
        for case in test_cases
    ]

    try:
        processed_inputs = [
            _apply_external_preprocessor(
                preprocessor,
                value,
            )
            for value in malformed_inputs
        ]

        predictions = model.predict(
            processed_inputs
        )

        if hasattr(
            predictions,
            "tolist",
        ):
            predictions = (
                predictions.tolist()
            )
        else:
            predictions = list(
                predictions
            )

        predictions = [
            _to_serializable_prediction(
                prediction
            )
            for prediction in predictions
        ]

        return {
            "status": "passed",
            "prediction_count": len(
                predictions
            ),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }

def run_preprocess_checks(
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run the V2 Preprocessing Attack Surface test.

    The test evaluates whether a raw-text ML
    pipeline safely handles malformed and
    edge-case text inputs without preprocessing
    failures.

    This test measures preprocessing robustness.
    It does not claim application-layer injection
    vulnerabilities such as SQL injection or XSS.
    """

    model = state.get(
        "model"
    )

    (
        external_preprocessor,
        external_preprocessor_name,
    ) = _get_external_preprocessor(
        state
    )

    if model is None:
        return {
            "preprocessing_evidence": {
                "vulnerability_id": "V2",
                "vulnerability_name": (
                    "Preprocessing Attack Surface"
                ),
                "status": "inconclusive",
                "severity": None,
                "evidence": {
                    "status": "skipped",
                    "reason": (
                        "Model was not available "
                        "for dynamic preprocessing "
                        "testing."
                    ),
                },
                "summary": (
                    "Dynamic preprocessing test "
                    "was inconclusive because the "
                    "model was unavailable."
                ),
            }
        }

    model_accepts_raw_text = (
        _accepts_raw_text(
            model
        )
    )

    if (
        not model_accepts_raw_text
        and external_preprocessor is None
    ):
        return {
            "preprocessing_evidence": {
                "vulnerability_id": "V2",
                "vulnerability_name": (
                    "Preprocessing Attack Surface"
                ),
                "status": "not_applicable",
                "severity": None,
                "evidence": {
                    "status": "not_applicable",
                    "reason": (
                        "The model does not appear "
                        "to accept raw text directly, "
                        "and no external text "
                        "preprocessing function was "
                        "available for dynamic testing."
                    ),
                },
                "summary": (
                    "Dynamic malformed-text "
                    "preprocessing test was not "
                    "applicable."
                ),
            }
        }

    test_cases = _build_test_cases()

    individual_results = (
        _run_individual_tests(
            model,
            test_cases,
            external_preprocessor,
        )
    )

    batch_result = _run_batch_test(
        model,
        test_cases,
        external_preprocessor,
    )

    failed_cases = [
        result
        for result in individual_results
        if result.get(
            "status"
        )
        == "failed"
    ]

    total_cases = len(
        individual_results
    )

    failure_count = len(
        failed_cases
    )

    failure_rate = (
        failure_count / total_cases
        if total_cases
        else 0.0
    )

    semantic_results = [
        result
        for result in individual_results
        if result.get(
            "semantic_flags"
        )
    ]

    semantic_issue_count = len(
        semantic_results
    )

    semantic_evaluable_cases = [
        result
        for result in individual_results
        if result.get("case")
        in {
            "very_long_input",
            "unicode_input",
            "control_characters",
            "repeated_punctuation",
            "mixed_unicode_text",
        }
    ]

    semantic_evaluable_count = len(
        semantic_evaluable_cases
    )

    semantic_issue_rate = (
        semantic_issue_count
        / semantic_evaluable_count
        if semantic_evaluable_count
        else 0.0
    )

    batch_failed = (
        batch_result.get(
            "status"
        )
        == "failed"
    )

    if (
        failure_count == 0
        and not batch_failed
        and semantic_issue_count == 0
    ):
        status = "not_vulnerable"
        severity = "low"

        interpretation = (
            "The tested preprocessing path handled "
            "all malformed and edge-case raw-text "
            "inputs without prediction failures or "
            "observable preprocessing degradation "
            "under the configured checks."
        )

    else:
        status = "vulnerable"

        if (
            failure_rate >= 0.5
            or (
                batch_failed
                and failure_count > 0
            )
            or semantic_issue_rate >= 0.5
        ):
            severity = "high"

        else:
            severity = "medium"

        if failure_count > 0 or batch_failed:
            interpretation = (
                "One or more malformed or edge-case "
                "raw-text inputs caused preprocessing "
                "or prediction failures, indicating "
                "insufficient robustness at the model "
                "input boundary."
            )

        else:
            interpretation = (
                "Dynamic testing observed measurable "
                "preprocessing degradation even though "
                "prediction calls completed successfully. "
                "One or more non-empty edge-case inputs "
                "collapsed to an empty representation or "
                "lost a substantial portion of their "
                "content during external preprocessing."
            )

    evidence = {
        "method": (
            "Malformed and edge-case raw-text "
            "preprocessing robustness and degradation test"
        ),
        "execution_path": (
            "external_preprocessor_then_model"
            if external_preprocessor is not None
            else "serialized_model_pipeline"
        ),
        "external_preprocessor": (
            external_preprocessor_name
        ),
        "total_test_cases": total_cases,
        "failed_test_cases": failure_count,
        "failure_rate": round(
            failure_rate,
            4,
        ),
        "semantic_issue_count": (
            semantic_issue_count
        ),
        "semantic_evaluable_cases": (
            semantic_evaluable_count
        ),
        "semantic_issue_rate": round(
            semantic_issue_rate,
            4,
        ),
        "semantic_issue_cases": [
            {
                "case": result.get("case"),
                "semantic_flags": (
                    result.get(
                        "semantic_flags",
                        [],
                    )
                ),
            }
            for result in semantic_results
        ],
        "individual_tests": (
            individual_results
        ),
        "batch_test": batch_result,
        "interpretation": interpretation,
        "scope_note": (
            "This dynamic test evaluates malformed "
            "raw-text handling at the ML preprocessing "
            "boundary, including an external text "
            "preprocessor when one is supplied by the "
            "execution state. It does not test or claim SQL "
            "injection, XSS, or other application-layer "
            "injection vulnerabilities."
        ),
    }

    return {
        "preprocessing_evidence": {
            "vulnerability_id": "V2",
            "vulnerability_name": (
                "Preprocessing Attack Surface"
            ),
            "status": status,
            "severity": severity,
            "evidence": evidence,
            "summary": (
                "Dynamic preprocessing robustness "
                f"test: {status}; "
                f"{failure_count}/{total_cases} "
                "individual malformed-input cases "
                "failed; "
                f"{semantic_issue_count}/"
                f"{semantic_evaluable_count} "
                "semantic degradation checks "
                "were triggered."
            ),
        }
    }