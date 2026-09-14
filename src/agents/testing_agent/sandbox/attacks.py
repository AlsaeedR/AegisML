import os
import random
import importlib.util
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
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


# =====================================================================
# Shared Pipeline & Model Helpers
# =====================================================================

def _get_classifier_step(model: Any, vectorizer_override: Optional[Any] = None) -> Tuple[Optional[Any], Any]:
    """
    Extracts the complete preprocessing pipeline and final classifier from an sklearn Pipeline.
    Supports a vectorizer_override if the vectorizer was serialized separately.
    """
    if hasattr(model, "steps"):
        steps = list(model.steps)
        if len(steps) > 1:
            preprocessor = Pipeline(steps[:-1])
            classifier = steps[-1][1]
            return preprocessor, classifier
    if vectorizer_override is not None:
        return vectorizer_override, model
    return None, model


def _prepare_art_classifier(classifier: Any) -> Any:
    art_model = deepcopy(classifier)
    if hasattr(art_model, "classes_"):
        classes = np.asarray(art_model.classes_)
        if not np.issubdtype(classes.dtype, np.integer):
            art_model.classes_ = np.arange(len(classes), dtype=int)
    return art_model


def _load_module_from_path(path: str):
    """Dynamically loads a python module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location("dynamic_pipeline_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def _to_serializable_prediction(prediction: Any) -> Any:
    if hasattr(prediction, "item"):
        try:
            return prediction.item()
        except Exception:
            pass
    return prediction


# =====================================================================
# V1: Data Poisoning Test
# =====================================================================

def _label_flip_at_fraction(
    model: Any,
    X_train,
    X_test,
    y_train,
    y_test,
    clean_acc: float,
    flip_fraction: float,
    rng: random.Random,
) -> Dict[str, Any]:
    labels_pool = sorted(set(y_train), key=str)
    if len(labels_pool) < 2:
        return {
            "flip_fraction": flip_fraction,
            "status": "inconclusive",
            "reason": "Label-flip poisoning requires at least two target classes.",
        }

    y_train_poisoned = list(y_train)
    n_flip = min(
        len(y_train_poisoned),
        max(1, int(len(y_train_poisoned) * flip_fraction)),
    )
    flip_indices = rng.sample(range(len(y_train_poisoned)), n_flip)

    for idx in flip_indices:
        current = y_train_poisoned[idx]
        alt_labels = [label for label in labels_pool if label != current]
        if alt_labels:
            y_train_poisoned[idx] = rng.choice(alt_labels)

    poisoned_model = clone(model)
    poisoned_model.fit(X_train, y_train_poisoned)

    poisoned_preds = poisoned_model.predict(X_test)
    poisoned_acc = accuracy_score(y_test, poisoned_preds)
    raw_accuracy_drop = clean_acc - poisoned_acc
    accuracy_drop = max(0.0, raw_accuracy_drop)

    return {
        "flip_fraction": flip_fraction,
        "n_labels_flipped": n_flip,
        "poisoned_accuracy": round(float(poisoned_acc), 4),
        "accuracy_drop": round(float(accuracy_drop), 4),
    }


def _generic_label_flip_test(
    model: Any,
    X_text: List[str],
    y_true: List[Any],
    flip_fractions: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if flip_fractions is None:
        flip_fractions = [0.05, 0.15, 0.30]

    if len(X_text) < 10:
        return {
            "method": "generic_label_flip_retrain",
            "status": "inconclusive",
            "reason": "Insufficient samples for a reliable poisoning test.",
        }

    if len(set(y_true)) < 2:
        return {
            "method": "generic_label_flip_retrain",
            "status": "inconclusive",
            "reason": "At least two target classes are required for label-flip poisoning.",
        }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_true, test_size=0.3, random_state=42, stratify=y_true
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_true, test_size=0.3, random_state=42
        )

    clean_model = clone(model)
    clean_model.fit(X_train, y_train)

    clean_preds = clean_model.predict(X_test)
    clean_acc = accuracy_score(y_test, clean_preds)

    rng = random.Random(42)
    trend = [
        _label_flip_at_fraction(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            clean_acc=clean_acc,
            flip_fraction=frac,
            rng=rng,
        )
        for frac in flip_fractions
    ]

    valid_results = [item for item in trend if "accuracy_drop" in item]
    if not valid_results:
        return {
            "method": "generic_label_flip_retrain",
            "status": "inconclusive",
            "clean_accuracy": round(float(clean_acc), 4),
            "flip_trend": trend,
            "reason": "No valid poisoning measurements were produced.",
        }

    headline = min(valid_results, key=lambda item: abs(item["flip_fraction"] - 0.15))
    worst = max(valid_results, key=lambda item: item["accuracy_drop"])
    max_accuracy_drop = worst["accuracy_drop"]

    effective_results = [item for item in valid_results if item["accuracy_drop"] > 0.05]
    minimum_effective_flip_fraction = (
        min(item["flip_fraction"] for item in effective_results)
        if effective_results
        else None
    )

    sorted_results = sorted(valid_results, key=lambda item: item["flip_fraction"])
    is_monotonic = all(
        sorted_results[i]["accuracy_drop"] <= sorted_results[i + 1]["accuracy_drop"] + 1e-9
        for i in range(len(sorted_results) - 1)
    )

    return {
        "method": "generic_label_flip_retrain",
        "clean_accuracy": round(float(clean_acc), 4),
        "flip_trend": trend,
        "headline_flip_fraction": headline["flip_fraction"],
        "headline_accuracy_drop": headline["accuracy_drop"],
        "accuracy_drop": headline["accuracy_drop"],
        "max_accuracy_drop": max_accuracy_drop,
        "worst_flip_fraction": worst["flip_fraction"],
        "minimum_effective_flip_fraction": minimum_effective_flip_fraction,
        "drop_increases_with_more_poisoning": is_monotonic,
        "interpretation": (
            "The model showed measurable performance degradation under "
            "label-flip poisoning. The result indicates empirical susceptibility "
            "to training-data label corruption under the evaluated conditions."
            if max_accuracy_drop > 0.05
            else
            "No material performance degradation was observed under "
            "label-flip poisoning across the evaluated poisoning fractions."
        ),
    }


def _art_svm_poisoning_test(
    classifier: Any,
    X_vec,
    y_true: List[Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(classifier, SVC):
        return None

    try:
        from art.estimators.classification.scikitlearn import ScikitlearnSVC
        from art.attacks.poisoning import PoisoningAttackSVM
    except ImportError:
        return {
            "method": "art_poisoning_attack_svm",
            "status": "skipped",
            "reason": "adversarial-robustness-toolbox (ART) is not installed.",
        }

    try:
        X_arr = (
            np.asarray(X_vec.todense(), dtype=np.float32)
            if hasattr(X_vec, "todense")
            else np.asarray(X_vec, dtype=np.float32)
        )
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if len(X_arr) == 0:
            return {
                "method": "art_poisoning_attack_svm",
                "status": "skipped",
                "reason": "No transformed samples were available for ART testing.",
            }

        y_arr = np.asarray(y_true)
        if len(y_arr) != len(X_arr):
            return {
                "method": "art_poisoning_attack_svm",
                "status": "skipped",
                "reason": "Feature and target sample counts do not match for ART poisoning analysis.",
            }

        unique_labels = sorted(set(y_arr.tolist()), key=str)
        label_to_int = {label: index for index, label in enumerate(unique_labels)}
        y_art = np.asarray([label_to_int[label] for label in y_arr], dtype=np.int64)

        art_model = clone(classifier)
        art_model.fit(X_arr, y_art)

        art_classifier = ScikitlearnSVC(model=art_model)
        validation_size = min(10, len(X_arr))

        attack = PoisoningAttackSVM(
            classifier=art_classifier,
            step=0.1,
            eps=1.0,
            x_train=X_arr,
            y_train=y_art,
            x_val=X_arr[:validation_size],
            y_val=y_art[:validation_size],
            max_iter=20,
        )
        poison_point, _ = attack.poison(X_arr[:1], y_art[:1])

        return {
            "method": "art_poisoning_attack_svm",
            "status": "completed",
            "generated_points": int(len(poison_point)),
            "note": (
                "ART successfully generated an SVM poisoning candidate. "
                "This auxiliary result does not independently determine "
                "the vulnerability status."
            ),
        }
    except Exception as e:
        return {
            "method": "art_poisoning_attack_svm",
            "status": "error",
            "reason": str(e),
        }


def run_poisoning_test(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes V1 Data Poisoning testing.
    Primary assessment is based on empirical label-flip retraining across multiple fractions.
    """
    model = state["model"]
    X_text = state["X_text"]
    y_true = state["y_true"]

    evidence: Dict[str, Any] = {}
    generic_result = _generic_label_flip_test(model, X_text, y_true)
    evidence["generic_test"] = generic_result

    preprocessor, classifier = _get_classifier_step(model)
    if preprocessor is not None:
        try:
            X_vec = preprocessor.transform(X_text)
            art_result = _art_svm_poisoning_test(classifier, X_vec, y_true)
            if art_result is not None:
                evidence["art_test"] = art_result
            else:
                evidence["art_test"] = {
                    "status": "skipped",
                    "reason": "The final estimator is not supported by the ART SVM poisoning test.",
                }
        except Exception as e:
            evidence["art_test"] = {"status": "error", "reason": str(e)}
    else:
        evidence["art_test"] = {
            "status": "skipped",
            "reason": "Could not extract a preprocessing pipeline for ART SVM poisoning analysis.",
        }

    if generic_result.get("status") == "inconclusive":
        status = "inconclusive"
        severity = None
    else:
        max_accuracy_drop = float(generic_result.get("max_accuracy_drop", 0.0))
        if max_accuracy_drop > 0.15:
            status = "vulnerable"
            severity = "high"
        elif max_accuracy_drop > 0.05:
            status = "vulnerable"
            severity = "medium"
        else:
            status = "not_vulnerable"
            severity = "low"

    return {
        "poisoning_evidence": {
            "vulnerability_id": "V1",
            "vulnerability_name": "Data Poisoning",
            "status": status,
            "severity": severity,
            "evidence": evidence,
        }
    }


# =====================================================================
# V4: Adversarial Robustness Test
# =====================================================================

def run_adversarial_test(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes V4 Adversarial Robustness evasion attack using HopSkipJump via ART.
    """
    model = state["model"]
    vectorizer_override = state.get("vectorizer")
    X_text = state["X_text"]
    y_true = state["y_true"]

    preprocessor, classifier = _get_classifier_step(model, vectorizer_override)
    if preprocessor is None:
        return {
            "adversarial_evidence": {
                "vulnerability_id": "V4",
                "vulnerability_name": "Adversarial Robustness",
                "status": "inconclusive",
                "severity": "low",
                "evidence": {
                    "status": "skipped",
                    "reason": "Could not extract or resolve a preprocessing step to produce numeric feature vectors.",
                },
            }
        }

    try:
        from art.estimators.classification import SklearnClassifier
        from art.attacks.evasion import HopSkipJump
    except ImportError:
        return {
            "adversarial_evidence": {
                "vulnerability_id": "V4",
                "vulnerability_name": "Adversarial Robustness",
                "status": "inconclusive",
                "severity": "low",
                "evidence": {
                    "status": "skipped",
                    "reason": "adversarial-robustness-toolbox (ART) is not installed in the environment.",
                },
            }
        }

    try:
        configured_sample_size = (state.get("adversarial_config") or {}).get("sample_size", 50)
        sample_size = min(configured_sample_size, len(X_text))
        sample_texts = X_text[:sample_size]

        X_vec = preprocessor.transform(sample_texts)
        X_arr = np.array(X_vec.todense()) if hasattr(X_vec, "todense") else np.asarray(X_vec)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        n_classes = len(classifier.classes_) if hasattr(classifier, "classes_") else len(set(y_true))
        clip_min = float(np.min(X_arr))
        clip_max = float(np.max(X_arr)) + 1e-06
        if clip_max <= clip_min:
            clip_max = clip_min + 1e-06

        art_model = _prepare_art_classifier(classifier)
        art_classifier = SklearnClassifier(model=art_model, clip_values=(clip_min, clip_max))

        original_preds = np.asarray(classifier.predict(X_arr))
        attack = HopSkipJump(
            classifier=art_classifier,
            targeted=False,
            max_iter=20,
            max_eval=200,
            init_eval=20,
        )
        X_adv = attack.generate(x=X_arr)
        adv_preds = np.asarray(classifier.predict(X_adv))

        flipped_mask = original_preds != adv_preds
        flipped = int(np.sum(flipped_mask))
        success_rate = round(flipped / sample_size, 4)

        max_relative_perturbation = 0.5
        original_norms = np.linalg.norm(X_arr, axis=1)
        perturbation_norms = np.linalg.norm(X_adv - X_arr, axis=1)
        zero_norm_epsilon = 1e-06
        valid_mask = original_norms > zero_norm_epsilon
        n_zero_norm_samples = int(np.sum(~valid_mask))

        relative_perturbations = np.full_like(perturbation_norms, np.inf, dtype=float)
        relative_perturbations[valid_mask] = (
            perturbation_norms[valid_mask] / original_norms[valid_mask]
        )

        within_budget_mask = flipped_mask & (relative_perturbations <= max_relative_perturbation)
        flipped_within_budget = int(np.sum(within_budget_mask))
        success_rate_within_budget = round(flipped_within_budget / sample_size, 4)

        avg_perturbation = float(np.mean(perturbation_norms))
        avg_relative_perturbation = (
            float(np.mean(relative_perturbations[valid_mask]))
            if np.any(valid_mask)
            else None
        )

        status = "vulnerable" if success_rate_within_budget >= 0.3 else "not_vulnerable"
        severity = (
            "high"
            if success_rate_within_budget >= 0.6
            else "medium"
            if success_rate_within_budget >= 0.3
            else "low"
        )

        return {
            "adversarial_evidence": {
                "vulnerability_id": "V4",
                "vulnerability_name": "Adversarial Robustness",
                "status": status,
                "severity": severity,
                "evidence": {
                    "method": "art_hopskipjump_evasion",
                    "n_samples_tested": sample_size,
                    "n_classes": n_classes,
                    "flipped_predictions_any_perturbation": flipped,
                    "success_rate_any_perturbation": success_rate,
                    "max_relative_perturbation_budget": max_relative_perturbation,
                    "flipped_predictions_within_budget": flipped_within_budget,
                    "attack_success_rate_within_budget": success_rate_within_budget,
                    "avg_perturbation_norm": round(avg_perturbation, 4),
                    "avg_relative_perturbation": (
                        round(avg_relative_perturbation, 4)
                        if avg_relative_perturbation is not None
                        else None
                    ),
                    "n_near_zero_vector_samples_excluded": n_zero_norm_samples,
                    "interpretation": (
                        "A significant portion of evaluation samples were flipped within a "
                        "realistic perturbation budget, confirming model susceptibility to "
                        "adversarial evasion."
                        if success_rate_within_budget >= 0.3
                        else "Attacks required unrealistically large perturbations; the model "
                        "proved empirically robust within normal perturbation boundaries."
                    ),
                },
            }
        }
    except Exception as e:
        return {
            "adversarial_evidence": {
                "vulnerability_id": "V4",
                "vulnerability_name": "Adversarial Robustness",
                "status": "inconclusive",
                "severity": "low",
                "evidence": {"status": "error", "reason": str(e)},
            }
        }


# =====================================================================
# V2: Preprocessing Attack Surface Test
# =====================================================================

def _get_external_preprocessor(state: Dict[str, Any]) -> Tuple[Optional[Callable[[str], Any]], Optional[str]]:
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


def _apply_external_preprocessor(preprocessor: Optional[Callable[[str], Any]], value: str) -> Any:
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
            for _, nested in transformer.transformer_list:
                if _contains_text_vectorizer(nested):
                    return True
        except Exception:
            pass

    if hasattr(transformer, "transformers"):
        try:
            for item in transformer.transformers:
                if len(item) >= 2:
                    nested = item[1]
                    if nested in ("drop", "passthrough"):
                        continue
                    if _contains_text_vectorizer(nested):
                        return True
        except Exception:
            pass

    if hasattr(transformer, "steps"):
        try:
            for _, nested in transformer.steps:
                if _contains_text_vectorizer(nested):
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
    for _, transformer in steps[:-1]:
        if _contains_text_vectorizer(transformer):
            return True
    return False


def _build_test_cases() -> List[Dict[str, str]]:
    return [
        {"name": "empty_input", "input": ""},
        {"name": "whitespace_only", "input": " " * 1000},
        {"name": "very_long_input", "input": "a" * 10000},
        {"name": "unicode_input", "input": "\U0001F600" * 500},
        {"name": "control_characters", "input": "\x00\x01\x02"},
        {"name": "repeated_punctuation", "input": "!@#$%^&*()_" * 500},
        {"name": "mixed_unicode_text", "input": "Hello \u0645\u0631\u062d\u0628\u0627 \u4f60\u597d \U0001F600 security test"},
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
            predictions = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
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
        predictions = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
        predictions = [_to_serializable_prediction(p) for p in predictions]
        return {"status": "passed", "prediction_count": len(predictions)}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def run_preprocess_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes V2 Preprocessing Attack Surface test.
    Injects malformed raw strings, edge-case Unicode, control characters, and checks resilience.
    """
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


# =====================================================================
# V3: Data Validation Weaknesses Test
# =====================================================================

def run_validation_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes V3 Data Validation Weaknesses test.
    Injects duplicate rows and missing values to assess data validation pipeline resilience.
    """
    model = state.get("model")
    X_text = state.get("X_text")
    y_true = state.get("y_true")

    if not model or not X_text or not y_true:
        return {
            "validation_evidence": {
                "vulnerability_id": "V3",
                "vulnerability_name": "Data Validation Weaknesses",
                "status": "inconclusive",
                "severity": "low",
                "evidence": {"status": "skipped", "reason": "Missing model or preloaded dataset"},
            }
        }

    try:
        rng = np.random.default_rng(42)
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

        return {
            "validation_evidence": {
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
            }
        }
    except Exception as e:
        return {
            "validation_evidence": {
                "vulnerability_id": "V3",
                "vulnerability_name": "Data Validation Weaknesses",
                "status": "inconclusive",
                "severity": "low",
                "evidence": {"status": "error", "reason": str(e)},
            }
        }

