from typing import Any, Dict, List, Optional
import random

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC


def _get_classifier_step(model: Any):
    """
    Extracts the complete preprocessing pipeline and final classifier
    from an sklearn Pipeline.
    """
    if hasattr(model, "steps"):
        steps = list(model.steps)

        if len(steps) > 1:
            preprocessor = Pipeline(steps[:-1])
            classifier = steps[-1][1]
            return preprocessor, classifier

    return None, model


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
    """
    Simulates label-flip poisoning at a specific fraction of the training set
    and measures the resulting validation accuracy degradation.
    """
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

    flip_indices = rng.sample(
        range(len(y_train_poisoned)),
        n_flip,
    )

    for idx in flip_indices:
        current = y_train_poisoned[idx]

        alt_labels = [
            label
            for label in labels_pool
            if label != current
        ]

        if alt_labels:
            y_train_poisoned[idx] = rng.choice(alt_labels)

    poisoned_model = clone(model)
    poisoned_model.fit(X_train, y_train_poisoned)

    poisoned_preds = poisoned_model.predict(X_test)
    poisoned_acc = accuracy_score(y_test, poisoned_preds)

    raw_accuracy_drop = clean_acc - poisoned_acc

    # A negative value means the poisoned model happened to perform
    # slightly better because of experimental variance. For degradation
    # scoring, this is treated as zero impact.
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
    """
    Simulates training-data poisoning by corrupting labels across multiple
    poisoning fractions and evaluating the complete degradation trend.
    """
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
            X_text,
            y_true,
            test_size=0.3,
            random_state=42,
            stratify=y_true,
        )
    except ValueError:
        # Some small or highly imbalanced datasets cannot be stratified.
        X_train, X_test, y_train, y_test = train_test_split(
            X_text,
            y_true,
            test_size=0.3,
            random_state=42,
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

    valid_results = [
        item
        for item in trend
        if "accuracy_drop" in item
    ]

    if not valid_results:
        return {
            "method": "generic_label_flip_retrain",
            "status": "inconclusive",
            "clean_accuracy": round(float(clean_acc), 4),
            "flip_trend": trend,
            "reason": "No valid poisoning measurements were produced.",
        }

    # Keep a representative middle point for display/reporting,
    # while final vulnerability assessment uses the complete trend.
    headline = min(
        valid_results,
        key=lambda item: abs(item["flip_fraction"] - 0.15),
    )

    worst = max(
        valid_results,
        key=lambda item: item["accuracy_drop"],
    )

    max_accuracy_drop = worst["accuracy_drop"]

    effective_results = [
        item
        for item in valid_results
        if item["accuracy_drop"] > 0.05
    ]

    minimum_effective_flip_fraction = (
        min(
            item["flip_fraction"]
            for item in effective_results
        )
        if effective_results
        else None
    )

    sorted_results = sorted(
        valid_results,
        key=lambda item: item["flip_fraction"],
    )

    is_monotonic = all(
        sorted_results[i]["accuracy_drop"]
        <= sorted_results[i + 1]["accuracy_drop"] + 1e-9
        for i in range(len(sorted_results) - 1)
    )

    return {
        "method": "generic_label_flip_retrain",
        "clean_accuracy": round(float(clean_acc), 4),
        "flip_trend": trend,

        # Representative result for report readability.
        "headline_flip_fraction": headline["flip_fraction"],
        "headline_accuracy_drop": headline["accuracy_drop"],

        # Backward-compatible alias for older dashboard/aggregation code.
        # Final V1 assessment uses max_accuracy_drop below.
        "accuracy_drop": headline["accuracy_drop"],

        # Evidence used for vulnerability assessment.
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
    """
    Executes an ART PoisoningAttackSVM attack when the final estimator
    is an SVM.

    This ART result is auxiliary evidence only. Final V1 vulnerability
    status is determined by the empirical label-flip degradation test.
    """
    if not isinstance(classifier, SVC):
        return None

    try:
        from art.estimators.classification.scikitlearn import ScikitlearnSVC
        from art.attacks.poisoning import PoisoningAttackSVM

    except ImportError:
        return {
            "method": "art_poisoning_attack_svm",
            "status": "skipped",
            "reason": (
                "adversarial-robustness-toolbox (ART) is not installed."
            ),
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
                "reason": (
                    "Feature and target sample counts do not match "
                    "for ART poisoning analysis."
                ),
            }

        # ART poisoning is supplementary only. Convert arbitrary class labels
        # to stable numeric labels for ART without modifying the original model.
        unique_labels = sorted(set(y_arr.tolist()), key=str)
        label_to_int = {
            label: index
            for index, label in enumerate(unique_labels)
        }
        y_art = np.asarray(
            [label_to_int[label] for label in y_arr],
            dtype=np.int64,
        )

        art_model = clone(classifier)
        art_model.fit(X_arr, y_art)

        art_classifier = ScikitlearnSVC(
            model=art_model
        )

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

        poison_point, _ = attack.poison(
            X_arr[:1],
            y_art[:1],
        )

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


def run_poisoning_test(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes V1 Data Poisoning testing.

    The primary assessment is based on empirical label-flip retraining
    across multiple poisoning fractions. ART SVM poisoning is included
    as supplementary evidence when supported by the model.
    """
    model = state["model"]
    X_text = state["X_text"]
    y_true = state["y_true"]

    evidence: Dict[str, Any] = {}

    generic_result = _generic_label_flip_test(
        model,
        X_text,
        y_true,
    )

    evidence["generic_test"] = generic_result

    # Run the complete sklearn preprocessing chain before passing
    # transformed features to the final classifier for ART analysis.
    preprocessor, classifier = _get_classifier_step(model)

    if preprocessor is not None:
        try:
            X_vec = preprocessor.transform(X_text)

            art_result = _art_svm_poisoning_test(
                classifier,
                X_vec,
                y_true,
            )

            if art_result is not None:
                evidence["art_test"] = art_result
            else:
                evidence["art_test"] = {
                    "status": "skipped",
                    "reason": (
                        "The final estimator is not supported by the "
                        "ART SVM poisoning test."
                    ),
                }

        except Exception as e:
            evidence["art_test"] = {
                "status": "error",
                "reason": str(e),
            }

    else:
        evidence["art_test"] = {
            "status": "skipped",
            "reason": (
                "Could not extract a preprocessing pipeline "
                "for ART SVM poisoning analysis."
            ),
        }

    # Inconclusive testing must not be interpreted as evidence
    # that the model is safe.
    if generic_result.get("status") == "inconclusive":
        status = "inconclusive"
        severity = None

    else:
        max_accuracy_drop = float(
            generic_result.get(
                "max_accuracy_drop",
                0.0,
            )
        )

        # AegisML assessment thresholds.
        # These are project-defined empirical severity boundaries,
        # not NIST-prescribed thresholds.
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