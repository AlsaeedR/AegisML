from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
from sklearn.pipeline import Pipeline


def _get_classifier_step(
    model: Any,
    vectorizer_override: Optional[Any] = None,
) -> Tuple[Optional[Any], Any]:
    """
    Extracts the preprocessing pipeline and estimator components
    from a model object.

    Supports unified scikit-learn Pipelines with multiple
    preprocessing steps, as well as separately supplied vectorizers.

    Example:
        TF-IDF -> SVD -> SVC

    becomes:
        preprocessor = TF-IDF -> SVD
        classifier = SVC
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
    """
    Creates a copy of the fitted classifier for ART.

    ART expects numeric class indices internally. Some scikit-learn
    classifiers, especially text classifiers, may use string labels
    such as 'Film' or 'Technology'. The copied classifier is therefore
    given numeric class labels without modifying the original model.
    """

    art_model = deepcopy(classifier)

    if hasattr(art_model, "classes_"):
        classes = np.asarray(art_model.classes_)

        if not np.issubdtype(classes.dtype, np.integer):
            art_model.classes_ = np.arange(
                len(classes),
                dtype=int,
            )

    return art_model


def run_adversarial_test(
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes empirical adversarial evasion tests using the
    HopSkipJump decision attack.

    Evaluates whether the model flips predictions within a
    realistic L2 perturbation budget.
    """

    model = state["model"]
    vectorizer_override = state.get("vectorizer")

    X_text = state["X_text"]
    y_true = state["y_true"]

    preprocessor, classifier = _get_classifier_step(
        model,
        vectorizer_override,
    )

    if preprocessor is None:
        return {
            "adversarial_evidence": {
                "vulnerability_id": "V4",
                "vulnerability_name": "Adversarial Robustness",
                "status": "inconclusive",
                "severity": "low",
                "evidence": {
                    "status": "skipped",
                    "reason": (
                        "Could not extract or resolve "
                        "a preprocessing step to produce "
                        "numeric feature vectors."
                    ),
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
                    "reason": (
                        "adversarial-robustness-toolbox "
                        "(ART) is not installed in the environment."
                    ),
                },
            }
        }

    try:
        sample_size = min(
            50,
            len(X_text),
        )

        sample_texts = X_text[:sample_size]

        # Apply every preprocessing step before the final classifier.
        #
        # Example:
        # TF-IDF -> SVD -> SVC
        #
        # This prevents feature-dimension mismatches when intermediate
        # preprocessing steps are present.
        X_vec = preprocessor.transform(
            sample_texts
        )

        X_arr = (
            np.array(X_vec.todense())
            if hasattr(X_vec, "todense")
            else np.asarray(X_vec)
        )

        X_arr = np.asarray(
            X_arr,
            dtype=np.float32,
        )

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(
                1,
                -1,
            )

        n_classes = (
            len(classifier.classes_)
            if hasattr(classifier, "classes_")
            else len(set(y_true))
        )

        clip_min = float(
            np.min(X_arr)
        )

        clip_max = (
            float(np.max(X_arr))
            + 1e-6
        )

        if clip_max <= clip_min:
            clip_max = (
                clip_min
                + 1e-6
            )

        # ART internally expects numeric class indices.
        # Use a copied classifier so the original model remains unchanged.
        art_model = _prepare_art_classifier(
            classifier
        )

        art_classifier = SklearnClassifier(
            model=art_model,
            clip_values=(
                clip_min,
                clip_max,
            ),
        )

        # Predictions from the original classifier are kept for the
        # final before/after comparison.
        original_preds = np.asarray(
            classifier.predict(
                X_arr
            )
        )

        attack = HopSkipJump(
            classifier=art_classifier,
            targeted=False,
            max_iter=20,
            max_eval=200,
            init_eval=20,
        )

        X_adv = attack.generate(
            x=X_arr
        )

        # Evaluate adversarial examples using the original classifier
        # so original string labels remain intact.
        adv_preds = np.asarray(
            classifier.predict(
                X_adv
            )
        )

        flipped_mask = (
            original_preds
            != adv_preds
        )

        flipped = int(
            np.sum(
                flipped_mask
            )
        )

        success_rate = round(
            flipped / sample_size,
            4,
        )

        max_relative_perturbation = 0.5

        original_norms = np.linalg.norm(
            X_arr,
            axis=1,
        )

        perturbation_norms = np.linalg.norm(
            X_adv - X_arr,
            axis=1,
        )

        zero_norm_epsilon = 1e-6

        valid_mask = (
            original_norms
            > zero_norm_epsilon
        )

        n_zero_norm_samples = int(
            np.sum(
                ~valid_mask
            )
        )

        relative_perturbations = np.full_like(
            perturbation_norms,
            np.inf,
            dtype=float,
        )

        relative_perturbations[
            valid_mask
        ] = (
            perturbation_norms[
                valid_mask
            ]
            / original_norms[
                valid_mask
            ]
        )

        within_budget_mask = (
            flipped_mask
            & (
                relative_perturbations
                <= max_relative_perturbation
            )
        )

        flipped_within_budget = int(
            np.sum(
                within_budget_mask
            )
        )

        success_rate_within_budget = round(
            flipped_within_budget
            / sample_size,
            4,
        )

        avg_perturbation = float(
            np.mean(
                perturbation_norms
            )
        )

        avg_relative_perturbation = (
            float(
                np.mean(
                    relative_perturbations[
                        valid_mask
                    ]
                )
            )
            if np.any(valid_mask)
            else None
        )

        status = (
            "vulnerable"
            if success_rate_within_budget >= 0.3
            else "not_vulnerable"
        )

        severity = (
            "high"
            if success_rate_within_budget >= 0.6
            else (
                "medium"
                if success_rate_within_budget >= 0.3
                else "low"
            )
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
                    "max_relative_perturbation_budget": (
                        max_relative_perturbation
                    ),
                    "flipped_predictions_within_budget": (
                        flipped_within_budget
                    ),
                    "attack_success_rate_within_budget": (
                        success_rate_within_budget
                    ),
                    "avg_perturbation_norm": round(
                        avg_perturbation,
                        4,
                    ),
                    "avg_relative_perturbation": (
                        round(
                            avg_relative_perturbation,
                            4,
                        )
                        if avg_relative_perturbation is not None
                        else None
                    ),
                    "n_near_zero_vector_samples_excluded": (
                        n_zero_norm_samples
                    ),
                    "interpretation": (
                        "A significant portion of evaluation samples "
                        "were flipped within a realistic perturbation "
                        "budget, confirming model susceptibility to "
                        "adversarial evasion."
                        if success_rate_within_budget >= 0.3
                        else
                        "Attacks required unrealistically large "
                        "perturbations; the model proved empirically "
                        "robust within normal perturbation boundaries."
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
                "evidence": {
                    "status": "error",
                    "reason": str(e),
                },
            }
        }