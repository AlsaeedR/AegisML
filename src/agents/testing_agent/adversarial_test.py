from typing import Any, Dict, List
import numpy as np

def _get_classifier_step(model: Any):
    if hasattr(model, 'named_steps'):
        steps = list(model.named_steps.items())
        vectorizer = steps[0][1] if len(steps) > 1 else None
        classifier = steps[-1][1]
        return (vectorizer, classifier)
    return (None, model)

def run_adversarial_test(state: Dict[str, Any]) -> Dict[str, Any]:
    model = state['model']
    X_text = state['X_text']
    y_true = state['y_true']
    vectorizer, classifier = _get_classifier_step(model)
    if vectorizer is None:
        return {'adversarial_evidence': {'vulnerability_id': 'V4', 'vulnerability_name': 'Adversarial Robustness', 'status': 'inconclusive', 'severity': 'low', 'evidence': {'status': 'skipped', 'reason': 'Could not separate a vectorizer from the trained model, so numeric feature vectors could not be produced for the attack.'}}}
    try:
        from art.estimators.classification import SklearnClassifier
        from art.attacks.evasion import HopSkipJump
    except ImportError:
        return {'adversarial_evidence': {'vulnerability_id': 'V4', 'vulnerability_name': 'Adversarial Robustness', 'status': 'inconclusive', 'severity': 'low', 'evidence': {'status': 'skipped', 'reason': 'adversarial-robustness-toolbox (ART) is not installed.'}}}
    try:
        
        sample_size = min(50, len(X_text))
        sample_texts = X_text[:sample_size]
        X_vec = vectorizer.transform(sample_texts)
        X_arr = np.array(X_vec.todense()) if hasattr(X_vec, 'todense') else np.array(X_vec)
        n_classes = len(set(y_true))

        art_classifier = SklearnClassifier(model=classifier, clip_values=(0.0, X_arr.max() + 1e-06))
        original_preds = classifier.predict(X_arr)
        attack = HopSkipJump(classifier=art_classifier, targeted=False, max_iter=20, max_eval=200, init_eval=20)
        X_adv = attack.generate(x=X_arr)
        adv_preds = classifier.predict(X_adv)

        flipped_mask = original_preds != adv_preds
        flipped = int(np.sum(flipped_mask))
        success_rate = round(flipped / sample_size, 4)

     
        max_relative_perturbation = 0.5
        original_norms = np.linalg.norm(X_arr, axis=1)
        perturbation_norms = np.linalg.norm(X_adv - X_arr, axis=1)

       
        zero_norm_epsilon = 1e-6
        valid_mask = original_norms > zero_norm_epsilon
        n_zero_norm_samples = int(np.sum(~valid_mask))

        relative_perturbations = np.full_like(perturbation_norms, np.inf)
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

       
        status = 'vulnerable' if success_rate_within_budget >= 0.3 else 'not_vulnerable'
        severity = (
            'high' if success_rate_within_budget >= 0.6
            else 'medium' if success_rate_within_budget >= 0.3
            else 'low'
        )

        return {
            'adversarial_evidence': {
                'vulnerability_id': 'V4',
                'vulnerability_name': 'Adversarial Robustness',
                'status': status,
                'severity': severity,
                'evidence': {
                    'method': 'art_hopskipjump_evasion',
                    'n_samples_tested': sample_size,
                    'n_classes': n_classes,
                    'flipped_predictions_any_perturbation': flipped,
                    'success_rate_any_perturbation': success_rate,
                    'max_relative_perturbation_budget': max_relative_perturbation,
                    'flipped_predictions_within_budget': flipped_within_budget,
                    'success_rate_within_budget': success_rate_within_budget,
                    'avg_perturbation_norm': round(avg_perturbation, 4),
                    'avg_relative_perturbation': (
                        round(avg_relative_perturbation, 4)
                        if avg_relative_perturbation is not None
                        else None
                    ),
                    'n_near_zero_vector_samples_excluded': n_zero_norm_samples,
                    'interpretation': (
                        'A significant fraction of samples were misclassified using '
                        'perturbations within a realistic size budget, indicating the '
                        'model lacks adversarial robustness / hardening.'
                        if success_rate_within_budget >= 0.3
                        else 'Most successful attacks required unrealistically large '
                        'perturbations; within a realistic budget the model was '
                        'largely robust in this sample.'
                    ),
                },
            }
        }
    except Exception as e:
        return {'adversarial_evidence': {'vulnerability_id': 'V4', 'vulnerability_name': 'Adversarial Robustness', 'status': 'inconclusive', 'severity': 'low', 'evidence': {'status': 'error', 'reason': str(e)}}}
