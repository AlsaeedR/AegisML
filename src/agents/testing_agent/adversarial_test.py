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
        sample_size = min(10, len(X_text))
        sample_texts = X_text[:sample_size]
        X_vec = vectorizer.transform(sample_texts)
        X_arr = np.array(X_vec.todense()) if hasattr(X_vec, 'todense') else np.array(X_vec)
        n_classes = len(set(y_true))
        art_classifier = SklearnClassifier(model=classifier, clip_values=(0.0, X_arr.max() + 1e-06))
        original_preds = classifier.predict(X_arr)
        attack = HopSkipJump(classifier=art_classifier, targeted=False, max_iter=20, max_eval=200, init_eval=20)
        X_adv = attack.generate(x=X_arr)
        adv_preds = classifier.predict(X_adv)
        flipped = int(np.sum(original_preds != adv_preds))
        success_rate = round(flipped / sample_size, 4)
        avg_perturbation = float(np.mean(np.linalg.norm(X_adv - X_arr, axis=1)))
        status = 'vulnerable' if success_rate >= 0.3 else 'not_vulnerable'
        severity = 'high' if success_rate >= 0.6 else 'medium' if success_rate >= 0.3 else 'low'
        return {'adversarial_evidence': {'vulnerability_id': 'V4', 'vulnerability_name': 'Adversarial Robustness', 'status': status, 'severity': severity, 'evidence': {'method': 'art_hopskipjump_evasion', 'n_samples_tested': sample_size, 'n_classes': n_classes, 'flipped_predictions': flipped, 'success_rate': success_rate, 'avg_perturbation_norm': round(avg_perturbation, 4), 'interpretation': 'A high fraction of samples were misclassified after small crafted perturbations, indicating the model lacks adversarial robustness / hardening.' if success_rate >= 0.3 else 'The model resisted most crafted perturbations in this black-box attack sample.'}}}
    except Exception as e:
        return {'adversarial_evidence': {'vulnerability_id': 'V4', 'vulnerability_name': 'Adversarial Robustness', 'status': 'inconclusive', 'severity': 'low', 'evidence': {'status': 'error', 'reason': str(e)}}}
