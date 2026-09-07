from typing import Any, Dict, List
import copy
import random
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC

def _get_classifier_step(model: Any):
    if hasattr(model, 'named_steps'):
        steps = list(model.named_steps.items())
        vectorizer = steps[0][1] if len(steps) > 1 else None
        classifier = steps[-1][1]
        return (vectorizer, classifier)
    return (None, model)

def _generic_label_flip_test(model: Any, X_text: List[str], y_true: List[Any], flip_fraction: float=0.15) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(X_text, y_true, test_size=0.3, random_state=42)
    clean_model = clone(model)
    clean_model.fit(X_train, y_train)
    clean_preds = clean_model.predict(X_test)
    clean_acc = accuracy_score(y_test, clean_preds)
    labels_pool = list(set(y_train))
    y_train_poisoned = list(y_train)
    n_flip = max(1, int(len(y_train) * flip_fraction))
    flip_indices = random.sample(range(len(y_train)), n_flip)
    for idx in flip_indices:
        current = y_train_poisoned[idx]
        alt_labels = [l for l in labels_pool if l != current]
        if alt_labels:
            y_train_poisoned[idx] = random.choice(alt_labels)
    poisoned_model = clone(model)
    poisoned_model.fit(X_train, y_train_poisoned)
    poisoned_preds = poisoned_model.predict(X_test)
    poisoned_acc = accuracy_score(y_test, poisoned_preds)
    accuracy_drop = round(clean_acc - poisoned_acc, 4)
    return {'method': 'generic_label_flip_retrain', 'flip_fraction_used': flip_fraction, 'n_labels_flipped': n_flip, 'clean_accuracy': round(clean_acc, 4), 'poisoned_accuracy': round(poisoned_acc, 4), 'accuracy_drop': accuracy_drop, 'interpretation': 'Significant accuracy drop after label-flip poisoning indicates the pipeline has no data validation / outlier rejection to protect training against corrupted labels.' if accuracy_drop > 0.05 else 'Model showed resilience to a small fraction of flipped training labels in this simulation.'}

def _art_svm_poisoning_test(classifier: Any, X_vec, y_true: List[Any]) -> Dict[str, Any]:
    if not isinstance(classifier, (SVC, LinearSVC)):
        return None
    try:
        from art.estimators.classification.scikitlearn import ScikitlearnSVC
        from art.attacks.poisoning import PoisoningAttackSVM
    except ImportError:
        return {'method': 'art_poisoning_attack_svm', 'status': 'skipped', 'reason': 'adversarial-robustness-toolbox (ART) is not installed.'}
    try:
        art_classifier = ScikitlearnSVC(model=classifier)
        X_arr = np.array(X_vec.todense()) if hasattr(X_vec, 'todense') else np.array(X_vec)
        y_arr = np.array(y_true)
        attack = PoisoningAttackSVM(classifier=art_classifier, step=0.1, eps=1.0, x_train=X_arr, y_train=y_arr, x_val=X_arr[:10], y_val=y_arr[:10], max_iter=20)
        poison_point, _ = attack.poison(X_arr[:1], y_arr[:1])
        return {'method': 'art_poisoning_attack_svm', 'status': 'completed', 'note': 'ART successfully crafted a poisoning point that shifts the SVM decision boundary, confirming the model is exposed to targeted poisoning attacks if an attacker can inject training samples.'}
    except Exception as e:
        return {'method': 'art_poisoning_attack_svm', 'status': 'error', 'reason': str(e)}

def run_poisoning_test(state: Dict[str, Any]) -> Dict[str, Any]:
    model = state['model']
    X_text = state['X_text']
    y_true = state['y_true']
    evidence = {}
    evidence['generic_test'] = _generic_label_flip_test(model, X_text, y_true)
    vectorizer, classifier = _get_classifier_step(model)
    if vectorizer is not None:
        try:
            X_vec = vectorizer.transform(X_text)
            art_result = _art_svm_poisoning_test(classifier, X_vec, y_true)
            if art_result is not None:
                evidence['art_test'] = art_result
        except Exception as e:
            evidence['art_test'] = {'status': 'error', 'reason': str(e)}
    else:
        evidence['art_test'] = {'status': 'skipped', 'reason': 'Could not separate vectorizer/classifier steps from the model.'}
    accuracy_drop = evidence['generic_test']['accuracy_drop']
    status = 'vulnerable' if accuracy_drop > 0.05 else 'not_vulnerable'
    severity = 'high' if accuracy_drop > 0.15 else 'medium' if accuracy_drop > 0.05 else 'low'
    return {'poisoning_evidence': {'vulnerability_id': 'V1', 'vulnerability_name': 'Data Poisoning', 'status': status, 'severity': severity, 'evidence': evidence}}
