# /home/prayansh-chhablani/biometrics/codes/project/multi_ive_analyzer.py

import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluation import calculate_similarity_scores

def perform_analysis(
    unprotected_templates: np.ndarray,
    identity_labels: np.ndarray,
    sex_labels: np.ndarray,
    dr_model: Any,
    clf_model: Any,
    distribution_checkpoints: List[int],
    eliminations_per_step: int = 1
) -> Dict[str, Any]:
    """
    Performs a flexible Multi-IVE analysis and collects detailed data for a comprehensive report.
    """
    dr_name = dr_model.steps[-1][1].__class__.__name__ if hasattr(dr_model, 'steps') else dr_model.__class__.__name__
    clf_name = clf_model.__class__.__name__
    print(f"\n🔬 Starting Analysis with {dr_name} + {clf_name}...")

    transformed_templates = dr_model.fit_transform(unprotected_templates)
    
    initial_classifier = clf_model.__class__(**clf_model.get_params())
    initial_classifier.fit(transformed_templates, sex_labels)
    
    eliminated_indices = set()
    num_total_features = transformed_templates.shape[1]
    
    results = {
        "elimination_counts": [], "sex_accuracies": [], "recognition_scores": [],
        "initial_importances": initial_classifier.feature_importances_,
        "stored_distributions": {},
        "stored_templates": {0: transformed_templates} # Store initial templates
    }

    while len(eliminated_indices) <= num_total_features:
        current_eliminated_count = len(eliminated_indices)
        if current_eliminated_count > num_total_features: break
        results['elimination_counts'].append(current_eliminated_count)

        sanitized_transformed = transformed_templates.copy()
        if eliminated_indices:
            sanitized_transformed[:, list(eliminated_indices)] = 0
            
        reconstructed = dr_model.inverse_transform(sanitized_transformed)
        
        genuine, imposter = calculate_similarity_scores(reconstructed, identity_labels)
        results['recognition_scores'].append(np.mean(genuine) if genuine else 0.0)
        
        if current_eliminated_count in distribution_checkpoints:
            results['stored_distributions'][current_eliminated_count] = (genuine, imposter)
            results['stored_templates'][current_eliminated_count] = sanitized_transformed

        X_train, X_test, y_train, y_test = train_test_split(sanitized_transformed, sex_labels, test_size=0.3, stratify=sex_labels, random_state=42)
        classifier = clf_model.__class__(**clf_model.get_params())
        classifier.fit(X_train, y_train)
        results['sex_accuracies'].append(accuracy_score(y_test, classifier.predict(X_test)) * 100)

        if current_eliminated_count == num_total_features: break

        importances = classifier.feature_importances_
        importances[list(eliminated_indices)] = -1
        indices_to_eliminate = np.argsort(importances)[-eliminations_per_step:]
        eliminated_indices.update(set(indices_to_eliminate) - eliminated_indices)

    print(f"✅ Analysis with {dr_name} + {clf_name} complete.")
    return results