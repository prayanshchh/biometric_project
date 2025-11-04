# /home/prayansh-chhablani/biometrics/codes/project/multi_ive.py

import numpy as np
from typing import List, Dict, Any
from sklearn.decomposition import PCA
# **CHANGE 1**: Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluation import calculate_similarity_scores

def perform_multi_ive(
    unprotected_templates: np.ndarray,
    identity_labels: np.ndarray,
    soft_biometric_labels: np.ndarray,
    eliminations_per_step: int = 20,
    distribution_checkpoints: List[int] = [0, 10, 20] # Checkpoints for saving score distributions
) -> Dict[str, Any]:
    """
    Implements Multi-IVE, tracks performance, and collects detailed data for reporting.
    This version uses a single Decision Tree as per the user's request.
    """
    print("\n🔬 Starting Upgraded Multi-IVE process (with Decision Tree)...")
    
    # --- Step 1: PCA Transformation ---
    print("   Step 1: Applying PCA to the templates.")
    n_samples, n_features = unprotected_templates.shape
    n_components = min(n_samples, n_features)
    print(f"   - Data shape: {unprotected_templates.shape}. Max possible PCA components: {n_components}.")
    
    pca = PCA(n_components=n_components)
    pca_templates = pca.fit_transform(unprotected_templates)

    # --- Step 2: Initial Analysis (for reporting) ---
    # **CHANGE 2**: Use DecisionTreeClassifier for initial analysis as well
    initial_classifier = DecisionTreeClassifier(random_state=42)
    initial_classifier.fit(pca_templates, soft_biometric_labels)
    initial_importances = initial_classifier.feature_importances_
    
    # --- Step 3: Initialization for Iterative Loop ---
    eliminated_indices = set()
    num_total_features = pca_templates.shape[1]
    
    # Data stores for comprehensive results
    results = {
        "elimination_counts": [],
        "sex_accuracies": [],
        "recognition_scores": [],
        "importance_history": [],
        "stored_distributions": {},
        "pca_object": pca,
        "initial_templates_pca": pca_templates,
        "initial_importances": initial_importances,
        "late_stage_templates_pca": None # Initialize key
    }
    
    # --- Step 4: Iterative Elimination Loop ---
    while len(eliminated_indices) <= num_total_features:
        current_eliminated_count = len(eliminated_indices)
        if current_eliminated_count > num_total_features:
            break

        print(f"\n   Iteration: {len(results['elimination_counts']) + 1} ({current_eliminated_count}/{num_total_features} components eliminated)")
        results['elimination_counts'].append(current_eliminated_count)

        # Create a sanitized copy of templates for this iteration
        sanitized_templates_pca = pca_templates.copy()
        if eliminated_indices:
            sanitized_templates_pca[:, list(eliminated_indices)] = 0
            
        # Reconstruct templates to original space for similarity calculation
        reconstructed_templates = pca.inverse_transform(sanitized_templates_pca)
        
        # --- Robust Evaluation ---
        genuine_scores, imposter_scores = calculate_similarity_scores(reconstructed_templates, identity_labels)
        mean_genuine_score = np.mean(genuine_scores) if genuine_scores else 0.0
        results['recognition_scores'].append(mean_genuine_score)
        print(f"   - Utility (Mean Genuine Similarity): {mean_genuine_score:.4f}")

        # Store full distributions at specified checkpoints for later plotting
        if current_eliminated_count in distribution_checkpoints:
            print(f"   - Storing score distributions for checkpoint: {current_eliminated_count} eliminations.")
            results['stored_distributions'][current_eliminated_count] = (genuine_scores, imposter_scores)
            
            if current_eliminated_count == max(distribution_checkpoints):
                 results['late_stage_templates_pca'] = sanitized_templates_pca
        
        # --- Privacy Evaluation ---
        X_train, X_test, y_train, y_test = train_test_split(
            sanitized_templates_pca, soft_biometric_labels, test_size=0.3, stratify=soft_biometric_labels, random_state=42
        )
        
        # **CHANGE 3**: The classifier is now a single DecisionTreeClassifier
        classifier = DecisionTreeClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, classifier.predict(X_test))
        results['sex_accuracies'].append(accuracy * 100)
        print(f"   - Privacy (Sex Accuracy): {accuracy*100:.2f}%")

        if current_eliminated_count == num_total_features:
            results['final_sanitized_template_pca'] = sanitized_templates_pca[0]
            break

        # --- Identify and Eliminate Features ---
        importances = classifier.feature_importances_
        results['importance_history'].append(importances)
        importances[list(eliminated_indices)] = -1
        indices_to_eliminate = np.argsort(importances)[-eliminations_per_step:]
        eliminated_indices.update(set(indices_to_eliminate) - eliminated_indices)

    # Ensure we have a distribution for the final state
    if num_total_features not in results['stored_distributions']:
         results['stored_distributions'][num_total_features] = (genuine_scores, imposter_scores)

    # Store the final, fully-enhanced templates (reconstructed to original dimension)
    final_sanitized_pca = pca_templates.copy()
    final_sanitized_pca[:, list(eliminated_indices)] = 0
    results['final_enhanced_templates'] = pca.inverse_transform(final_sanitized_pca)

    print("✅ Multi-IVE process complete. All data collected.")
    return results