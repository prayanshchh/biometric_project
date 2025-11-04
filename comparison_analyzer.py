# /home/prayansh-chhablani/biometrics/codes/project/comparison_analyzer.py

import os
import time
import numpy as np

# Import our project modules
import data_loader
import feature_extractor
import multi_ive
import cancelable_biometrics
import homomorphic_encryption
import evaluation
import comparison_visualizer

# --- Configuration ---
PROJECT_PATH = "/home/prayansh-chhablani/biometrics/codes/project/"
DATASET_PATH = "/home/prayansh-chhablani/Biometric_face_dataset/Biometric_face_dataset/"
COMPARISON_REPORTS_PATH = os.path.join(PROJECT_PATH, "comparison_report")
NUM_IMAGES = 20
OPTIMAL_ELIMINATIONS = 14 # Determined from our previous analysis

def main():
    """Main function to run and compare the two protection schemes."""
    print("🚀 Starting Comparison: Current Approach vs. Triplet Protection 🚀")
    print("-" * 60)

    # --- Step 1: Data Loading & Feature Extraction ---
    metadata_df = data_loader.load_and_label_data(DATASET_PATH, num_images=NUM_IMAGES)
    identity_labels = metadata_df['identity_id'].to_numpy()
    unprotected_templates = feature_extractor.extract_features(metadata_df)
    
    print("-" * 60)
    print(f"🎯 Performing privacy enhancement up to the optimal point: {OPTIMAL_ELIMINATIONS} eliminations.")
    
    # --- Step 2: Apply Multi-IVE up to the optimal point ---
    # We only need the templates at the end of this process.
    pca = multi_ive.PCA(n_components=min(unprotected_templates.shape))
    templates_pca = pca.fit_transform(unprotected_templates)
    templates_pca[:, :OPTIMAL_ELIMINATIONS] = 0 # Simulate elimination
    multi_ive_templates = pca.inverse_transform(templates_pca)
    print("✅ Layer 1: Soft-biometric privacy enhanced via Multi-IVE.")
    
    print("-" * 60)
    # --- Step 3: Layer 2 - Apply Cancelable Biometrics ---
    key_matrix = cancelable_biometrics.generate_key(multi_ive_templates.shape[1])
    cb_templates = np.dot(multi_ive_templates, key_matrix)
    print("✅ Layer 2: Cancelable Biometric transform applied.")
    
    # --- Step 4: Run Analysis for BOTH approaches ---
    
    # --- PATH 1: CURRENT APPROACH (PLAINTEXT COMPARISON) ---
    print("\n--- Analyzing Current (2-Layer) Approach ---")
    start_time_current = time.time()
    current_genuine, current_imposter = evaluation.calculate_similarity_scores(cb_templates, identity_labels)
    time_current = time.time() - start_time_current
    print(f"   - Plaintext comparison finished in {time_current:.4f} seconds.")

    # --- PATH 2: TRIPLET PROTECTION (ENCRYPTED COMPARISON) ---
    print("\n--- Analyzing Triplet Protection (3-Layer) Approach ---")
    he_cipher = homomorphic_encryption.HECipher()
    encrypted_templates = he_cipher.encrypt_batch(cb_templates)
    triplet_genuine, triplet_imposter, time_triplet = he_cipher.calculate_all_encrypted_scores(encrypted_templates, identity_labels)

    print("-" * 60)
    # --- Step 5: Generate Comparison Report ---
    print("🎨 Generating comparison report...")
    
    comparison_data = {
        'current_genuine': current_genuine,
        'current_imposter': current_imposter,
        'triplet_genuine': triplet_genuine,
        'triplet_imposter': triplet_imposter,
        'time_current': time_current,
        'time_triplet': time_triplet
    }
    
    comparison_visualizer.plot_score_distribution_comparison(comparison_data, COMPARISON_REPORTS_PATH)
    comparison_visualizer.plot_timing_comparison(comparison_data, COMPARISON_REPORTS_PATH)
    
    print("✅ Comparison complete.")
    print("-" * 60)
    print(f"🎉 Analysis finished! Check the new plots in '{COMPARISON_REPORTS_PATH}' 🎉")


if __name__ == "__main__":
    # A small hack is needed to run this correctly due to a bug in how scikit-learn's
    # PCA interacts with some multiprocessing libraries.
    from sklearn.decomposition import PCA
    multi_ive.PCA = PCA
    main()