# /home/prayansh-chhablani/biometrics/codes/project/pca_vs_kernelpca_analyzer.py

import os
from sklearn.decomposition import PCA, KernelPCA

# Import our project modules
import data_loader
import feature_extractor
import multi_ive_analyzer
import comparison_visualizer_kernel

# --- Configuration ---
PROJECT_PATH = "/home/prayansh-chhablani/biometrics/codes/project/"
DATASET_PATH = "/home/prayansh-chhablani/Biometric_face_dataset/Biometric_face_dataset/"
COMPARISON_REPORTS_PATH = os.path.join(PROJECT_PATH, "kernel_comparison_report")
NUM_IMAGES = 20

def main():
    """Main function to compare PCA vs. KernelPCA for the Multi-IVE task."""
    print("🚀 Starting Comparison: PCA vs. KernelPCA 🚀")
    print("-" * 60)

    # --- Step 1: Data Loading & Feature Extraction ---
    metadata_df = data_loader.load_and_label_data(DATASET_PATH, num_images=NUM_IMAGES)
    identity_labels = metadata_df['identity_id'].to_numpy()
    sex_labels = metadata_df['sex'].to_numpy()
    unprotected_templates = feature_extractor.extract_features(metadata_df)
    
    # --- Step 2: Define the Models to Compare ---
    # The number of components cannot exceed the number of samples for KernelPCA
    n_components = NUM_IMAGES - 1 

    pca_model = PCA(n_components=n_components)
    
    kernel_pca_model = KernelPCA(
        n_components=n_components, 
        kernel='rbf', # A powerful non-linear kernel
        gamma=None,   # Auto-determine gamma
        fit_inverse_transform=True # Essential for reconstructing templates
    )

    # --- Step 3: Run Analysis for BOTH models ---
    pca_results = multi_ive_analyzer.perform_analysis(
        unprotected_templates, identity_labels, sex_labels, pca_model
    )
    
    kernel_pca_results = multi_ive_analyzer.perform_analysis(
        unprotected_templates, identity_labels, sex_labels, kernel_pca_model
    )

    print("-" * 60)
    
    # --- Step 4: Generate Comparison Report ---
    print("🎨 Generating comparison report...")
    
    comparison_data = {
        'pca': pca_results,
        'kernel_pca': kernel_pca_results
    }
    
    comparison_visualizer_kernel.plot_tradeoff_comparison(comparison_data, COMPARISON_REPORTS_PATH)
    
    print("-" * 60)
    print(f"🎉 Analysis finished! Check the new comparison plot in '{COMPARISON_REPORTS_PATH}' 🎉")

if __name__ == "__main__":
    main()