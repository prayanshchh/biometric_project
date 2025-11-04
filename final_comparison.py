# /home/prayansh-chhablani/biometrics/codes/project/final_comparison.py

import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

import data_loader
import feature_extractor
import multi_ive_analyzer
import final_report_generator

# --- Configuration ---
PROJECT_PATH = "/home/prayansh-chhablani/biometrics/codes/project/"
DATASET_PATH = "/home/prayansh-chhablani/Biometric_face_dataset/Biometric_face_dataset/"
FINAL_REPORT_PATH = os.path.join(PROJECT_PATH, "final_visual_report_suite")
NUM_IMAGES = 30
DISTRIBUTION_CHECKPOINTS = [0, 15, 28]

def main():
    """Main script to run a full comparison and generate a comprehensive 10-plot report."""
    print("🚀 Starting Final Comprehensive Comparison 🚀")
    print("-" * 70)

    # --- Step 1: Data Loading & Feature Extraction ---
    metadata_df = data_loader.load_and_label_data(DATASET_PATH, num_images=NUM_IMAGES)
    unprotected_templates = feature_extractor.extract_features(metadata_df)
    
    # --- Step 2: Define Models for Both Approaches ---
    n_components = NUM_IMAGES - 1
    
    # Approach 1: PCA + Decision Tree
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_components))])
    dt_model = DecisionTreeClassifier(random_state=42)
    
    # Approach 2: KernelPCA + Random Forest
    kpca_pipeline = Pipeline([('scaler', StandardScaler()), ('kpca', KernelPCA(n_components=n_components, kernel='rbf', fit_inverse_transform=True, random_state=42))])
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # --- Step 3: Run Analysis for Both Approaches ---
    results_pca_dt = multi_ive_analyzer.perform_analysis(
        unprotected_templates, metadata_df['identity_id'].to_numpy(), metadata_df['sex'].to_numpy(),
        pca_pipeline, dt_model, DISTRIBUTION_CHECKPOINTS
    )
    
    results_kpca_rf = multi_ive_analyzer.perform_analysis(
        unprotected_templates, metadata_df['identity_id'].to_numpy(), metadata_df['sex'].to_numpy(),
        kpca_pipeline, rf_model, DISTRIBUTION_CHECKPOINTS
    )

    print("-" * 70)
    
    # --- Step 4: Generate the Final Comparison Report ---
    final_report_generator.generate_full_report(
        results={'pca_dt': results_pca_dt, 'kpca_rf': results_kpca_rf},
        labels={'id': metadata_df['identity_id'], 'sex': metadata_df['sex']},
        output_path=FINAL_REPORT_PATH,
        checkpoints=DISTRIBUTION_CHECKPOINTS
    )
    
    print("-" * 70)
    print(f"🎉 Analysis finished! Check the new 10-plot report in '{FINAL_REPORT_PATH}' 🎉")

if __name__ == "__main__":
    main()