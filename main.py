# /home/prayansh-chhablani/biometrics/codes/project/main.py

import os
import numpy as np
import pandas as pd

# Import functions from our custom modules
import data_loader
import feature_extractor
import multi_ive
import cancelable_biometrics
import reporting # Our new, powerful visualization module
import data_exporter # Import the new module

# --- Configuration ---
PROJECT_PATH = "/home/prayansh-chhablani/biometrics/codes/project/"
DATASET_PATH = "/home/prayansh-chhablani/Biometric_face_dataset/Biometric_face_dataset/"
# Create a dedicated folder for the visual report
REPORTS_PATH = os.path.join(PROJECT_PATH, "visual_report")
DATA_EXPORT_PATH = os.path.join(PROJECT_PATH, "exported_data") # Path for exported data

# Multi-IVE parameters
# With only 30 components, eliminate fewer at each step to get a smoother curve
ELIMINATIONS_PER_STEP = 2 
# Checkpoints to save score distributions for plotting (start, middle, end)
DISTRIBUTION_CHECKPOINTS = [0, 14, 28]

def main():
    """Main function to execute the entire upgraded pipeline."""
    
    print("🚀 Starting Comprehensive Analysis of Cancelable Biometrics 🚀")
    print("-" * 60)

    # --- Step 1: Data Loading and Labeling ---
    try:
        metadata_df = data_loader.load_and_label_data(DATASET_PATH, num_images=30)
        identity_labels = metadata_df['identity_id']
        sex_labels = metadata_df['sex']
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Critical Error during data loading: {e}")
        return

    print("-" * 60)

    # --- Step 2: Face Feature Extraction ---
    unprotected_templates = feature_extractor.extract_features(metadata_df, model_name='ArcFace')
    
    print("-" * 60)

    # --- Step 3: Multi-IVE for Privacy Enhancement & Data Collection ---
    # This function now returns a dictionary with all the data we need
    results = multi_ive.perform_multi_ive(
        unprotected_templates=unprotected_templates,
        identity_labels=identity_labels.to_numpy(),
        soft_biometric_labels=sex_labels.to_numpy(),
        eliminations_per_step=ELIMINATIONS_PER_STEP,
        distribution_checkpoints=DISTRIBUTION_CHECKPOINTS
    )

    print("-" * 60)
    
    # --- Step 4: Generating Visual Report ---
    print("🎨 Generating a comprehensive suite of 10 visualizations...")
    
    # Plot 1: PCA Explained Variance
    reporting.plot_pca_variance(results['pca_object'], REPORTS_PATH)
    
    # Plot 2: Initial Feature Importance for 'Sex'
    reporting.plot_initial_importance(results['initial_importances'], REPORTS_PATH)
    
    # Plot 3: Template Transformation Heatmap
    reporting.plot_template_heatmap(
        template_before=results['initial_templates_pca'][0],
        template_after=results['final_sanitized_template_pca'],
        output_path=REPORTS_PATH
    )

    # Plot 4: t-SNE of Identity Clusters (Utility)
    reporting.plot_tsne_clusters(
        templates_pca=results['initial_templates_pca'], 
        labels=identity_labels, 
        title='t-SNE of Identity Clusters (Before Elimination)', 
        output_path=REPORTS_PATH, 
        filename='04_tsne_identity_clusters.png'
    )

    # Plot 5 & 6: t-SNE of 'Sex' Clusters (Privacy)
    reporting.plot_tsne_clusters(
        templates_pca=results['initial_templates_pca'], 
        labels=sex_labels, 
        title="t-SNE of 'Sex' Clusters (Before Elimination)", 
        output_path=REPORTS_PATH, 
        filename='05_tsne_sex_clusters_before.png'
    )
    
    # **FIX**: Use the stable late-stage templates instead of the crashing calculation.
    reporting.plot_tsne_clusters(
        templates_pca=results['late_stage_templates_pca'],
        labels=sex_labels, 
        title=f"t-SNE of 'Sex' Clusters (After {max(DISTRIBUTION_CHECKPOINTS)} Eliminations)",
        output_path=REPORTS_PATH,
        filename='06_tsne_sex_clusters_after.png'
    )

    # Plots 7, 8, 9: Score Distributions at key checkpoints
    reporting.plot_all_score_distributions(results['stored_distributions'], REPORTS_PATH)
    
    # Plot 10: The final, enhanced trade-off plot
    reporting.plot_final_tradeoff(results, REPORTS_PATH)
    
    print("✅ Visual report generation complete.")
    
    print("-" * 60)

    # --- Step 5: Exporting Processed Data ---
    data_exporter.save_templates(
        output_path=DATA_EXPORT_PATH,
        filename='processed_templates.npz',
        original_templates=unprotected_templates,
        enhanced_templates=results['final_enhanced_templates']
    )

    print("-" * 60)
    print(f"🎉 Pipeline finished successfully! Check reports in '{REPORTS_PATH}' and data in '{DATA_EXPORT_PATH}' 🎉")


if __name__ == "__main__":
    main()