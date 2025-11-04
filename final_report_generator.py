# /home/prayansh-chhablani/biometrics/codes/project/final_report_generator.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.manifold import TSNE

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def _save_plot(fig, output_path: str, filename: str):
    if not os.path.exists(output_path): os.makedirs(output_path)
    full_path = os.path.join(output_path, filename)
    fig.tight_layout(pad=2.5)
    fig.savefig(full_path, dpi=150)
    plt.close(fig)
    print(f"   💾 Report plot saved: {filename}")

def generate_full_report(results: Dict, labels: Dict, output_path: str, checkpoints: list):
    """Generates a comprehensive 10-plot visual report comparing the two approaches."""
    print("🎨 Generating final comparison report suite...")
    
    # --- Data Setup ---
    pca_dt = results['pca_dt']
    kpca_rf = results['kpca_rf']
    
    # --- Plot 1: Initial Feature Importance ---
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    ax1.bar(range(len(pca_dt['initial_importances'])), pca_dt['initial_importances'], color='r', alpha=0.7)
    ax1.set_title('PCA + Decision Tree', fontsize=16)
    ax1.set_xlabel('Principal Component Index'); ax1.set_ylabel('Feature Importance (Gini)')
    ax2.bar(range(len(kpca_rf['initial_importances'])), kpca_rf['initial_importances'], color='b', alpha=0.7)
    ax2.set_title('KernelPCA + Random Forest', fontsize=16)
    ax2.set_xlabel('Kernel Principal Component Index')
    fig.suptitle("Plot 1: Initial Feature Importance for 'Sex' Prediction", fontsize=22)
    _save_plot(fig, output_path, "01_initial_importances.png")

    # --- Plot 2: t-SNE of Original Data ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(29, len(labels['id'])-1))
    templates_2d = tsne.fit_transform(pca_dt['stored_templates'][0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    sns.scatterplot(x=templates_2d[:, 0], y=templates_2d[:, 1], hue=labels['id'], palette='viridis', ax=ax1, legend='full').set_title('Colored by Identity', fontsize=16)
    sns.scatterplot(x=templates_2d[:, 0], y=templates_2d[:, 1], hue=labels['sex'], palette='coolwarm', ax=ax2, legend='full').set_title('Colored by Sex', fontsize=16)
    fig.suptitle('Plot 2: t-SNE of Unprotected Templates (Baseline)', fontsize=22)
    _save_plot(fig, output_path, "02_tsne_baseline.png")

    # --- Plots 3 & 5: t-SNE at Mid-Elimination ---
    mid_checkpoint = checkpoints[1]
    tsne_pca = TSNE(n_components=2, random_state=42, perplexity=min(29, len(labels['id'])-1))
    tsne_kpca = TSNE(n_components=2, random_state=42, perplexity=min(29, len(labels['id'])-1))
    templates_pca_2d = tsne_pca.fit_transform(pca_dt['stored_templates'][mid_checkpoint])
    templates_kpca_2d = tsne_kpca.fit_transform(kpca_rf['stored_templates'][mid_checkpoint])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), sharey=True)
    sns.scatterplot(x=templates_pca_2d[:, 0], y=templates_pca_2d[:, 1], hue=labels['id'], palette='viridis', ax=ax1, legend=None).set_title('PCA+DT', fontsize=16)
    sns.scatterplot(x=templates_kpca_2d[:, 0], y=templates_kpca_2d[:, 1], hue=labels['id'], palette='viridis', ax=ax2, legend='full').set_title('KernelPCA+RF', fontsize=16)
    fig.suptitle(f'Plot 3: t-SNE of Identity Clusters after {mid_checkpoint} Eliminations (Utility)', fontsize=22)
    _save_plot(fig, output_path, "03_tsne_utility_mid.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), sharey=True)
    sns.scatterplot(x=templates_pca_2d[:, 0], y=templates_pca_2d[:, 1], hue=labels['sex'], palette='coolwarm', ax=ax1, legend=None).set_title('PCA+DT', fontsize=16)
    sns.scatterplot(x=templates_kpca_2d[:, 0], y=templates_kpca_2d[:, 1], hue=labels['sex'], palette='coolwarm', ax=ax2, legend='full').set_title('KernelPCA+RF', fontsize=16)
    fig.suptitle(f'Plot 5: t-SNE of Sex Clusters after {mid_checkpoint} Eliminations (Privacy)', fontsize=22)
    _save_plot(fig, output_path, "05_tsne_privacy_mid.png")
    
    # --- Plots 4 & 6: Main Trade-off Curves ---
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(pca_dt['elimination_counts'], [s * 100 for s in pca_dt['recognition_scores']], 'r-o', label='PCA + Decision Tree')
    ax.plot(kpca_rf['elimination_counts'], [s * 100 for s in kpca_rf['recognition_scores']], 'b--x', label='KernelPCA + RF')
    ax.set_title('Plot 4: Recognition Utility Trade-off', fontsize=18)
    ax.set_xlabel('Eliminated Components'); ax.set_ylabel('Mean Genuine Similarity (%)'); ax.legend()
    _save_plot(fig, output_path, "04_utility_curve.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(pca_dt['elimination_counts'], pca_dt['sex_accuracies'], 'r-o', label='PCA + Decision Tree')
    ax.plot(kpca_rf['elimination_counts'], kpca_rf['sex_accuracies'], 'b--x', label='KernelPCA + RF')
    ax.set_title('Plot 6: Privacy Enhancement Trade-off', fontsize=18)
    ax.set_xlabel('Eliminated Components'); ax.set_ylabel('Sex Estimation Accuracy (%)'); ax.legend()
    _save_plot(fig, output_path, "06_privacy_curve.png")

    # --- Plots 7, 8, 9: Score Distributions ---
    for i, elim_count in enumerate(checkpoints):
        fig, ax = plt.subplots(figsize=(14, 8))
        g_pca, i_pca = pca_dt['stored_distributions'][elim_count]
        g_kpca, i_kpca = kpca_rf['stored_distributions'][elim_count]
        sns.kdeplot(i_pca, color="salmon", label='Imposter (PCA+DT)', ax=ax, fill=True, alpha=0.2)
        sns.kdeplot(g_pca, color="skyblue", label='Genuine (PCA+DT)', ax=ax, fill=True, alpha=0.2)
        sns.kdeplot(i_kpca, color="red", label='Imposter (KernelPCA+RF)', ax=ax, linestyle='--')
        sns.kdeplot(g_kpca, color="blue", label='Genuine (KernelPCA+RF)', ax=ax, linestyle='--')
        ax.legend(); ax.set_xlabel('Similarity Score'); ax.set_ylabel('Density'); ax.set_xlim(-0.1, 1.1)
        ax.set_title(f'Plot {7+i}: Score Distribution Comparison after {elim_count} Eliminations', fontsize=18)
        _save_plot(fig, output_path, f"{7+i:02d}_distribution_{elim_count}_elims.png")
        
    # --- Plot 10: Final Summary Plot ---
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(22, 9), sharey=True)
    ax1.plot(pca_dt['elimination_counts'], pca_dt['sex_accuracies'], 'r-o', label='PCA+DT (Privacy)')
    ax1.plot(kpca_rf['elimination_counts'], kpca_rf['sex_accuracies'], 'b--x', label='KernelPCA+RF (Privacy)')
    ax1.plot(pca_dt['elimination_counts'], [s * 100 for s in pca_dt['recognition_scores']], 'r-s', alpha=0.5, label='PCA+DT (Utility)')
    ax1.plot(kpca_rf['elimination_counts'], [s * 100 for s in kpca_rf['recognition_scores']], 'b--s', alpha=0.5, label='KernelPCA+RF (Utility)')
    ax1.set_title('Combined View', fontsize=18)
    ax1.set_xlabel('Eliminated Components'); ax1.set_ylabel('Performance (%)'); ax1.legend()
    _save_plot(fig, output_path, "10_summary_plot.png")