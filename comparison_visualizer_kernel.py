# /home/prayansh-chhablani/biometrics/codes/project/comparison_visualizer_kernel.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def _save_plot(fig, output_path: str, filename: str):
    """Helper function to save plots."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    full_path = os.path.join(output_path, filename)
    fig.tight_layout(pad=2.0)
    fig.savefig(full_path, dpi=150)
    plt.close(fig)
    print(f"   💾 Comparison plot saved to: {full_path}")

def plot_tradeoff_comparison(results: Dict, output_path: str):
    """Generates a plot comparing the privacy-utility trade-offs of PCA and KernelPCA."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
    
    # --- Plot 1: Privacy (Sex Estimation Accuracy) ---
    ax1.plot(results['pca']['elimination_counts'], results['pca']['sex_accuracies'], 'r-o', label='PCA-based')
    ax1.plot(results['kernel_pca']['elimination_counts'], results['kernel_pca']['sex_accuracies'], 'r--x', label='KernelPCA-based')
    ax1.set_title('Privacy Enhancement Performance', fontsize=16)
    ax1.set_xlabel('Number of Eliminated Components')
    ax1.set_ylabel('Sex Estimation Accuracy (%)')
    ax1.legend()
    
    # --- Plot 2: Utility (Recognition Performance) ---
    ax2.plot(results['pca']['elimination_counts'], [s * 100 for s in results['pca']['recognition_scores']], 'b-o', label='PCA-based')
    ax2.plot(results['kernel_pca']['elimination_counts'], [s * 100 for s in results['kernel_pca']['recognition_scores']], 'b--x', label='KernelPCA-based')
    ax2.set_title('Recognition Utility', fontsize=16)
    ax2.set_xlabel('Number of Eliminated Components')
    ax2.set_ylabel('Mean Genuine Similarity (%)')
    ax2.legend()
    
    fig.suptitle('Comparison of Dimensionality Reduction Methods for Privacy Enhancement', fontsize=20)
    _save_plot(fig, output_path, "comparison_pca_vs_kernelpca.png")