# /home/prayansh-chhablani/biometrics/codes/project/reporting.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from sklearn.manifold import TSNE

# Set a consistent, professional style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def _save_plot(fig, output_path: str, filename: str):
    """Helper function to save plots consistently."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    full_path = os.path.join(output_path, filename)
    fig.tight_layout(pad=2.0)
    fig.savefig(full_path, dpi=150)
    plt.close(fig)
    print(f"   💾 Plot saved to: {full_path}")

# --- 📊 A. Initial Data Analysis ---

def plot_pca_variance(pca, output_path: str):
    """(1/10) Plots the explained variance of principal components."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    explained_variance = pca.explained_variance_ratio_
    
    ax1.bar(range(len(explained_variance)), explained_variance, alpha=0.7, label='Individual Variance')
    ax1.set_xlabel('Principal Component Index')
    ax1.set_ylabel('Explained Variance Ratio', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    cumulative_variance = np.cumsum(explained_variance)
    ax2.plot(cumulative_variance, color='tab:red', marker='.', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Explained Variance', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1.05)
    
    fig.suptitle('PCA Explained Variance', fontsize=18)
    _save_plot(fig, output_path, "01_pca_variance.png")

def plot_initial_importance(importances, output_path: str):
    """(2/10) Plots the initial feature importance for predicting 'sex'."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(range(len(importances)), importances)
    ax.set_xlabel('Principal Component Index')
    ax.set_ylabel('Feature Importance (Gini)')
    ax.set_title("Initial Importance of Principal Components for 'Sex' Prediction")
    _save_plot(fig, output_path, "02_initial_feature_importance.png")
    
# --- 🧬 B. Visualizing the Templates ---

def plot_template_heatmap(template_before, template_after, output_path: str):
    """(3/10) Shows a single template before and after elimination."""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 4), sharex=True)
    
    ax1.imshow(template_before.reshape(1, -1), cmap='viridis', aspect='auto')
    ax1.set_title('Template Before Elimination (PCA Domain)')
    ax1.set_yticks([])

    ax2.imshow(template_after.reshape(1, -1), cmap='viridis', aspect='auto')
    ax2.set_title('Template After Elimination (Zeros indicate removed components)')
    ax2.set_xlabel('Principal Component Index')
    ax2.set_yticks([])
    
    _save_plot(fig, output_path, "03_template_transformation_heatmap.png")

def plot_tsne_clusters(templates_pca, labels, title: str, output_path: str, filename: str):
    """Helper for t-SNE plots."""
    tsne = TSNE(n_components=2, perplexity=min(29, len(templates_pca)-1), random_state=42, init='pca', learning_rate='auto')
    templates_2d = tsne.fit_transform(templates_pca)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = sns.scatterplot(
        x=templates_2d[:, 0], 
        y=templates_2d[:, 1], 
        hue=labels, 
        palette='viridis', 
        ax=ax,
        legend='full'
    )
    ax.set_title(title)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(title=str(labels.name), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    _save_plot(fig, output_path, filename)
    
# --- 🤫 C. Visualizing Privacy Enhancement ---
# plot_tsne_clusters is used for plots 4 and 5

# --- 📉 D. Biometric Performance Distributions ---

def plot_score_distributions(scores: Tuple[List[float], List[float]], elim_count: int, output_path: str):
    """(7-9/10) Plots overlapping genuine and imposter score distributions."""
    genuine_scores, imposter_scores = scores
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.histplot(imposter_scores, color="salmon", label='Imposter Scores', ax=ax, kde=True, stat='density')
    sns.histplot(genuine_scores, color="skyblue", label='Genuine Scores', ax=ax, kde=True, stat='density')
    
    ax.legend()
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title(f'Score Distribution after {elim_count} Eliminations')
    ax.set_xlim(0, 1)
    
    filename = f"0{7 + list(output_path.keys()).index(elim_count)}_score_distribution_{elim_count}_elims.png"
    _save_plot(fig, output_path['path'], filename.replace("07", "07_").replace("08", "08_").replace("09", "09_")) # hacky filename numbering

def plot_all_score_distributions(distributions: Dict, output_path: str):
    """Wrapper to generate plots 7, 8, and 9."""
    # Ensure keys are sorted for consistent file naming
    sorted_keys = sorted(distributions.keys())
    # Create a mapping for filename generation
    path_map = {'path': output_path}
    for key in sorted_keys:
        path_map[key] = None # Add key to the map
        plot_score_distributions(distributions[key], key, path_map)

# --- 🏆 E. The Final Summary ---

def plot_final_tradeoff(results: Dict[str, Any], output_path: str):
    """(10/10) The main summary plot, enhanced with annotations."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    elim_counts = results['elimination_counts']
    sex_accuracies = results['sex_accuracies']
    recog_scores = [s * 100 for s in results['recognition_scores']]

    ax.plot(elim_counts, sex_accuracies, 'r--', marker='x', label='Sex Estimation Accuracy (Privacy)')
    ax.plot(elim_counts, recog_scores, 'b-', marker='o', label='Recognition Performance (Mean Genuine Score %)')
    
    # --- Enhancements ---
    # Find optimal point: where accuracy is lowest before recognition drops significantly
    recog_scores_np = np.array(recog_scores)
    # A simple heuristic: find max recognition score before it drops by >15% from its peak
    peak_recog_idx = np.argmax(recog_scores_np)
    try:
        collapse_idx = np.where(recog_scores_np[peak_recog_idx:] < recog_scores_np[peak_recog_idx] * 0.85)[0][0] + peak_recog_idx
        collapse_elims = elim_counts[collapse_idx]
        ax.axvline(x=collapse_elims, color='gray', linestyle='--', linewidth=2)
        ax.annotate('Utility Collapse', xy=(collapse_elims, 50), xytext=(collapse_elims + 5, 60),
                    arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
    except IndexError:
        pass # No significant collapse found
    
    ax.set_title('Refined Privacy-Utility Trade-off in Cancelable Face Biometrics', fontsize=20)
    ax.set_xlabel('Number of Eliminated Principal Components', fontsize=14)
    ax.set_ylabel('Performance (%)', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(-5, 105)
    
    _save_plot(fig, output_path, "10_final_tradeoff_plot.png")