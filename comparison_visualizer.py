# /home/prayansh-chhablani/biometrics/codes/project/comparison_visualizer.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

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
    print(f"   💾 Comparison plot saved to: {full_path}")

def plot_score_distribution_comparison(data: Dict, output_path: str):
    """Plots the score distributions of both methods on one chart."""
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.histplot(data['current_imposter'], color="salmon", label='Imposter (Current)', ax=ax, kde=True, stat='density', alpha=0.6)
    sns.histplot(data['current_genuine'], color="skyblue", label='Genuine (Current)', ax=ax, kde=True, stat='density', alpha=0.6)
    
    sns.kdeplot(data['triplet_imposter'], color="red", label='Imposter (Triplet HE)', ax=ax, linewidth=3, linestyle='--')
    sns.kdeplot(data['triplet_genuine'], color="blue", label='Genuine (Triplet HE)', ax=ax, linewidth=3, linestyle='--')

    ax.legend()
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Density')
    ax.set_title('Recognition Performance: Current vs. Triplet Protection')
    ax.set_xlim(0, 1)
    _save_plot(fig, output_path, "comparison_01_score_distributions.png")

def plot_timing_comparison(data: Dict, output_path: str):
    """Plots a bar chart comparing the processing time."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    methods = ['Current Approach (Plaintext)', 'Triplet Protection (Encrypted)']
    times = [data['time_current'], data['time_triplet']]
    
    bars = ax.bar(methods, times, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Cost of Comparison')
    ax.bar_label(bars, fmt='%.2f s')
    
    _save_plot(fig, output_path, "comparison_02_timing.png")