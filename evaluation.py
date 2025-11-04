# /home/prayansh-chhablani/biometrics/codes/project/evaluation.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

def calculate_similarity_scores(templates: np.ndarray, labels: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Calculates the distribution of genuine (same identity) and imposter
    (different identity) similarity scores.

    Args:
        templates (np.ndarray): The (N, D) array of biometric templates.
        labels (np.ndarray): The (N,) array of corresponding identity labels.

    Returns:
        A tuple containing two lists:
        - genuine_scores (List[float]): Similarities for pairs with the same identity.
        - imposter_scores (List[float]): Similarities for pairs with different identities.
    """
    # Calculate the pairwise cosine similarity for all templates
    cos_sim_matrix = cosine_similarity(templates)
    
    genuine_scores = []
    imposter_scores = []
    
    # Iterate through the upper triangle of the similarity matrix to avoid duplicates
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                genuine_scores.append(cos_sim_matrix[i, j])
            else:
                imposter_scores.append(cos_sim_matrix[i, j])
                
    return genuine_scores, imposter_scores