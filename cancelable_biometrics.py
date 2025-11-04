# /home/prayansh-chhablani/biometrics/codes/project/cancelable_biometrics.py

import numpy as np
from scipy.stats import ortho_group

def generate_key(dimension: int) -> np.ndarray:
    """
    Generates a random orthogonal matrix to be used as a key for BioHashing.

    Args:
        dimension (int): The dimension of the square key matrix (e.g., 512 for ArcFace).

    Returns:
        np.ndarray: A random (dimension x dimension) orthogonal matrix.
    """
    print(f"🔑 Generating a new random orthogonal key of size {dimension}x{dimension}.")
    return ortho_group.rvs(dim=dimension)

def biohash(template: np.ndarray, key_matrix: np.ndarray) -> np.ndarray:
    """
    Applies the BioHashing transformation to a biometric template.

    Args:
        template (np.ndarray): A 1D feature vector.
        key_matrix (np.ndarray): The orthogonal key matrix.

    Returns:
        np.ndarray: The binarized, cancelable biometric template (hash).
    """
    # 1. Project the template using the random key
    transformed_template = np.dot(template, key_matrix)
    
    # 2. Binarize the result based on a zero threshold
    hashed_template = (transformed_template > 0).astype(int)
    
    return hashed_template
