# /home/prayansh-chhablani/biometrics/codes/project/data_exporter.py

import numpy as np
import os

def save_templates(
    output_path: str,
    filename: str,
    original_templates: np.ndarray,
    enhanced_templates: np.ndarray
):
    """
    Saves the original and privacy-enhanced templates to a compressed NumPy file (.npz).

    Args:
        output_path (str): The directory where the file will be saved.
        filename (str): The name of the output file (e.g., 'processed_templates.npz').
        original_templates (np.ndarray): The unprotected templates from the feature extractor.
        enhanced_templates (np.ndarray): The final templates after the Multi-IVE process.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    full_path = os.path.join(output_path, filename)
    
    print(f"\n💾 Exporting processed templates...")
    print(f"   - Original templates shape: {original_templates.shape}")
    print(f"   - Enhanced templates shape: {enhanced_templates.shape}")
    
    np.savez_compressed(
        full_path,
        original=original_templates,
        enhanced=enhanced_templates
    )
    
    print(f"✅ Templates successfully saved to: {full_path}")