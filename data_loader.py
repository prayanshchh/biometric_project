# /home/prayansh-chhablani/biometrics/codes/project/data_loader.py

import os
import pandas as pd
from typing import List, Dict

def load_and_label_data(dataset_path: str, num_images: int = 30) -> pd.DataFrame:
    """
    Scans a directory for images and manually assigns identity and sex labels
    using the correct, user-provided ground truth.
    """
    print(f"🔍 Scanning for images in: {dataset_path}")
    
    try:
        valid_extensions = ('.png', '.jpg', '.jpeg', '.jfif')
        all_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(valid_extensions)])
        
        if len(all_files) < num_images:
            raise ValueError(f"Error: Found only {len(all_files)} images, but {num_images} are required.")
        
        image_files = all_files[:num_images]
        print(f"✅ Found and selected {len(image_files)} images.")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The specified dataset path does not exist: {dataset_path}")

    metadata: List[Dict] = []
    
    # User-provided sex labels for the 30 images.
    correct_sexes = [
        'female', 'male', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'male',
        'male', 'male', 'male', 'female', 'female', 'female', 'female', 'male', 'male', 'male',
        'female', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'male', 'male'
    ][:num_images] # Slice the list to match the number of images requested
    
    for i, filename in enumerate(image_files):
        # Identity: e.g., 10 people, 2 images each for 20 images
        identity_id = (i // 2) + 1
        
        # Use the correct sex label from the user-provided list.
        sex_label = correct_sexes[i]
        
        metadata.append({
            'filepath': os.path.join(dataset_path, filename),
            'identity_id': identity_id,
            'sex': sex_label,
        })
        
    df = pd.DataFrame(metadata)
    print("✅ Manually assigned labels for identity and sex.")
    return df