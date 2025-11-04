# /home/prayansh-chhablani/biometrics/codes/project/feature_extractor.py

import pandas as pd
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

def extract_features(df: pd.DataFrame, model_name: str = 'ArcFace') -> np.ndarray:
    """
    Extracts deep face embeddings from a list of image filepaths using DeepFace.

    Args:
        df (pd.DataFrame): DataFrame containing a 'filepath' column with image paths.
        model_name (str): The name of the face recognition model to use.

    Returns:
        np.ndarray: A (N, D) numpy array where N is the number of images and D is
                    the embedding dimension (512 for ArcFace).
    """
    print(f"🤖 Extracting face features using '{model_name}' model...")
    
    # The represent function returns a list of dictionaries, we need to extract the 'embedding'
    # We use a tqdm progress bar for better user experience during this potentially long step.
    embeddings = [
        DeepFace.represent(
            img_path=row['filepath'],
            model_name=model_name,
            enforce_detection=False # Set to False to avoid errors if face detection is tricky
        )[0]['embedding']
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Embeddings")
    ]
    
    unprotected_templates = np.array(embeddings, dtype=np.float32)
    print(f"✅ Feature extraction complete. Shape of templates: {unprotected_templates.shape}")
    
    return unprotected_templates