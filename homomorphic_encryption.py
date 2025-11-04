# /home/prayansh-chhablani/biometrics/codes/project/homomorphic_encryption.py

import numpy as np
import time
from Pyfhel import Pyfhel
from typing import List, Tuple

class HECipher:
    """
    An upgraded wrapper class to handle batch Homomorphic Encryption operations
    for a full biometric performance evaluation.
    """
    def __init__(self):
        print("INITIALIZING HOMOMORPHIC ENCRYPTION CONTEXT...")
        self.he = Pyfhel()
        self.he.contextGen(scheme='bfv', n=2**14, t_bits=20, sec=128)
        self.he.keyGen()
        # Relinearization keys are essential for managing ciphertext size after multiplication.
        self.he.relinKeyGen()
        print("✅ HE Context and Keys Generated.")

    def encrypt_batch(self, templates: np.ndarray) -> List:
        """Encrypts a batch of numpy vectors."""
        print(f"   - Encrypting a batch of {templates.shape[0]} templates...")
        encrypted_templates = []
        for t in templates:
            scaled_vector = (t * 1000).astype(np.int64)
            encrypted_templates.append(self.he.encrypt(scaled_vector))
        return encrypted_templates

    def calculate_all_encrypted_scores(self, encrypted_templates: List, labels: np.ndarray) -> Tuple[List[float], List[float], float]:
        """
        Calculates all genuine and imposter scores from a list of encrypted templates.
        The core comparison happens entirely in the encrypted domain.
        """
        print("   - Starting all-pairs distance calculation in the ENCRYPTED domain...")
        num_templates = len(encrypted_templates)
        encrypted_distances = []
        
        start_time = time.time()

        # Iterate through the upper triangle of the matrix
        for i in range(num_templates):
            for j in range(i + 1, num_templates):
                ctxt1 = encrypted_templates[i]
                ctxt2 = encrypted_templates[j]
                
                # Perform encrypted subtraction
                ctxt_diff = ctxt1 - ctxt2
                
                # **THE FIX**: Perform squaring and then immediately relinearize
                # The `~` operator performs relinearization, returning the ciphertext
                # to a standard size that other functions can use.
                ctxt_sq = ~(ctxt_diff * ctxt_diff)

                # Now cumul_add will receive a correctly sized ciphertext
                encrypted_distances.append(self.he.cumul_add(ctxt_sq, in_new_ctxt=True))

        processing_time = time.time() - start_time
        print(f"   - Encrypted calculations finished in {processing_time:.2f} seconds.")

        # --- Decryption occurs ONLY on the final distance scores ---
        print("   - Decrypting final distance scores...")
        decrypted_distances = [self.he.decrypt(d)[0] for d in encrypted_distances]
        scaled_distances = np.array(decrypted_distances) / (1000**2)

        # Sort into genuine and imposter lists
        genuine_scores_dist = []
        imposter_scores_dist = []
        k = 0
        for i in range(num_templates):
            for j in range(i + 1, num_templates):
                similarity = 1 / (1 + scaled_distances[k])
                if labels[i] == labels[j]:
                    genuine_scores_dist.append(similarity)
                else:
                    imposter_scores_dist.append(similarity)
                k += 1

        return genuine_scores_dist, imposter_scores_dist, processing_time