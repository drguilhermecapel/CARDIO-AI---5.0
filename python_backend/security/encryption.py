import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Optional

class EncryptionManager:
    """
    Manages AES-GCM encryption for sensitive patient data.
    """
    def __init__(self, key: Optional[bytes] = None):
        # In production, this key should come from a KMS or secure env var
        # AES-GCM requires 256-bit (32 bytes) key
        self.key = key or AESGCM.generate_key(bit_length=256)
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypts plaintext string using AES-GCM.
        Returns base64 encoded string containing nonce + ciphertext.
        """
        nonce = os.urandom(12) # 96-bit nonce recommended for GCM
        data = plaintext.encode('utf-8')
        ct = self.aesgcm.encrypt(nonce, data, None)
        
        # Combine nonce + ciphertext
        combined = nonce + ct
        return base64.b64encode(combined).decode('utf-8')

    def decrypt(self, encrypted_b64: str) -> str:
        """
        Decrypts base64 encoded string.
        """
        try:
            combined = base64.b64decode(encrypted_b64)
            nonce = combined[:12]
            ct = combined[12:]
            
            plaintext = self.aesgcm.decrypt(nonce, ct, None)
            return plaintext.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")

# Example
if __name__ == "__main__":
    manager = EncryptionManager()
    secret = "Patient: John Doe, Diagnosis: STEMI"
    enc = manager.encrypt(secret)
    print(f"Encrypted: {enc}")
    dec = manager.decrypt(enc)
    print(f"Decrypted: {dec}")
