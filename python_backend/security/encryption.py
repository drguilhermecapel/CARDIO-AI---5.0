import os
import base64
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

class EncryptionManager:
    """
    Manages AES-256-GCM encryption for data at rest.
    """
    def __init__(self, key_hex: str = None):
        """
        Initialize with a 32-byte (64 hex chars) key for AES-256.
        If no key provided, attempts to load from env var ENCRYPTION_KEY.
        """
        key_env = os.getenv("ENCRYPTION_KEY")
        if key_hex:
            self.key = bytes.fromhex(key_hex)
        elif key_env:
            self.key = bytes.fromhex(key_env)
        else:
            # Generate a new key for demo purposes (In prod, use KMS)
            print("WARNING: Generating ephemeral encryption key. Data will be lost on restart.")
            self.key = AESGCM.generate_key(bit_length=256)

        if len(self.key) != 32:
            raise ValueError("Encryption key must be 32 bytes (256 bits) for AES-256.")
            
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypts string data using AES-256-GCM.
        Returns base64 encoded string containing nonce + ciphertext + tag.
        """
        nonce = os.urandom(12) # NIST recommended 96-bit nonce
        data = plaintext.encode('utf-8')
        ciphertext = self.aesgcm.encrypt(nonce, data, None)
        
        # Combine nonce + ciphertext (tag is included in ciphertext by cryptography lib usually, 
        # but AESGCM.encrypt returns ciphertext + tag appended)
        combined = nonce + ciphertext
        return base64.b64encode(combined).decode('utf-8')

    def decrypt(self, encrypted_b64: str) -> str:
        """
        Decrypts base64 encoded string.
        """
        try:
            combined = base64.b64decode(encrypted_b64)
            nonce = combined[:12]
            ciphertext = combined[12:]
            
            plaintext_bytes = self.aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext_bytes.decode('utf-8')
        except (InvalidTag, ValueError) as e:
            raise ValueError("Decryption failed: Invalid key or corrupted data.") from e

    def rotate_key(self, new_key_hex: str) -> None:
        """
        Updates the active key. 
        Note: In a real system, you'd need to re-encrypt old data.
        """
        self.key = bytes.fromhex(new_key_hex)
        self.aesgcm = AESGCM(self.key)

# Singleton instance
# crypto_manager = EncryptionManager()
