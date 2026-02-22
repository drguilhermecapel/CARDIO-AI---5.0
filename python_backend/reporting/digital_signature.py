import hashlib
import base64
import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives import serialization
from typing import Tuple

class DigitalSignature:
    """
    Handles digital signing of medical reports to ensure integrity and non-repudiation.
    """
    def __init__(self, private_key_pem: bytes = None):
        if private_key_pem:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem, password=None
            )
        else:
            # Generate ephemeral key for demo
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
        
        self.public_key = self.private_key.public_key()

    def sign_data(self, data: str) -> str:
        """
        Signs the string data (usually JSON or XML content).
        Returns Base64 encoded signature.
        """
        signature = self.private_key.sign(
            data.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def verify_signature(self, data: str, signature_b64: str, public_key_pem: bytes = None) -> bool:
        """
        Verifies the signature.
        """
        if public_key_pem:
            public_key = serialization.load_pem_public_key(public_key_pem)
        else:
            public_key = self.public_key

        try:
            signature = base64.b64decode(signature_b64)
            public_key.verify(
                signature,
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def get_public_key_pem(self) -> str:
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
