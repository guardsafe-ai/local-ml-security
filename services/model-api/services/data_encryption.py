"""
Data Encryption Service
Provides encryption/decryption for sensitive data
"""

import logging
import base64
import hashlib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DataEncryption:
    """Data encryption service for sensitive information"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize encryption service
        
        Args:
            encryption_key: Optional encryption key. If not provided, uses a default.
        """
        if encryption_key:
            self.key = encryption_key.encode()
        else:
            # Use a simple key for demo purposes (in production, use proper encryption)
            self.key = b"demo_encryption_key_32_bytes_long"
        
        # Simple XOR encryption for demo purposes
        self.key_bytes = self.key[:32]  # Ensure key is 32 bytes
    
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data using simple XOR encryption
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            data_bytes = data.encode()
            encrypted = bytearray()
            key_len = len(self.key_bytes)
            
            for i, byte in enumerate(data_bytes):
                encrypted.append(byte ^ self.key_bytes[i % key_len])
            
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"❌ [ENCRYPTION] Failed to encrypt data: {e}")
            return data  # Return original data if encryption fails
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data using simple XOR encryption
        
        Args:
            encrypted_data: Encrypted data as base64 string
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = bytearray()
            key_len = len(self.key_bytes)
            
            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ self.key_bytes[i % key_len])
            
            return decrypted.decode()
        except Exception as e:
            logger.error(f"❌ [ENCRYPTION] Failed to decrypt data: {e}")
            return encrypted_data  # Return original data if decryption fails
    
    def encrypt_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in a dictionary
        
        Args:
            data_dict: Dictionary to encrypt
            
        Returns:
            Dictionary with sensitive fields encrypted
        """
        sensitive_fields = ['text', 'input', 'prompt', 'query', 'message']
        encrypted_dict = data_dict.copy()
        
        for key, value in encrypted_dict.items():
            if key.lower() in sensitive_fields and isinstance(value, str):
                encrypted_dict[key] = self.encrypt_data(value)
        
        return encrypted_dict
    
    def decrypt_dict(self, encrypted_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt sensitive fields in a dictionary
        
        Args:
            encrypted_dict: Dictionary with encrypted fields
            
        Returns:
            Dictionary with sensitive fields decrypted
        """
        sensitive_fields = ['text', 'input', 'prompt', 'query', 'message']
        decrypted_dict = encrypted_dict.copy()
        
        for key, value in decrypted_dict.items():
            if key.lower() in sensitive_fields and isinstance(value, str):
                try:
                    decrypted_dict[key] = self.decrypt_data(value)
                except:
                    # If decryption fails, keep original value
                    pass
        
        return decrypted_dict
    
    def hash_data(self, data: str) -> str:
        """
        Create a hash of data for indexing/searching
        
        Args:
            data: Data to hash
            
        Returns:
            SHA-256 hash of the data
        """
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get_encryption_key(self) -> str:
        """Get the current encryption key"""
        return base64.b64encode(self.key_bytes).decode()

# Global encryption instance
data_encryption = DataEncryption()
