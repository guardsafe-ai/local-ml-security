"""
Data Encryption at Rest Service
Implements encryption for sensitive data storage
"""

import logging
import os
import base64
import json
from typing import Dict, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib

logger = logging.getLogger(__name__)

class DataEncryption:
    """Handles encryption and decryption of sensitive data"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv("ENCRYPTION_MASTER_KEY")
        if not self.master_key:
            # Generate a default key for development (NOT for production!)
            self.master_key = Fernet.generate_key().decode()
            logger.warning("⚠️ [ENCRYPT] Using generated master key - set ENCRYPTION_MASTER_KEY for production")
        
        self.fernet = Fernet(self.master_key.encode() if isinstance(self.master_key, str) else self.master_key)
        
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]], 
                    data_type: str = "general") -> Dict[str, Any]:
        """
        Encrypt sensitive data
        
        Args:
            data: Data to encrypt
            data_type: Type of data for classification
            
        Returns:
            Encrypted data with metadata
        """
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Encrypt data
            encrypted_data = self.fernet.encrypt(data_bytes)
            
            # Create hash for integrity check
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            
            result = {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "data_type": data_type,
                "encryption_method": "fernet",
                "data_hash": data_hash,
                "encrypted": True
            }
            
            logger.debug(f"✅ [ENCRYPT] Encrypted {data_type} data ({len(data_bytes)} bytes)")
            return result
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to encrypt {data_type} data: {e}")
            raise
    
    def decrypt_data(self, encrypted_package: Dict[str, Any]) -> Union[str, bytes, Dict[str, Any]]:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_package: Encrypted data package
            
        Returns:
            Decrypted data
        """
        try:
            # Extract encrypted data
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
            
            # Decrypt data
            decrypted_bytes = self.fernet.decrypt(encrypted_data)
            
            # Verify integrity
            if "data_hash" in encrypted_package:
                expected_hash = encrypted_package["data_hash"]
                actual_hash = hashlib.sha256(decrypted_bytes).hexdigest()
                
                if expected_hash != actual_hash:
                    raise ValueError("Data integrity check failed - data may be corrupted")
            
            # Convert back to original format
            data_type = encrypted_package.get("data_type", "general")
            
            if data_type in ["json", "dict"]:
                try:
                    return json.loads(decrypted_bytes.decode('utf-8'))
                except json.JSONDecodeError:
                    return decrypted_bytes.decode('utf-8')
            elif data_type == "string":
                return decrypted_bytes.decode('utf-8')
            else:
                return decrypted_bytes
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to decrypt data: {e}")
            raise
    
    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """
        Encrypt a file
        
        Args:
            file_path: Path to file to encrypt
            output_path: Path for encrypted file (optional)
            
        Returns:
            Path to encrypted file
        """
        try:
            if output_path is None:
                output_path = file_path + ".encrypted"
            
            # Read file
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Encrypt data
            encrypted_package = self.encrypt_data(file_data, "file")
            
            # Write encrypted file
            with open(output_path, 'w') as f:
                json.dump(encrypted_package, f)
            
            logger.info(f"✅ [ENCRYPT] Encrypted file: {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to encrypt file {file_path}: {e}")
            raise
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str = None) -> str:
        """
        Decrypt a file
        
        Args:
            encrypted_file_path: Path to encrypted file
            output_path: Path for decrypted file (optional)
            
        Returns:
            Path to decrypted file
        """
        try:
            if output_path is None:
                output_path = encrypted_file_path.replace(".encrypted", "")
            
            # Read encrypted file
            with open(encrypted_file_path, 'r') as f:
                encrypted_package = json.load(f)
            
            # Decrypt data
            decrypted_data = self.decrypt_data(encrypted_package)
            
            # Write decrypted file
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info(f"✅ [ENCRYPT] Decrypted file: {encrypted_file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to decrypt file {encrypted_file_path}: {e}")
            raise
    
    def encrypt_database_field(self, value: str, field_type: str = "text") -> str:
        """
        Encrypt a database field value
        
        Args:
            value: Value to encrypt
            field_type: Type of field (text, email, phone, etc.)
            
        Returns:
            Encrypted value as base64 string
        """
        try:
            encrypted_package = self.encrypt_data(value, field_type)
            return encrypted_package["encrypted_data"]
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to encrypt database field: {e}")
            raise
    
    def decrypt_database_field(self, encrypted_value: str, field_type: str = "text") -> str:
        """
        Decrypt a database field value
        
        Args:
            encrypted_value: Encrypted value
            field_type: Type of field
            
        Returns:
            Decrypted value
        """
        try:
            encrypted_package = {
                "encrypted_data": encrypted_value,
                "data_type": field_type,
                "encryption_method": "fernet"
            }
            return self.decrypt_data(encrypted_package)
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to decrypt database field: {e}")
            raise
    
    def generate_field_key(self, field_name: str, salt: bytes = None) -> bytes:
        """
        Generate a field-specific encryption key
        
        Args:
            field_name: Name of the field
            salt: Optional salt for key derivation
            
        Returns:
            Derived key
        """
        try:
            if salt is None:
                salt = os.urandom(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            key = kdf.derive(f"{self.master_key}:{field_name}".encode())
            return key
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to generate field key: {e}")
            raise
    
    def encrypt_with_field_key(self, data: str, field_name: str) -> Dict[str, Any]:
        """
        Encrypt data with field-specific key
        
        Args:
            data: Data to encrypt
            field_name: Name of the field
            
        Returns:
            Encrypted data with salt
        """
        try:
            salt = os.urandom(16)
            key = self.generate_field_key(field_name, salt)
            
            # Use AES encryption
            cipher = Cipher(algorithms.AES(key), modes.CBC(salt), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            data_bytes = data.encode('utf-8')
            padding_length = 16 - (len(data_bytes) % 16)
            padded_data = data_bytes + bytes([padding_length] * padding_length)
            
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "salt": base64.b64encode(salt).decode('utf-8'),
                "field_name": field_name,
                "encryption_method": "aes_cbc"
            }
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to encrypt with field key: {e}")
            raise
    
    def decrypt_with_field_key(self, encrypted_package: Dict[str, Any]) -> str:
        """
        Decrypt data with field-specific key
        
        Args:
            encrypted_package: Encrypted data package
            
        Returns:
            Decrypted data
        """
        try:
            salt = base64.b64decode(encrypted_package["salt"])
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
            field_name = encrypted_package["field_name"]
            
            key = self.generate_field_key(field_name, salt)
            
            # Use AES decryption
            cipher = Cipher(algorithms.AES(key), modes.CBC(salt), backend=default_backend())
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            padding_length = padded_data[-1]
            decrypted_data = padded_data[:-padding_length]
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ [ENCRYPT] Failed to decrypt with field key: {e}")
            raise

# Global encryption instance
data_encryption = DataEncryption()
