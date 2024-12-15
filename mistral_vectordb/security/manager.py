"""Security manager for VectorDB."""

import jwt
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64
import os

from .crypto import SecureStorage, SecureComm, SecureDataProcessor
from .schemas import SECURITY_CONFIG_SCHEMA

logger = logging.getLogger(__name__)

class SecurityManager:
    """Manages security features for VectorDB."""
    
    def __init__(self, api_key: str, config_dir: Optional[str] = None):
        self.api_key = api_key
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.mistral_vectordb' / 'security'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.secure_storage = SecureStorage(api_key)
        self.secure_comm = SecureComm(api_key)
        self.data_processor = SecureDataProcessor(api_key)
        
        # Load or generate keys
        self._init_keys()
        
        # Load configuration
        self.config = self._load_config()
        
        # Start key rotation scheduler
        self._schedule_key_rotation()
    
    def _init_keys(self):
        """Initialize RSA key pair for JWT signing."""
        key_file = self.config_dir / 'private_key.pem'
        pub_key_file = self.config_dir / 'public_key.pem'
        
        if not key_file.exists():
            # Generate new key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Save private key
            with open(key_file, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            public_key = private_key.public_key()
            with open(pub_key_file, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
    
    def _load_config(self) -> Dict[str, Any]:
        """Load security configuration."""
        config_file = self.config_dir / 'security_config.json'
        
        if not config_file.exists():
            # Create default config
            config = {
                "encryption": {
                    "algorithm": "AES-256-GCM",
                    "key_rotation_days": 30,
                    "backup_encryption": True
                },
                "authentication": {
                    "token_expiry_minutes": 60,
                    "max_failed_attempts": 5,
                    "require_2fa": False
                },
                "audit": {
                    "log_level": "INFO",
                    "log_retention_days": 90,
                    "enable_audit_trail": True
                }
            }
            
            # Validate and save config
            self.data_processor.validate_schema(config, "security_config")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            return config
        
        with open(config_file, 'r') as f:
            config = json.load(f)
            self.data_processor.validate_schema(config, "security_config")
            return config
    
    def generate_token(self, user_id: str, permissions: Dict[str, Any]) -> str:
        """Generate JWT token with permissions."""
        expiry = datetime.utcnow() + timedelta(
            minutes=self.config["authentication"]["token_expiry_minutes"]
        )
        
        payload = {
            "sub": user_id,
            "permissions": permissions,
            "exp": expiry
        }
        
        with open(self.config_dir / 'private_key.pem', 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
        
        return jwt.encode(payload, private_key, algorithm="RS256")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload."""
        with open(self.config_dir / 'public_key.pem', 'rb') as f:
            public_key = serialization.load_pem_public_key(f.read())
        
        try:
            return jwt.decode(token, public_key, algorithms=["RS256"])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def rotate_keys(self):
        """Rotate encryption keys."""
        logger.info("Rotating encryption keys")
        self._init_keys()  # Generate new RSA key pair
        self.secure_storage.rotate_keys()  # Rotate storage encryption keys
    
    def _schedule_key_rotation(self):
        """Schedule periodic key rotation."""
        rotation_days = self.config["encryption"]["key_rotation_days"]
        next_rotation = datetime.now() + timedelta(days=rotation_days)
        logger.info(f"Next key rotation scheduled for: {next_rotation}")
    
    def audit_log(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log security audit event."""
        if not self.config["audit"]["enable_audit_trail"]:
            return
            
        log_file = self.config_dir / 'audit.log'
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def cleanup_audit_logs(self):
        """Clean up old audit logs."""
        if not self.config["audit"]["enable_audit_trail"]:
            return
            
        retention_days = self.config["audit"]["log_retention_days"]
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        log_file = self.config_dir / 'audit.log'
        if not log_file.exists():
            return
            
        # Read existing logs
        with open(log_file, 'r') as f:
            logs = [json.loads(line) for line in f]
        
        # Filter out old logs
        current_logs = [
            log for log in logs
            if datetime.fromisoformat(log["timestamp"]) > cutoff_date
        ]
        
        # Write back current logs
        with open(log_file, 'w') as f:
            for log in current_logs:
                f.write(json.dumps(log) + '\n')
