"""Setup tool for Mistral VectorDB initial configuration."""

import os
from pathlib import Path
import json
from getpass import getpass
import keyring
from cryptography.fernet import Fernet
import base64
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SetupTool:
    """Setup tool for initial configuration."""
    
    def __init__(self):
        """Initialize setup tool."""
        self.config_dir = Path.home() / '.mistral_vectordb'
        self.env_file = self.config_dir / '.env'
        self.config_file = self.config_dir / 'config.json'
        
        # Ensure directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def initial_setup(self, api_key: Optional[str] = None) -> None:
        """Run initial setup process.
        
        Args:
            api_key: Optional API key. If not provided, will prompt user.
        """
        try:
            # Get API key
            if not api_key:
                api_key = self._get_api_key_input()
            
            # Store API key securely
            self._store_api_key(api_key)
            
            # Create default config
            self._create_default_config()
            
            print("\nSetup completed successfully!")
            print(f"Configuration stored in: {self.config_dir}")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def _get_api_key_input(self) -> str:
        """Get API key from user input."""
        print("\nMistral VectorDB Setup")
        print("=====================")
        print("\nPlease enter your Mistral API key.")
        print("You can find this at: https://console.mistral.ai/api-keys/")
        
        while True:
            api_key = getpass("Mistral API Key: ")
            if api_key.strip():
                return api_key
            print("API key cannot be empty. Please try again.")
    
    def _store_api_key(self, api_key: str) -> None:
        """Store API key securely.
        
        Uses system keyring for secure storage and creates an encrypted
        reference in .env file.
        """
        # Generate encryption key
        encryption_key = Fernet.generate_key()
        fernet = Fernet(encryption_key)
        
        # Encrypt API key
        encrypted_key = fernet.encrypt(api_key.encode())
        
        # Store encryption key in system keyring
        keyring.set_password(
            "mistral_vectordb",
            "encryption_key",
            base64.b64encode(encryption_key).decode()
        )
        
        # Store encrypted API key in .env
        with open(self.env_file, 'w') as f:
            f.write(f"MISTRAL_API_KEY={base64.b64encode(encrypted_key).decode()}\n")
        
        # Secure the file
        os.chmod(self.env_file, 0o600)
    
    def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = {
            'storage': {
                'base_dir': str(self.config_dir),
                'index_dir': 'indices',
                'document_dir': 'documents',
                'backup_dir': 'backups',
                'max_file_size': 100 * 1024 * 1024,  # 100MB
            },
            'security': {
                'enable_encryption': True,
                'encryption_method': 'fernet',
                'key_derivation': 'pbkdf2',
            },
            'processing': {
                'chunk_size': 500,
                'chunk_overlap': 0.1,
                'enable_ocr': False,
                'language_model': 'en_core_web_sm',
            },
            'indexing': {
                'vector_dim': 1024,
                'distance_metric': 'cosine',
                'ef_construction': 200,
                'M': 16,
            },
            'search': {
                'enable_hybrid_search': True,
                'hybrid_alpha': 0.7,
                'enable_mmr': True,
                'mmr_lambda': 0.5,
                'enable_reranking': True,
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        # Secure the file
        os.chmod(self.config_file, 0o600)
    
    @staticmethod
    def get_api_key() -> str:
        """Get stored API key.
        
        Returns:
            Decrypted API key
        """
        try:
            config_dir = Path.home() / '.mistral_vectordb'
            env_file = config_dir / '.env'
            
            # Read encrypted key from .env
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('MISTRAL_API_KEY='):
                        encrypted_key = base64.b64decode(line.split('=')[1].strip())
                        break
                else:
                    raise ValueError("API key not found in .env file")
            
            # Get encryption key from keyring
            encryption_key = base64.b64decode(
                keyring.get_password("mistral_vectordb", "encryption_key")
            )
            
            # Decrypt API key
            fernet = Fernet(encryption_key)
            api_key = fernet.decrypt(encrypted_key).decode()
            
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            raise

def setup_mistral_vectordb(api_key: Optional[str] = None) -> None:
    """Setup Mistral VectorDB with API key."""
    setup_tool = SetupTool()
    setup_tool.initial_setup(api_key)
