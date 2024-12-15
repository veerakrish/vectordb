"""Configuration management for Mistral VectorDB."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json

class Config:
    """Configuration manager for Mistral VectorDB."""
    
    DEFAULT_CONFIG = {
        'storage': {
            'base_dir': '~/.mistral_vectordb',
            'index_dir': 'indices',
            'document_dir': 'documents',
            'backup_dir': 'backups',
            'max_file_size': 100 * 1024 * 1024,  # 100MB
        },
        'security': {
            'enable_encryption': False,
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
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to custom config file. If None, uses default.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load custom config if provided
        if config_path:
            self.load_config(config_path)
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def base_dir(self) -> Path:
        """Get base directory for all storage."""
        return Path(os.path.expanduser(self.config['storage']['base_dir']))
    
    @property
    def index_dir(self) -> Path:
        """Get directory for vector indices."""
        return self.base_dir / self.config['storage']['index_dir']
    
    @property
    def document_dir(self) -> Path:
        """Get directory for document storage."""
        return self.base_dir / self.config['storage']['document_dir']
    
    @property
    def backup_dir(self) -> Path:
        """Get directory for backups."""
        return self.base_dir / self.config['storage']['backup_dir']
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            self._update_config(custom_config)
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to file."""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _update_config(self, custom_config: Dict[str, Any]) -> None:
        """Update configuration with custom values."""
        for section, values in custom_config.items():
            if section in self.config:
                self.config[section].update(values)
    
    def get_storage_path(self, name: str, create: bool = True) -> Path:
        """Get path for storing data with given name.
        
        Args:
            name: Name of the storage (e.g., collection name)
            create: Whether to create directory if it doesn't exist
        
        Returns:
            Path to storage directory
        """
        path = self.base_dir / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
