"""Initialization module for VectorDB."""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json

from .security.crypto import SecureStorage, SecureComm, SecureDataProcessor
from .security.rate_limiter import RateLimiter
from .security.audit import AuditLogger, EventCategory, EventSeverity
from .store import VectorStore
from .server.api import init_api

logger = logging.getLogger(__name__)

class VectorDBInitializer:
    """Initialize and configure VectorDB components."""
    
    def __init__(
        self,
        storage_dir: str,
        dim: int,
        api_key: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.storage_dir = Path(storage_dir)
        self.dim = dim
        self.api_key = api_key
        self.config = config or self._load_default_config()
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize components
        self._init_components()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "security": {
                "rate_limit": {
                    "default_rate": 100,
                    "block_threshold": 5,
                    "block_duration": 3600,
                    "whitelist_ips": ["127.0.0.1"]
                },
                "audit": {
                    "rotation_size_mb": 100,
                    "retention_days": 90,
                    "enable_encryption": True
                },
                "authentication": {
                    "token_expiry_minutes": 60,
                    "require_2fa": False,
                    "allowed_auth_methods": ["jwt", "oauth2"]
                }
            },
            "storage": {
                "auto_save_interval": 60,
                "enable_compression": True,
                "backup_enabled": True,
                "backup_interval_hours": 24
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "enable_cors": True,
                "allowed_origins": ["*"],
                "enable_https": False
            },
            "vector_store": {
                "distance_metric": "cosine",
                "enable_hybrid_search": True,
                "chunk_overlap": 0.1,
                "enable_cache": True,
                "cache_size": 1000
            }
        }
    
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.storage_dir,
            self.storage_dir / "vectors",
            self.storage_dir / "security",
            self.storage_dir / "audit",
            self.storage_dir / "backups"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _init_components(self):
        """Initialize all components."""
        try:
            # Initialize security components
            self.secure_storage = SecureStorage(self.api_key)
            self.secure_comm = SecureComm(self.api_key)
            self.data_processor = SecureDataProcessor(self.api_key)
            
            # Initialize rate limiter
            self.rate_limiter = RateLimiter(
                config_dir=str(self.storage_dir / "security"),
                **self.config["security"]["rate_limit"]
            )
            
            # Initialize audit logger
            self.audit_logger = AuditLogger(
                config_dir=str(self.storage_dir / "audit"),
                **self.config["security"]["audit"]
            )
            
            # Initialize vector store
            self.vector_store = VectorStore(
                storage_dir=str(self.storage_dir / "vectors"),
                dim=self.dim,
                api_key=self.api_key,
                **self.config["vector_store"]
            )
            
            # Initialize API
            init_api(
                storage_dir=str(self.storage_dir),
                dim=self.dim,
                api_key=self.api_key
            )
            
            # Log successful initialization
            self.audit_logger.log_event(
                category=EventCategory.SYSTEM,
                severity=EventSeverity.INFO,
                event_type="SYSTEM_INIT",
                user_id="system",
                ip_address="127.0.0.1",
                endpoint="init",
                request_id="init",
                details={"config": self.config},
                status="success",
                duration_ms=0
            )
            
            logger.info("VectorDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VectorDB: {e}")
            
            # Log initialization error
            if hasattr(self, 'audit_logger'):
                self.audit_logger.log_event(
                    category=EventCategory.SYSTEM,
                    severity=EventSeverity.ERROR,
                    event_type="SYSTEM_INIT",
                    user_id="system",
                    ip_address="127.0.0.1",
                    endpoint="init",
                    request_id="init",
                    details={"error": str(e)},
                    status="error",
                    duration_ms=0
                )
            raise
    
    def save_config(self):
        """Save current configuration."""
        config_file = self.storage_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self):
        """Load configuration from file."""
        config_file = self.storage_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration."""
        self.config.update(new_config)
        self.save_config()
        
        # Reinitialize components with new config
        self._init_components()
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all components."""
        status = {
            "vector_store": "healthy",
            "security": "healthy",
            "api": "healthy",
            "overall": "healthy"
        }
        
        try:
            # Check vector store
            self.vector_store.get_stats()
        except Exception as e:
            status["vector_store"] = f"unhealthy: {str(e)}"
            status["overall"] = "unhealthy"
        
        try:
            # Check security components
            self.secure_storage.test_connection()
            self.rate_limiter.check_rate_limit("127.0.0.1", "health_check")
        except Exception as e:
            status["security"] = f"unhealthy: {str(e)}"
            status["overall"] = "unhealthy"
        
        return status
