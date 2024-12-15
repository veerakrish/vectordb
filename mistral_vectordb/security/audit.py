"""Enhanced audit logging for VectorDB."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
import time
import hashlib
import os
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class EventSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventCategory(Enum):
    """Categories for audit events."""
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    DATA_ACCESS = "DATA_ACCESS"
    CONFIGURATION = "CONFIGURATION"
    SECURITY = "SECURITY"
    SYSTEM = "SYSTEM"

@dataclass
class AuditEvent:
    """Audit event data structure."""
    timestamp: str
    event_id: str
    category: EventCategory
    severity: EventSeverity
    event_type: str
    user_id: str
    ip_address: str
    endpoint: str
    request_id: str
    details: Dict[str, Any]
    status: str
    duration_ms: float
    metadata: Dict[str, Any]

class AuditLogger:
    """Enhanced audit logging system."""
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        rotation_size_mb: int = 100,
        retention_days: int = 90,
        enable_encryption: bool = True
    ):
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.mistral_vectordb' / 'security' / 'audit'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.rotation_size_mb = rotation_size_mb
        self.retention_days = retention_days
        self.enable_encryption = enable_encryption
        
        # Initialize encryption key if enabled
        if self.enable_encryption:
            self._init_encryption()
        
        # Start maintenance thread
        self._start_maintenance_thread()
    
    def _init_encryption(self):
        """Initialize encryption for audit logs."""
        key_file = self.config_dir / 'audit_key'
        if not key_file.exists():
            key = os.urandom(32)
            with open(key_file, 'wb') as f:
                f.write(key)
    
    def _get_current_log_file(self) -> Path:
        """Get the current log file path."""
        current_file = self.config_dir / 'audit.log'
        
        # Check if rotation needed
        if current_file.exists() and current_file.stat().st_size > self.rotation_size_mb * 1024 * 1024:
            # Rotate file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_file = self.config_dir / f'audit_{timestamp}.log'
            current_file.rename(new_file)
            
            # Compress old file
            self._compress_log_file(new_file)
        
        return current_file
    
    def _compress_log_file(self, file_path: Path):
        """Compress and optionally encrypt old log files."""
        import gzip
        import shutil
        
        # Compress file
        with open(file_path, 'rb') as f_in:
            with gzip.open(f"{file_path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Encrypt if enabled
        if self.enable_encryption:
            self._encrypt_file(f"{file_path}.gz")
        
        # Remove original file
        file_path.unlink()
    
    def _encrypt_file(self, file_path: str):
        """Encrypt a file using the audit key."""
        from cryptography.fernet import Fernet
        
        # Read encryption key
        with open(self.config_dir / 'audit_key', 'rb') as f:
            key = f.read()
        
        # Encrypt file
        f = Fernet(base64.b64encode(key))
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = f.encrypt(file_data)
        with open(f"{file_path}.encrypted", 'wb') as file:
            file.write(encrypted_data)
        
        # Remove unencrypted file
        os.remove(file_path)
    
    def log_event(
        self,
        category: EventCategory,
        severity: EventSeverity,
        event_type: str,
        user_id: str,
        ip_address: str,
        endpoint: str,
        request_id: str,
        details: Dict[str, Any],
        status: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event."""
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_id=self._generate_event_id(),
            category=category,
            severity=severity,
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            request_id=request_id,
            details=details,
            status=status,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        self._write_event(event)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = datetime.utcnow().isoformat()
        random_bytes = os.urandom(8)
        return hashlib.sha256(f"{timestamp}{random_bytes}".encode()).hexdigest()[:16]
    
    def _write_event(self, event: AuditEvent):
        """Write event to log file."""
        log_file = self._get_current_log_file()
        
        event_dict = asdict(event)
        event_dict['category'] = event.category.value
        event_dict['severity'] = event.severity.value
        
        with open(log_file, 'a') as f:
            json.dump(event_dict, f)
            f.write('\n')
    
    def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[AuditEvent]:
        """Query audit logs with filters."""
        events = []
        
        # Get all log files
        log_files = list(self.config_dir.glob('audit*.log'))
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    event_dict = json.loads(line)
                    event_time = datetime.fromisoformat(event_dict['timestamp'])
                    
                    # Apply filters
                    if start_time and event_time < start_time:
                        continue
                    if end_time and event_time > end_time:
                        continue
                    if category and event_dict['category'] != category.value:
                        continue
                    if severity and event_dict['severity'] != severity.value:
                        continue
                    if user_id and event_dict['user_id'] != user_id:
                        continue
                    if ip_address and event_dict['ip_address'] != ip_address:
                        continue
                    if status and event_dict['status'] != status:
                        continue
                    
                    events.append(AuditEvent(**event_dict))
        
        return events
    
    def _maintenance_thread(self):
        """Perform periodic maintenance tasks."""
        while True:
            try:
                # Clean old logs
                cutoff_date = datetime.now() - timedelta(days=self.retention_days)
                for log_file in self.config_dir.glob('audit_*.log*'):
                    # Extract timestamp from filename
                    try:
                        timestamp_str = log_file.stem.split('_')[1]
                        file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        if file_date < cutoff_date:
                            log_file.unlink()
                    except (IndexError, ValueError):
                        continue
                
                # Sleep for a day
                time.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error in audit maintenance thread: {e}")
                time.sleep(3600)  # Sleep for an hour on error
    
    def _start_maintenance_thread(self):
        """Start the maintenance thread."""
        thread = threading.Thread(target=self._maintenance_thread, daemon=True)
        thread.start()
