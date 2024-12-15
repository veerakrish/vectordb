"""Rate limiting and IP blocking for VectorDB."""

from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List
import json
from pathlib import Path
import ipaddress
import logging
from collections import defaultdict
import threading
import time

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting with IP blocking capabilities."""
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        default_rate_limit: int = 100,  # requests per minute
        block_threshold: int = 5,  # violations before blocking
        block_duration: int = 3600,  # seconds to block
        whitelist_ips: Optional[List[str]] = None
    ):
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.mistral_vectordb' / 'security'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_rate_limit = default_rate_limit
        self.block_threshold = block_threshold
        self.block_duration = block_duration
        
        # Initialize storage
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.violations: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Initialize IP whitelist
        self.whitelist_ips = set()
        if whitelist_ips:
            for ip in whitelist_ips:
                self.add_to_whitelist(ip)
        
        # Load existing blocks
        self._load_blocks()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def add_to_whitelist(self, ip: str):
        """Add IP or CIDR range to whitelist."""
        try:
            # Check if it's a CIDR range
            if '/' in ip:
                ipaddress.ip_network(ip)
            else:
                ipaddress.ip_address(ip)
            self.whitelist_ips.add(ip)
        except ValueError as e:
            logger.error(f"Invalid IP or CIDR range: {ip}")
            raise ValueError(f"Invalid IP or CIDR range: {ip}")
    
    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        if ip in self.whitelist_ips:
            return True
            
        # Check CIDR ranges
        ip_obj = ipaddress.ip_address(ip)
        for whitelist_ip in self.whitelist_ips:
            if '/' in whitelist_ip:
                if ip_obj in ipaddress.ip_network(whitelist_ip):
                    return True
        return False
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        if self.is_whitelisted(ip):
            return False
            
        if ip in self.blocked_ips:
            block_time = self.blocked_ips[ip]
            if datetime.now() - block_time > timedelta(seconds=self.block_duration):
                del self.blocked_ips[ip]
                self.violations[ip] = 0
                self._save_blocks()
                return False
            return True
        return False
    
    def check_rate_limit(self, ip: str, endpoint: str) -> bool:
        """Check if request is within rate limit."""
        if self.is_whitelisted(ip):
            return True
            
        if self.is_blocked(ip):
            return False
        
        # Clean old requests
        now = datetime.now()
        self.requests[ip] = [
            req_time for req_time in self.requests[ip]
            if now - req_time <= timedelta(minutes=1)
        ]
        
        # Check rate limit
        if len(self.requests[ip]) >= self.default_rate_limit:
            self.violations[ip] += 1
            if self.violations[ip] >= self.block_threshold:
                self.blocked_ips[ip] = now
                self._save_blocks()
                logger.warning(f"IP {ip} blocked due to rate limit violations")
            return False
        
        # Add request
        self.requests[ip].append(now)
        return True
    
    def _load_blocks(self):
        """Load blocked IPs from file."""
        blocks_file = self.config_dir / 'ip_blocks.json'
        if blocks_file.exists():
            with open(blocks_file, 'r') as f:
                data = json.load(f)
                self.blocked_ips = {
                    ip: datetime.fromisoformat(time)
                    for ip, time in data.items()
                }
    
    def _save_blocks(self):
        """Save blocked IPs to file."""
        blocks_file = self.config_dir / 'ip_blocks.json'
        with open(blocks_file, 'w') as f:
            json.dump(
                {
                    ip: time.isoformat()
                    for ip, time in self.blocked_ips.items()
                },
                f,
                indent=2
            )
    
    def _cleanup_thread(self):
        """Periodically clean up expired blocks and requests."""
        while True:
            now = datetime.now()
            
            # Clean expired blocks
            expired_blocks = [
                ip for ip, block_time in self.blocked_ips.items()
                if now - block_time > timedelta(seconds=self.block_duration)
            ]
            for ip in expired_blocks:
                del self.blocked_ips[ip]
                self.violations[ip] = 0
            
            if expired_blocks:
                self._save_blocks()
            
            # Clean old requests
            for ip in list(self.requests.keys()):
                self.requests[ip] = [
                    req_time for req_time in self.requests[ip]
                    if now - req_time <= timedelta(minutes=1)
                ]
                if not self.requests[ip]:
                    del self.requests[ip]
            
            # Sleep for a minute
            time.sleep(60)
    
    def _start_cleanup_thread(self):
        """Start the cleanup thread."""
        thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        thread.start()
