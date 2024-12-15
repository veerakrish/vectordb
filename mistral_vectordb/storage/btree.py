"""B+ Tree implementation for efficient vector storage and retrieval."""

from typing import Any, List, Optional, Tuple, Dict
import pickle
import os
from pathlib import Path
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class BPlusNode:
    def __init__(self, leaf: bool = False):
        self.leaf = leaf
        self.keys: List[str] = []
        self.children: List[Any] = []  # BPlusNode for internal nodes, data for leaves
        self.next: Optional['BPlusNode'] = None  # For leaf nodes
        self.parent: Optional['BPlusNode'] = None

class BPlusTree:
    def __init__(self, order: int = 4, storage_path: Path = None, encryption_key: Optional[str] = None):
        self.root = BPlusNode(leaf=True)
        self.order = order
        self.storage_path = storage_path
        self._lock = threading.RLock()
        
        # Initialize encryption if key provided
        self.fernet = None
        if encryption_key:
            self.fernet = self._setup_encryption(encryption_key)
    
    def _setup_encryption(self, key: str) -> Fernet:
        """Set up encryption using provided key."""
        salt = b'mistral_vectordb_salt'  # In production, use a secure random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        return Fernet(key)

    def insert(self, key: str, value: Any) -> None:
        """Insert a key-value pair into the B+ tree."""
        with self._lock:
            if self.fernet:
                value = self.fernet.encrypt(pickle.dumps(value))
            
            node = self._find_leaf(key)
            
            # Insert into leaf node
            insert_idx = 0
            while insert_idx < len(node.keys) and node.keys[insert_idx] < key:
                insert_idx += 1
            
            node.keys.insert(insert_idx, key)
            node.children.insert(insert_idx, value)
            
            # Split if necessary
            if len(node.keys) >= self.order:
                self._split_node(node)
            
            # Persist changes
            self._persist_changes()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        with self._lock:
            node = self._find_leaf(key)
            try:
                idx = node.keys.index(key)
                value = node.children[idx]
                
                if self.fernet:
                    value = pickle.loads(self.fernet.decrypt(value))
                return value
            except ValueError:
                return None

    def range_query(self, start_key: str, end_key: str) -> List[Tuple[str, Any]]:
        """Retrieve all key-value pairs within a range."""
        with self._lock:
            results = []
            node = self._find_leaf(start_key)
            
            while node:
                for i, key in enumerate(node.keys):
                    if start_key <= key <= end_key:
                        value = node.children[i]
                        if self.fernet:
                            value = pickle.loads(self.fernet.decrypt(value))
                        results.append((key, value))
                    elif key > end_key:
                        return results
                node = node.next
            
            return results

    def _find_leaf(self, key: str) -> BPlusNode:
        """Find the leaf node where the key should be located."""
        node = self.root
        while not node.leaf:
            idx = 0
            while idx < len(node.keys) and key >= node.keys[idx]:
                idx += 1
            node = node.children[idx]
        return node

    def _split_node(self, node: BPlusNode) -> None:
        """Split a node when it exceeds the order."""
        mid = self.order // 2
        
        new_node = BPlusNode(leaf=node.leaf)
        new_node.keys = node.keys[mid:]
        new_node.children = node.children[mid:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid]
        
        if node.leaf:
            new_node.next = node.next
            node.next = new_node
        
        if node == self.root:
            new_root = BPlusNode(leaf=False)
            new_root.keys = [new_node.keys[0]]
            new_root.children = [node, new_node]
            self.root = new_root
            node.parent = new_root
            new_node.parent = new_root
        else:
            parent = node.parent
            insert_idx = parent.children.index(node) + 1
            parent.keys.insert(insert_idx - 1, new_node.keys[0])
            parent.children.insert(insert_idx, new_node)
            new_node.parent = parent
            
            if len(parent.keys) >= self.order:
                self._split_node(parent)

    def _persist_changes(self) -> None:
        """Persist the B+ tree to disk."""
        if self.storage_path:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, storage_path: Path, encryption_key: Optional[str] = None) -> 'BPlusTree':
        """Load a B+ tree from disk."""
        with open(storage_path, 'rb') as f:
            tree = pickle.load(f)
            if encryption_key:
                tree.fernet = tree._setup_encryption(encryption_key)
            return tree
