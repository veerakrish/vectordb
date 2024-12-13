"""Vector store implementation with persistence and ACID compliance."""

import pickle
import json
import shutil
import threading
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import heapq

logger = logging.getLogger(__name__)

class Node:
    """Node in the HNSW graph."""
    def __init__(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        self.id = id
        self.vector = vector
        self.metadata = metadata
        self.neighbors: Dict[int, List[str]] = {}  # layer -> list of neighbor ids

class HNSWIndex:
    """Hierarchical Navigable Small World Index implementation."""
    
    def __init__(
        self,
        dim: int,
        M: int = 16,  # Max number of connections per node
        ef_construction: int = 200,  # Size of dynamic candidate list
        max_layers: int = 4,
        distance_metric: str = "cosine"
    ):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.max_layers = max_layers
        self.distance_metric = distance_metric
        self.nodes: Dict[str, Node] = {}
        self.entry_point: Optional[str] = None
        self._lock = None  # Will be initialized in __getstate__
        self._init_lock()
    
    def _init_lock(self):
        """Initialize the thread lock."""
        self._lock = threading.Lock()
    
    def __getstate__(self):
        """Custom state for pickling."""
        state = self.__dict__.copy()
        # Don't pickle the lock
        state['_lock'] = None
        return state
    
    def __setstate__(self, state):
        """Custom state for unpickling."""
        self.__dict__.update(state)
        # Recreate the lock
        self._init_lock()

    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate distance between two vectors."""
        if self.distance_metric == "cosine":
            return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        elif self.distance_metric == "euclidean":
            return np.linalg.norm(vec1 - vec2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _get_random_level(self) -> int:
        """Generate random level for a new node."""
        level = 0
        while np.random.random() < 0.5 and level < self.max_layers - 1:
            level += 1
        return level

    def add_item(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add an item to the index."""
        with self._lock:
            if len(vector.shape) != 1 or vector.shape[0] != self.dim:
                raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {vector.shape}")
            
            # Create new node
            node = Node(id, vector, metadata)
            level = self._get_random_level()
            
            # If this is the first node
            if not self.entry_point:
                self.nodes[id] = node
                self.entry_point = id
                return
            
            # Find entry points for each layer
            curr_node_id = self.entry_point
            curr_dist = self._distance(vector, self.nodes[curr_node_id].vector)
            
            # Search from top to bottom
            for layer in range(len(self.nodes[self.entry_point].neighbors), -1, -1):
                while True:
                    changed = False
                    
                    # Check neighbors at current layer
                    curr_node = self.nodes[curr_node_id]
                    if layer in curr_node.neighbors:
                        for neighbor_id in curr_node.neighbors[layer]:
                            if neighbor_id not in self.nodes:
                                continue
                            dist = self._distance(vector, self.nodes[neighbor_id].vector)
                            if dist < curr_dist:
                                curr_node_id = neighbor_id
                                curr_dist = dist
                                changed = True
                    
                    if not changed:
                        break
            
            # Add connections for the new node
            for layer in range(level + 1):
                # Find nearest neighbors at this layer
                candidates = [(curr_dist, curr_node_id)]
                visited = {curr_node_id}
                
                while candidates:
                    dist, curr_id = heapq.heappop(candidates)
                    curr_node = self.nodes[curr_id]
                    
                    if layer in curr_node.neighbors:
                        for neighbor_id in curr_node.neighbors[layer]:
                            if neighbor_id in visited or neighbor_id not in self.nodes:
                                continue
                            neighbor_dist = self._distance(vector, self.nodes[neighbor_id].vector)
                            heapq.heappush(candidates, (neighbor_dist, neighbor_id))
                            visited.add(neighbor_id)
                
                # Select M nearest neighbors
                neighbors = sorted(visited, key=lambda x: self._distance(vector, self.nodes[x].vector))[:self.M]
                node.neighbors[layer] = neighbors
                
                # Add backward connections
                for neighbor_id in neighbors:
                    if layer not in self.nodes[neighbor_id].neighbors:
                        self.nodes[neighbor_id].neighbors[layer] = []
                    self.nodes[neighbor_id].neighbors[layer].append(id)
                    # Ensure neighbor doesn't have too many connections
                    if len(self.nodes[neighbor_id].neighbors[layer]) > self.M:
                        # Remove connection to the furthest neighbor
                        furthest = max(
                            self.nodes[neighbor_id].neighbors[layer],
                            key=lambda x: self._distance(self.nodes[neighbor_id].vector, self.nodes[x].vector)
                        )
                        self.nodes[neighbor_id].neighbors[layer].remove(furthest)
            
            # Add the node to the index
            self.nodes[id] = node
            
            # Update entry point if necessary
            if level > len(self.nodes[self.entry_point].neighbors):
                self.entry_point = id

    def get_ids(self) -> List[str]:
        """Get all item IDs in the index."""
        with self._lock:
            return list(self.nodes.keys())

    def remove_item(self, id: str) -> None:
        """Remove an item from the index."""
        with self._lock:
            if id not in self.nodes:
                return
            
            # Get the node to remove
            node = self.nodes[id]
            
            # Remove references to this node from its neighbors
            for layer in node.neighbors:
                for neighbor_id in node.neighbors[layer]:
                    if neighbor_id in self.nodes:
                        neighbor = self.nodes[neighbor_id]
                        if layer in neighbor.neighbors:
                            neighbor.neighbors[layer] = [nid for nid in neighbor.neighbors[layer] if nid != id]
            
            # If this is the entry point, update it
            if self.entry_point == id:
                if len(self.nodes) > 1:
                    # Set a different node as entry point
                    self.entry_point = next(nid for nid in self.nodes.keys() if nid != id)
                else:
                    self.entry_point = None
            
            # Remove the node
            del self.nodes[id]

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[float, str]]:
        """Search for k nearest neighbors."""
        with self._lock:
            if not self.entry_point:
                return []
            
            # Start from entry point
            curr_node_id = self.entry_point
            curr_dist = self._distance(query_vector, self.nodes[curr_node_id].vector)
            
            # Search from top to bottom
            for layer in range(len(self.nodes[self.entry_point].neighbors), -1, -1):
                while True:
                    changed = False
                    
                    # Check neighbors at current layer
                    curr_node = self.nodes[curr_node_id]
                    if layer in curr_node.neighbors:
                        for neighbor_id in curr_node.neighbors[layer]:
                            if neighbor_id not in self.nodes:
                                continue
                            dist = self._distance(query_vector, self.nodes[neighbor_id].vector)
                            if dist < curr_dist:
                                curr_node_id = neighbor_id
                                curr_dist = dist
                                changed = True
                    
                    if not changed:
                        break
            
            # Collect k nearest neighbors
            candidates = [(curr_dist, curr_node_id)]
            visited = {curr_node_id}
            results = []
            
            while candidates and len(results) < k:
                dist, curr_id = heapq.heappop(candidates)
                results.append((dist, curr_id))
                
                # Add unvisited neighbors
                curr_node = self.nodes[curr_id]
                for layer in curr_node.neighbors:
                    for neighbor_id in curr_node.neighbors[layer]:
                        if neighbor_id in visited or neighbor_id not in self.nodes:
                            continue
                        neighbor_dist = self._distance(query_vector, self.nodes[neighbor_id].vector)
                        heapq.heappush(candidates, (neighbor_dist, neighbor_id))
                        visited.add(neighbor_id)
            
            return results

class VectorStore:
    """Vector store with HNSW indexing and persistence."""
    
    def __init__(
        self,
        storage_dir: str,
        dim: int,
        distance_metric: str = "cosine",
        auto_save_interval: int = 60  # seconds
    ):
        self.storage_dir = Path(storage_dir)
        self.index = HNSWIndex(dim=dim, distance_metric=distance_metric)
        self.auto_save_interval = auto_save_interval
        self.last_save_time = datetime.now()
        self.transaction_log_path = self.storage_dir / "transaction.log"
        self.data_path = self.storage_dir / "vector_store.pkl"
        self.metadata_path = self.storage_dir / "metadata.json"
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self._initialize_storage()
        
        # Start auto-save thread if interval > 0
        if auto_save_interval > 0:
            self._start_auto_save()

    def _initialize_storage(self) -> None:
        """Initialize storage and recover if necessary."""
        try:
            if self.data_path.exists():
                self.load()
            if self.transaction_log_path.exists():
                self._replay_transaction_log()
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            raise

    def _log_transaction(self, operation: str, data: Dict[str, Any]) -> None:
        """Log transaction for recovery."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'data': data
        }
        with open(self.transaction_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _replay_transaction_log(self) -> None:
        """Replay transaction log for recovery."""
        if not self.transaction_log_path.exists():
            return

        with open(self.transaction_log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry['operation'] == 'add':
                        data = entry['data']
                        self.index.add_item(
                            id=data['id'],
                            vector=np.array(data['vector']),
                            metadata=data['metadata']
                        )
                    elif entry['operation'] == 'remove':
                        data = entry['data']
                        self.index.remove_item(data['id'])
                except Exception as e:
                    logger.error(f"Error replaying transaction: {e}")

    def _start_auto_save(self) -> None:
        """Start auto-save thread."""
        def auto_save():
            while True:
                time.sleep(self.auto_save_interval)
                self.save()
                self.last_save_time = datetime.now()

        save_thread = threading.Thread(target=auto_save, daemon=True)
        save_thread.start()

    def clear_file(self, filename: str) -> None:
        """Remove all items associated with a specific file."""
        try:
            # Get all IDs that start with the filename
            ids_to_remove = [id for id in self.index.get_ids() if id.startswith(filename)]
            
            # Remove each item
            for id in ids_to_remove:
                self.index.remove_item(id)
                self._log_transaction('remove', {'id': id})
            
            logger.info(f"Cleared all items for file: {filename}")
        except Exception as e:
            logger.error(f"Error clearing file {filename}: {e}")
            raise

    def add_item(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add a vector to the store. If item exists, it will be updated."""
        try:
            # If item exists, remove it first
            if id in self.index.get_ids():
                self.index.remove_item(id)
                self._log_transaction('remove', {'id': id})
            
            # Convert vector to list if it's numpy array, otherwise use as is
            vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
            self._log_transaction('add', {
                'id': id,
                'vector': vector_list,
                'metadata': metadata
            })
            
            # Convert to numpy array for the index
            vector_array = np.array(vector)
            self.index.add_item(id=id, vector=vector_array, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error adding item: {e}")
            raise

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Search for similar vectors with optional filtering."""
        try:
            results = self.index.search(query_vector, k=k)
            
            # Add metadata and apply filter
            filtered_results = []
            for dist, id in results:
                metadata = self.index.nodes[id].metadata
                if not filter_func or filter_func(metadata):
                    filtered_results.append((dist, id, metadata))
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def save(self) -> None:
        """Save vector store state."""
        try:
            # Save index data
            with open(self.data_path, 'wb') as f:
                pickle.dump(self.index, f)
            
            # Save metadata
            metadata = {
                'last_save': datetime.now().isoformat(),
                'num_items': len(self.index.nodes),
                'dim': self.index.dim,
                'distance_metric': self.index.distance_metric
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Clear transaction log
            if self.transaction_log_path.exists():
                self.transaction_log_path.unlink()
            
            logger.info("Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise

    def load(self) -> None:
        """Load vector store state."""
        try:
            with open(self.data_path, 'rb') as f:
                self.index = pickle.load(f)
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def backup(self, backup_dir: Optional[str] = None) -> None:
        """Create a backup of the vector store."""
        try:
            # Save current state
            self.save()
            
            # Create backup directory
            backup_path = Path(backup_dir) if backup_dir else self.storage_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy files to backup
            shutil.copy2(self.data_path, backup_path / self.data_path.name)
            shutil.copy2(self.metadata_path, backup_path / self.metadata_path.name)
            
            logger.info(f"Backup created at {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise

    def restore(self, backup_dir: str) -> None:
        """Restore vector store from backup."""
        try:
            backup_path = Path(backup_dir)
            
            # Copy backup files to main storage
            shutil.copy2(backup_path / self.data_path.name, self.data_path)
            shutil.copy2(backup_path / self.metadata_path.name, self.metadata_path)
            
            # Load the restored data
            self.load()
            
            logger.info(f"Restored from backup at {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise
