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
from .security.crypto import SecureStorage, SecureDataProcessor

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

    def get_ids(self) -> List[str]:
        """Get all item IDs in the index."""
        with self._lock:
            return list(self.nodes.keys())

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
        auto_save_interval: int = 60,  # seconds
        enable_hybrid_search: bool = True,
        ef_construction: int = 200,
        M: int = 16,
        chunk_overlap: float = 0.1,
        enable_cache: bool = True,
        cache_size: int = 1000,
        reranking_model: Optional[str] = None,
        api_key: Optional[str] = None  # Mistral API key for security
    ):
        self.storage_dir = Path(storage_dir)
        
        # Initialize security components if API key provided
        if api_key:
            self.secure_storage = SecureStorage(api_key)
            self.data_processor = SecureDataProcessor(api_key)
        else:
            self.secure_storage = None
            self.data_processor = None
        
        self.index = HNSWIndex(
            dim=dim, 
            distance_metric=distance_metric,
            ef_construction=ef_construction,
            M=M
        )
        self.auto_save_interval = auto_save_interval
        self.last_save_time = datetime.now()
        self.transaction_log_path = self.storage_dir / "transaction.log"
        self.data_path = self.storage_dir / "vector_store.pkl"
        self.metadata_path = self.storage_dir / "metadata.json"
        
        # Hybrid search settings
        self.enable_hybrid_search = enable_hybrid_search
        if enable_hybrid_search:
            self.sparse_index = BM25Index()
        
        # Chunking settings
        self.chunk_overlap = chunk_overlap
        
        # Caching
        if enable_cache:
            self.cache = LRUCache(cache_size)
        else:
            self.cache = None
        
        # Reranking
        self.reranking_model = None
        if reranking_model:
            self.reranking_model = CrossEncoder(reranking_model)
        
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
            if self.data_path.exists() and self.data_path.stat().st_size > 0:
                self.load()
            if self.transaction_log_path.exists():
                self._replay_transaction_log()
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            # If there's an error loading, start with a fresh index
            self.index = HNSWIndex(dim=self.index.dim, distance_metric=self.index.distance_metric)

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
        # Validate and sanitize data if security is enabled
        if self.data_processor:
            metadata = self.data_processor.sanitize_input(metadata)
            self.data_processor.validate_schema(metadata, "vector_metadata")
            
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
        query_text: Optional[str] = None,
        k: int = 10,
        hybrid_alpha: float = 0.5,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        rerank: bool = True
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Search for similar vectors with hybrid search and re-ranking.
        
        Args:
            query_vector: Query embedding
            query_text: Original query text for sparse search
            k: Number of results to return
            hybrid_alpha: Weight for combining dense and sparse scores (0 = sparse only, 1 = dense only)
            filter_func: Optional function to filter results
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: MMR diversity parameter
            rerank: Whether to apply re-ranking
        """
        try:
            # Check cache first
            if self.cache is not None:
                cache_key = f"{hash(query_vector.tobytes())}{query_text}{k}{hybrid_alpha}"
                cached_results = self.cache.get(cache_key)
                if cached_results is not None:
                    return cached_results

            # Get dense search results
            dense_results = self.index.search(query_vector, k=k*2)  # Get more for hybrid
            
            final_results = []
            
            if self.enable_hybrid_search and query_text:
                # Get sparse search results
                sparse_results = self.sparse_index.search(query_text, k=k*2)
                
                # Combine results using RRF
                all_ids = set(id for _, id in dense_results) | set(id for _, id in sparse_results)
                combined_scores = {}
                
                for id in all_ids:
                    dense_rank = next((i for i, (_, doc_id) in enumerate(dense_results) if doc_id == id), len(dense_results))
                    sparse_rank = next((i for i, (_, doc_id) in enumerate(sparse_results) if doc_id == id), len(sparse_results))
                    
                    # RRF formula with hybrid weighting
                    dense_score = 1 / (dense_rank + 60) * hybrid_alpha
                    sparse_score = 1 / (sparse_rank + 60) * (1 - hybrid_alpha)
                    combined_scores[id] = dense_score + sparse_score
                
                # Sort by combined score
                sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                final_results = [(score, id, self.index.nodes[id].metadata) for id, score in sorted_results[:k]]
            else:
                final_results = [(dist, id, self.index.nodes[id].metadata) for dist, id in dense_results[:k]]
            
            # Apply MMR if requested
            if use_mmr:
                final_results = self._apply_mmr(query_vector, final_results, k, mmr_lambda)
            
            # Apply re-ranking if enabled
            if rerank and self.reranking_model and query_text:
                pairs = [(query_text, result[2].get('content', '')) for result in final_results]
                rerank_scores = self.reranking_model.predict(pairs)
                
                # Combine with original scores
                for i in range(len(final_results)):
                    orig_score = final_results[i][0]
                    rerank_score = rerank_scores[i]
                    final_results[i] = (0.5 * orig_score + 0.5 * rerank_score, *final_results[i][1:])
                
                # Re-sort based on combined scores
                final_results.sort(key=lambda x: x[0], reverse=True)
            
            # Apply filtering
            if filter_func:
                final_results = [r for r in final_results if filter_func(r[2])]
            
            # Cache results
            if self.cache is not None:
                self.cache.put(cache_key, final_results[:k])
            
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def chunk_text(
        self,
        text: str,
        chunk_type: str = "semantic",
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        overlap_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text using semantic or fixed-size chunking.
        
        Args:
            text: Text to chunk
            chunk_type: Type of chunking ("semantic" or "fixed")
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            overlap_size: Size of overlap between chunks (if None, uses self.chunk_overlap)
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if overlap_size is None:
            overlap_size = int(max_chunk_size * self.chunk_overlap)
        
        chunks = []
        
        if chunk_type == "semantic":
            # Use spacy for sentence segmentation
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            
            current_chunk = []
            current_size = 0
            
            # Group sentences into semantic chunks
            for sent in doc.sents:
                sent_text = sent.text.strip()
                sent_size = len(sent_text)
                
                # If adding this sentence exceeds max size, store current chunk
                if current_size + sent_size > max_chunk_size and current_size >= min_chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "size": current_size,
                        "sentences": len(current_chunk),
                        "start_char": len("".join(chunks)),
                        "end_char": len("".join(chunks)) + len(chunk_text)
                    })
                    
                    # Start new chunk with overlap
                    overlap_sents = []
                    overlap_size_current = 0
                    for prev_sent in reversed(current_chunk):
                        if overlap_size_current + len(prev_sent) > overlap_size:
                            break
                        overlap_sents.insert(0, prev_sent)
                        overlap_size_current += len(prev_sent)
                    
                    current_chunk = overlap_sents + [sent_text]
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk.append(sent_text)
                    current_size += sent_size
            
            # Add final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "size": current_size,
                    "sentences": len(current_chunk),
                    "start_char": len("".join(chunks)),
                    "end_char": len("".join(chunks)) + len(chunk_text)
                })
        
        else:  # fixed-size chunking
            text_size = len(text)
            start = 0
            
            while start < text_size:
                # Find end of current chunk
                end = start + max_chunk_size
                
                # Adjust end to not split words
                if end < text_size:
                    while end > start + min_chunk_size and not text[end].isspace():
                        end -= 1
                else:
                    end = text_size
                
                chunk_text = text[start:end].strip()
                chunks.append({
                    "text": chunk_text,
                    "size": len(chunk_text),
                    "start_char": start,
                    "end_char": end
                })
                
                # Move start position considering overlap
                start = end - overlap_size
        
        return chunks

    def save(self) -> None:
        """Save vector store state with encryption if enabled."""
        with self.index._lock:
            # Prepare data for saving
            data = {
                "index": self.index,
                "sparse_index": self.sparse_index if self.enable_hybrid_search else None,
                "metadata": {
                    "dim": self.index.dim,
                    "distance_metric": self.index.distance_metric,
                    "last_save": datetime.now().isoformat()
                }
            }
            
            # Use secure storage if available
            if self.secure_storage:
                self.secure_storage.store_encrypted(
                    self.data_path,
                    data
                )
            else:
                with open(self.data_path, "wb") as f:
                    pickle.dump(data, f)

    def load(self) -> None:
        """Load vector store state with decryption if enabled."""
        if not self.data_path.exists():
            logger.warning(f"No existing store found at {self.data_path}")
            return
            
        try:
            # Use secure storage if available
            if self.secure_storage:
                data = self.secure_storage.load_encrypted(self.data_path)
            else:
                with open(self.data_path, "rb") as f:
                    data = pickle.load(f)
            
            self.index = data["index"]
            if self.enable_hybrid_search:
                self.sparse_index = data["sparse_index"]
            
            # Update metadata
            self.index.dim = data["metadata"]["dim"]
            self.index.distance_metric = data["metadata"]["distance_metric"]
            
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading vector store: {e}")
            # If there's an error loading, start with a fresh index
            self.index = HNSWIndex(dim=self.index.dim, distance_metric=self.index.distance_metric)

    def backup(self, backup_dir: Optional[str] = None) -> None:
        """Create a backup of the vector store.
        
        This creates a complete backup including:
        - vector_store.pkl: The vector store data (all documents)
        - metadata.json: List of processed files and their hashes
        - transaction.log: Pending transactions
        - backup_info.json: Detailed backup information
        
        Each backup is stored in a new directory with format:
        backup_[timestamp]_[files]
        
        Example:
        backup_20241213_1200_document1_document2/
        
        Old backups are preserved, not overwritten.
        
        Args:
            backup_dir: Optional directory to store backup in. If None, uses storage_dir.
        """
        try:
            # First save current state to ensure metadata.json is up to date
            self.save()
            
            # Generate backup directory name with timestamp and file info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get list of unique file names (remove chunk numbers)
            processed_files = sorted(set(
                node_id.split('_')[0] 
                for node_id in self.index.nodes.keys()
            ))
            
            # Create a descriptive backup name
            file_info = "_".join(processed_files)
            if len(file_info) > 50:  # Truncate if too long
                file_info = f"{file_info[:47]}..."
            backup_name = f"backup_{timestamp}_{file_info}"
            
            # Use provided backup dir or create in storage dir
            if backup_dir:
                backup_path = Path(backup_dir) / backup_name
            else:
                backup_path = self.storage_dir / backup_name
            
            # Ensure we're not overwriting an existing backup
            if backup_path.exists():
                raise ValueError(f"Backup directory {backup_path} already exists")
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all essential files to backup
            files_to_backup = [
                (self.data_path, "vector_store.pkl"),
                (self.metadata_path, "metadata.json"),
                (self.transaction_log_path, "transaction.log")
            ]
            
            for src_path, backup_name in files_to_backup:
                if src_path.exists():
                    dst_path = backup_path / backup_name
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Backed up {src_path.name} ({src_path.stat().st_size} bytes)")
            
            # Save detailed backup information
            backup_info = {
                'backup_time': timestamp,
                'files_included': processed_files,
                'vector_store_size': self.data_path.stat().st_size if self.data_path.exists() else 0,
                'num_vectors': len(self.index.nodes),
                'processed_files': {
                    node_id: {
                        'file': node_id.split('_')[0],
                        'metadata': node.metadata,
                        'last_modified': datetime.now().isoformat()
                    }
                    for node_id, node in self.index.nodes.items()
                }
            }
            
            with open(backup_path / "backup_info.json", 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            logger.info(f"Created new backup at {backup_path}")
            logger.info(f"Backup contains {len(processed_files)} files:")
            for file in processed_files:
                logger.info(f"  - {file}")
            logger.info(f"Total vectors: {len(self.index.nodes)}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with their contents.
        
        Returns:
            List of dictionaries containing backup information:
            - backup_dir: Path to backup directory
            - timestamp: When backup was created
            - files: List of files in the backup
            - num_vectors: Number of vectors in the backup
        """
        backups = []
        backup_dirs = [d for d in self.storage_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('backup_')]
        
        for backup_dir in sorted(backup_dirs, reverse=True):  # Most recent first
            try:
                info_file = backup_dir / "backup_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                        backups.append({
                            'backup_dir': str(backup_dir),
                            'timestamp': info.get('backup_time', ''),
                            'files': info.get('files_included', []),
                            'num_vectors': info.get('num_vectors', 0),
                            'processed_files': info.get('processed_files', {})
                        })
            except Exception as e:
                logger.error(f"Error reading backup {backup_dir}: {e}")
                continue
        
        return backups

    def load_specific_files(self, backup_dir: str, files_to_load: List[str]) -> None:
        """Load specific files from a backup.
        
        Args:
            backup_dir: Path to backup directory
            files_to_load: List of file names to load
        """
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                raise ValueError(f"Backup directory {backup_dir} does not exist")
            
            # Load backup info
            with open(backup_path / "backup_info.json", 'r') as f:
                backup_info = json.load(f)
                available_files = backup_info.get('files_included', [])
                
                # Verify all requested files are in the backup
                missing_files = [f for f in files_to_load if f not in available_files]
                if missing_files:
                    raise ValueError(f"Files not in backup: {', '.join(missing_files)}")
            
            # Load the vector store data
            with open(backup_path / "vector_store.pkl", 'rb') as f:
                backup_index = pickle.load(f)
            
            # Create a new index with only the requested files
            new_index = HNSWIndex(dim=self.index.dim, distance_metric=self.index.distance_metric)
            loaded_files = set()
            
            # Copy only the vectors for requested files
            for node_id, node in backup_index.nodes.items():
                file_name = node_id.split('_')[0]  # Get base filename
                if file_name in files_to_load:
                    new_index.nodes[node_id] = node
                    loaded_files.add(file_name)
            
            # Update the current index
            self.index = new_index
            
            # Save the new state
            self.save()
            
            logger.info(f"Successfully loaded {len(loaded_files)} files from backup")
            for file in loaded_files:
                logger.info(f"  - {file}")
            
        except Exception as e:
            logger.error(f"Error loading files from backup: {e}")
            raise

    def restore(self, backup_dir: str) -> None:
        """Restore vector store from backup."""
        try:
            backup_path = Path(backup_dir)
            
            # Copy backup files to main storage
            shutil.copy2(backup_path / "vector_store.pkl", self.data_path)
            shutil.copy2(backup_path / "metadata.json", self.metadata_path)
            
            # Load the restored data
            self.load()
            
            logger.info(f"Restored from backup at {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise
