"""Main vector store implementation."""

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator

import hnswlib
import numpy as np
from tqdm import tqdm

from .types import SearchResult, Transaction, VectorMetadata

class VectorStore:
    """High-performance vector database with HNSW indexing."""

    def __init__(
        self,
        storage_dir: str,
        dim: int,
        distance_metric: str = "cosine",
        ef_construction: int = 200,
        M: int = 16,
        auto_save: bool = True
    ):
        """Initialize vector store.
        
        Args:
            storage_dir: Directory for storing vectors and metadata
            dim: Dimension of vectors
            distance_metric: Distance metric ("cosine", "euclidean", "inner_product")
            ef_construction: HNSW index construction parameter
            M: HNSW index parameter
            auto_save: Whether to automatically save after operations
        """
        self.storage_dir = Path(storage_dir)
        self.dim = dim
        self.distance_metric = distance_metric
        self.ef_construction = ef_construction
        self.M = M
        self.auto_save = auto_save
        
        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize HNSW index
        self.index = hnswlib.Index(space=distance_metric, dim=dim)
        self.index.init_index(
            max_elements=1000,  # Will be resized automatically
            ef_construction=ef_construction,
            M=M
        )
        
        # Initialize metadata storage
        self.metadata: Dict[str, Dict] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.current_index = 0
        
        # Load existing data if available
        self._load_if_exists()
    
    def add_item(
        self,
        id: str,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict] = None
    ) -> None:
        """Add single item to store.
        
        Args:
            id: Unique identifier for the vector
            vector: Vector to store
            metadata: Optional metadata to store with vector
        """
        if isinstance(vector, list):
            vector = np.array(vector)
        
        if id in self.id_to_index:
            raise ValueError(f"Item with id {id} already exists")
        
        # Add to index
        self.index.add_items(vector.reshape(1, -1), [self.current_index])
        
        # Update mappings
        self.id_to_index[id] = self.current_index
        self.index_to_id[self.current_index] = id
        if metadata:
            self.metadata[id] = metadata
        
        self.current_index += 1
        
        if self.auto_save:
            self.save()
    
    def batch_add(
        self,
        items: List[Tuple[Union[List[float], np.ndarray], str, Optional[Dict]]]
    ) -> None:
        """Add multiple items in batch.
        
        Args:
            items: List of (vector, id, metadata) tuples
        """
        vectors = []
        indices = []
        
        for vector, id, metadata in items:
            if id in self.id_to_index:
                raise ValueError(f"Item with id {id} already exists")
            
            if isinstance(vector, list):
                vector = np.array(vector)
            
            vectors.append(vector)
            indices.append(self.current_index)
            
            # Update mappings
            self.id_to_index[id] = self.current_index
            self.index_to_id[self.current_index] = id
            if metadata:
                self.metadata[id] = metadata
            
            self.current_index += 1
        
        # Add to index
        self.index.add_items(
            np.array(vectors),
            np.array(indices)
        )
        
        if self.auto_save:
            self.save()
    
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        include_vectors: bool = False
    ) -> List[SearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            include_vectors: Whether to include vectors in results
            
        Returns:
            List of SearchResult objects
        """
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
        
        # Search index
        scores, indices = self.index.knn_query(query_vector.reshape(1, -1), k=top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            id = self.index_to_id[idx]
            metadata = self.metadata.get(id)
            
            result = SearchResult(
                id=id,
                score=float(score),
                metadata=metadata
            )
            
            if include_vectors:
                result.vector = self.get_vector(id).tolist()
            
            results.append(result)
        
        return results
    
    def batch_search(
        self,
        query_vectors: List[Union[List[float], np.ndarray]],
        top_k: int = 10,
        include_vectors: bool = False
    ) -> List[List[SearchResult]]:
        """Search for multiple query vectors in batch.
        
        Args:
            query_vectors: List of query vectors
            top_k: Number of results to return per query
            include_vectors: Whether to include vectors in results
            
        Returns:
            List of lists of SearchResult objects
        """
        # Convert to numpy arrays
        query_vectors = [
            np.array(v) if isinstance(v, list) else v
            for v in query_vectors
        ]
        query_array = np.array(query_vectors)
        
        # Search index
        scores, indices = self.index.knn_query(query_array, k=top_k)
        
        # Format results
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                id = self.index_to_id[idx]
                metadata = self.metadata.get(id)
                
                result = SearchResult(
                    id=id,
                    score=float(score),
                    metadata=metadata
                )
                
                if include_vectors:
                    result.vector = self.get_vector(id).tolist()
                
                results.append(result)
            all_results.append(results)
        
        return all_results
    
    def get_vector(self, id: str) -> np.ndarray:
        """Get vector by id.
        
        Args:
            id: Vector identifier
            
        Returns:
            Vector as numpy array
        """
        if id not in self.id_to_index:
            raise KeyError(f"No vector found with id {id}")
        
        idx = self.id_to_index[id]
        vector = self.index.get_items([idx])
        return vector[0]
    
    def delete_item(self, id: str) -> None:
        """Delete item from store.
        
        Args:
            id: Item identifier
        """
        if id not in self.id_to_index:
            raise KeyError(f"No item found with id {id}")
        
        idx = self.id_to_index[id]
        
        # Mark as deleted in index
        self.index.mark_deleted(idx)
        
        # Remove from mappings
        del self.id_to_index[id]
        del self.index_to_id[idx]
        if id in self.metadata:
            del self.metadata[id]
        
        if self.auto_save:
            self.save()
    
    def save(self) -> None:
        """Save vector store to disk."""
        # Save index
        self.index.save_index(str(self.storage_dir / "hnsw.index"))
        
        # Save metadata and mappings
        with open(self.storage_dir / "metadata.json", "w") as f:
            json.dump({
                "metadata": self.metadata,
                "id_to_index": self.id_to_index,
                "index_to_id": {str(k): v for k, v in self.index_to_id.items()},
                "current_index": self.current_index,
                "dim": self.dim,
                "distance_metric": self.distance_metric,
                "ef_construction": self.ef_construction,
                "M": self.M
            }, f)
    
    def _load_if_exists(self) -> None:
        """Load vector store from disk if it exists."""
        index_path = self.storage_dir / "hnsw.index"
        metadata_path = self.storage_dir / "metadata.json"
        
        if not (index_path.exists() and metadata_path.exists()):
            return
        
        # Load metadata and mappings
        with open(metadata_path) as f:
            data = json.load(f)
            self.metadata = data["metadata"]
            self.id_to_index = {k: int(v) for k, v in data["id_to_index"].items()}
            self.index_to_id = {int(k): v for k, v in data["index_to_id"].items()}
            self.current_index = data["current_index"]
        
        # Load index
        self.index.load_index(str(index_path))
    
    def backup(self, backup_dir: str) -> None:
        """Create backup of vector store.
        
        Args:
            backup_dir: Directory to store backup
        """
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Save current state
        self.save()
        
        # Copy files to backup directory
        shutil.copytree(
            self.storage_dir,
            backup_path / self.storage_dir.name,
            dirs_exist_ok=True
        )
    
    def restore(self, backup_dir: str) -> None:
        """Restore vector store from backup.
        
        Args:
            backup_dir: Directory containing backup
        """
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            raise ValueError(f"Backup directory {backup_dir} does not exist")
        
        # Copy files from backup
        shutil.rmtree(self.storage_dir, ignore_errors=True)
        shutil.copytree(
            backup_path / self.storage_dir.name,
            self.storage_dir,
            dirs_exist_ok=True
        )
        
        # Reload data
        self._load_if_exists()
    
    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """Create transaction for batch operations.
        
        Usage:
            with store.transaction() as txn:
                txn.add_item("id1", vector1, metadata1)
                txn.delete_item("id2")
        """
        txn = Transaction(operations=[], timestamp=time.time(), status="in_progress")
        
        try:
            yield txn
            txn.status = "committed"
            if self.auto_save:
                self.save()
        except Exception:
            txn.status = "rolled_back"
            raise
