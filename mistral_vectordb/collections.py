"""Collections management for Mistral VectorDB."""

from typing import Dict, List, Optional, Union, Any
import uuid
from datetime import datetime
import json
from pathlib import Path
import threading
import numpy as np

from .storage.btree import BPlusTree
from .document_processor import DocumentProcessor
from .config import Config

class Collection:
    """A collection of documents and their embeddings."""
    
    def __init__(
        self,
        name: str,
        config: Config,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[callable] = None
    ):
        """Initialize collection.
        
        Args:
            name: Collection name
            config: Configuration object
            metadata: Optional collection metadata
            embedding_function: Function to generate embeddings
        """
        self.name = name
        self.config = config
        self.metadata = metadata or {}
        self._embedding_function = embedding_function
        self._lock = threading.RLock()
        
        # Initialize storage
        self.storage_path = config.get_storage_path(f"collections/{name}")
        self.tree = BPlusTree(
            order=4,
            storage_path=self.storage_path / "vectors.db",
            encryption_key=config.config['security'].get('encryption_key')
        )
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor(
            language_model=config.config['processing']['language_model'],
            enable_ocr=config.config['processing']['enable_ocr'],
            chunk_size=config.config['processing']['chunk_size'],
            chunk_overlap=config.config['processing']['chunk_overlap']
        )
        
        # Save collection metadata
        self._save_metadata()

    def add(
        self,
        documents: Union[str, List[str]],
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        embeddings: Optional[Union[List[float], List[List[float]]]] = None
    ) -> List[str]:
        """Add documents to collection.
        
        Args:
            documents: Document(s) to add
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of document IDs
        """
        with self._lock:
            # Normalize inputs
            if isinstance(documents, str):
                documents = [documents]
            if isinstance(metadatas, dict):
                metadatas = [metadatas]
            if isinstance(ids, str):
                ids = [ids]
            
            # Generate IDs if not provided
            if not ids:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Process documents
            processed_docs = []
            for i, doc in enumerate(documents):
                doc_info = self.doc_processor.process_document(
                    Path(doc) if isinstance(doc, str) else doc,
                    metadata=metadatas[i] if metadatas else None
                )
                processed_docs.append(doc_info)
            
            # Generate or use provided embeddings
            if not embeddings:
                if not self._embedding_function:
                    raise ValueError("No embedding function provided")
                embeddings = []
                for doc in processed_docs:
                    doc_embeddings = []
                    for chunk in doc['chunks']:
                        embedding = self._embedding_function(chunk['text'])
                        doc_embeddings.append(embedding)
                    embeddings.append(doc_embeddings)
            
            # Store documents and embeddings
            for i, doc in enumerate(processed_docs):
                doc_embeddings = embeddings[i] if isinstance(embeddings[i], list) else [embeddings[i]]
                for j, chunk in enumerate(doc['chunks']):
                    chunk_id = f"{ids[i]}_chunk_{j}"
                    self.tree.insert(
                        chunk_id,
                        {
                            'embedding': doc_embeddings[j],
                            'text': chunk['text'],
                            'metadata': {
                                **doc['metadata'],
                                'chunk_index': j,
                                'parent_id': ids[i]
                            }
                        }
                    )
            
            return ids

    def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get documents from collection.
        
        Args:
            ids: Optional specific IDs to retrieve
            where: Optional filter conditions
            limit: Optional maximum number of results
            offset: Optional offset for pagination
            
        Returns:
            Dictionary with documents, metadatas, and embeddings
        """
        with self._lock:
            results = {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'embeddings': []
            }
            
            # Get by IDs
            if ids:
                if isinstance(ids, str):
                    ids = [ids]
                for id in ids:
                    data = self.tree.get(id)
                    if data:
                        results['ids'].append(id)
                        results['documents'].append(data['text'])
                        results['metadatas'].append(data['metadata'])
                        results['embeddings'].append(data['embedding'])
            
            # Get by filter
            elif where:
                def match_filter(metadata):
                    for key, value in where.items():
                        if key not in metadata or metadata[key] != value:
                            return False
                    return True
                
                # Scan through tree (can be optimized with proper indexing)
                matches = []
                current = self.tree.root
                while current.leaf:
                    for i, key in enumerate(current.keys):
                        data = current.children[i]
                        if match_filter(data['metadata']):
                            matches.append((key, data))
                    current = current.next
                
                # Apply pagination
                if offset:
                    matches = matches[offset:]
                if limit:
                    matches = matches[:limit]
                
                # Format results
                for key, data in matches:
                    results['ids'].append(key)
                    results['documents'].append(data['text'])
                    results['metadatas'].append(data['metadata'])
                    results['embeddings'].append(data['embedding'])
            
            return results

    def query(
        self,
        query_texts: Optional[Union[str, List[str]]] = None,
        query_embeddings: Optional[Union[List[float], List[List[float]]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query collection for similar documents.
        
        Args:
            query_texts: Text queries
            query_embeddings: Vector queries
            n_results: Number of results per query
            where: Optional filter conditions
            
        Returns:
            Dictionary with results
        """
        with self._lock:
            # Generate embeddings if needed
            if query_texts and not query_embeddings:
                if not self._embedding_function:
                    raise ValueError("No embedding function provided")
                if isinstance(query_texts, str):
                    query_texts = [query_texts]
                query_embeddings = [self._embedding_function(text) for text in query_texts]
            
            if isinstance(query_embeddings, list) and not isinstance(query_embeddings[0], list):
                query_embeddings = [query_embeddings]
            
            results = {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': []
            }
            
            for query_embedding in query_embeddings:
                # Search vectors
                matches = []
                current = self.tree.root
                while current.leaf:
                    for i, key in enumerate(current.keys):
                        data = current.children[i]
                        # Calculate distance
                        distance = np.linalg.norm(
                            np.array(query_embedding) - np.array(data['embedding'])
                        )
                        # Apply filter if provided
                        if not where or all(
                            data['metadata'].get(k) == v for k, v in where.items()
                        ):
                            matches.append((distance, key, data))
                    current = current.next
                
                # Sort by distance and get top results
                matches.sort(key=lambda x: x[0])
                matches = matches[:n_results]
                
                # Format results
                query_results = {
                    'ids': [],
                    'documents': [],
                    'metadatas': [],
                    'distances': []
                }
                
                for distance, key, data in matches:
                    query_results['ids'].append(key)
                    query_results['documents'].append(data['text'])
                    query_results['metadatas'].append(data['metadata'])
                    query_results['distances'].append(float(distance))
                
                # Add to overall results
                for key in results:
                    results[key].append(query_results[key])
            
            return results

    def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Delete documents from collection.
        
        Args:
            ids: Optional specific IDs to delete
            where: Optional filter conditions
            
        Returns:
            List of deleted IDs
        """
        with self._lock:
            deleted_ids = []
            
            if ids:
                if isinstance(ids, str):
                    ids = [ids]
                for id in ids:
                    if self.tree.delete(id):
                        deleted_ids.append(id)
            
            elif where:
                def match_filter(metadata):
                    return all(
                        metadata.get(k) == v for k, v in where.items()
                    )
                
                # Find matching documents
                to_delete = []
                current = self.tree.root
                while current.leaf:
                    for i, key in enumerate(current.keys):
                        data = current.children[i]
                        if match_filter(data['metadata']):
                            to_delete.append(key)
                    current = current.next
                
                # Delete matches
                for key in to_delete:
                    if self.tree.delete(key):
                        deleted_ids.append(key)
            
            return deleted_ids

    def _save_metadata(self) -> None:
        """Save collection metadata."""
        metadata_path = self.storage_path / "metadata.json"
        metadata = {
            'name': self.name,
            'created_at': datetime.now().isoformat(),
            'metadata': self.metadata
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
