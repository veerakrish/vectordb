"""Tests for VectorStore."""

import numpy as np
import pytest
from pathlib import Path
import shutil

from mistral_vectordb import VectorStore

@pytest.fixture
def vector_store(tmp_path):
    """Create a temporary vector store for testing."""
    store = VectorStore(
        storage_dir=str(tmp_path / "test_vectors"),
        dim=128,
        distance_metric="cosine"
    )
    yield store
    shutil.rmtree(tmp_path)

def test_add_item(vector_store):
    """Test adding a single item."""
    vector = np.random.rand(128)
    metadata = {"text": "test document"}
    
    vector_store.add_item("test1", vector, metadata)
    
    # Verify item was added
    results = vector_store.search(vector, top_k=1)
    assert len(results) == 1
    assert results[0].id == "test1"
    assert results[0].metadata == metadata
    assert results[0].score < 1e-6  # Should be very close to 0 for same vector

def test_batch_add(vector_store):
    """Test batch adding items."""
    vectors = np.random.rand(3, 128)
    items = [
        (vector, f"batch_{i}", {"text": f"doc {i}"})
        for i, vector in enumerate(vectors)
    ]
    
    vector_store.batch_add(items)
    
    # Verify all items were added
    for i, vector in enumerate(vectors):
        results = vector_store.search(vector, top_k=1)
        assert results[0].id == f"batch_{i}"
        assert results[0].metadata == {"text": f"doc {i}"}

def test_search(vector_store):
    """Test vector search."""
    # Add some vectors
    vectors = np.random.rand(5, 128)
    for i, vector in enumerate(vectors):
        vector_store.add_item(f"doc_{i}", vector)
    
    # Test search
    query = vectors[0]
    results = vector_store.search(query, top_k=3)
    
    assert len(results) == 3
    assert results[0].id == "doc_0"  # First result should be the same vector
    assert results[0].score < 1e-6

def test_batch_search(vector_store):
    """Test batch search."""
    # Add vectors
    vectors = np.random.rand(5, 128)
    for i, vector in enumerate(vectors):
        vector_store.add_item(f"doc_{i}", vector)
    
    # Test batch search
    queries = vectors[:2]
    results = vector_store.batch_search(queries, top_k=2)
    
    assert len(results) == 2
    assert len(results[0]) == 2
    assert results[0][0].id == "doc_0"
    assert results[1][0].id == "doc_1"

def test_delete_item(vector_store):
    """Test deleting items."""
    vector = np.random.rand(128)
    vector_store.add_item("test1", vector)
    
    vector_store.delete_item("test1")
    
    # Verify item was deleted
    with pytest.raises(KeyError):
        vector_store.get_vector("test1")

def test_backup_restore(vector_store, tmp_path):
    """Test backup and restore functionality."""
    # Add some data
    vector = np.random.rand(128)
    metadata = {"text": "test document"}
    vector_store.add_item("test1", vector, metadata)
    
    # Create backup
    backup_dir = tmp_path / "backup"
    vector_store.backup(str(backup_dir))
    
    # Create new store and restore
    new_store = VectorStore(
        storage_dir=str(tmp_path / "restored"),
        dim=128
    )
    new_store.restore(str(backup_dir))
    
    # Verify data was restored
    results = new_store.search(vector, top_k=1)
    assert results[0].id == "test1"
    assert results[0].metadata == metadata

def test_transaction(vector_store):
    """Test transaction functionality."""
    vector = np.random.rand(128)
    
    # Test successful transaction
    with vector_store.transaction() as txn:
        vector_store.add_item("test1", vector)
    
    assert "test1" in vector_store.id_to_index
    
    # Test failed transaction
    with pytest.raises(ValueError):
        with vector_store.transaction():
            vector_store.add_item("test1", vector)  # Should fail - duplicate ID
