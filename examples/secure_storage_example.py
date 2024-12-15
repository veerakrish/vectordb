"""Example of secure storage with Mistral VectorDB."""

from pathlib import Path
import numpy as np
from mistral_vectordb import Config
from mistral_vectordb.storage.btree import BPlusTree

def secure_storage_example():
    # Initialize configuration
    config = Config()
    
    # Create a secure B+ tree with encryption
    secure_tree = BPlusTree(
        order=4,
        storage_path=config.get_storage_path("secure_index") / "tree.db",
        encryption_key="your-secure-key-here"
    )
    
    # Store some sample vectors
    print("Storing sample vectors...")
    for i in range(5):
        # Create sample vector
        vector = np.random.rand(128)
        
        # Create metadata
        metadata = {
            "id": f"doc_{i}",
            "description": f"Sample vector {i}",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # Store in B+ tree
        secure_tree.insert(
            key=f"vector_{i}",
            value={
                "vector": vector.tolist(),
                "metadata": metadata
            }
        )
    
    # Retrieve vectors
    print("\nRetrieving vectors...")
    for i in range(5):
        key = f"vector_{i}"
        result = secure_tree.get(key)
        if result:
            print(f"\nRetrieved {key}:")
            print(f"Metadata: {result['metadata']}")
            print(f"Vector shape: {len(result['vector'])}")
    
    # Perform range query
    print("\nPerforming range query...")
    range_results = secure_tree.range_query("vector_1", "vector_3")
    print(f"Found {len(range_results)} vectors in range")
    for key, value in range_results:
        print(f"Key: {key}, ID: {value['metadata']['id']}")

if __name__ == "__main__":
    secure_storage_example()
