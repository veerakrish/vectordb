"""Example demonstrating backup management features in Mistral VectorDB."""

import numpy as np
from pathlib import Path
from mistral_vectordb import VectorStore

def create_sample_vectors(num_vectors: int, dim: int) -> list:
    """Create sample vectors for demonstration."""
    return [np.random.rand(dim) for _ in range(num_vectors)]

def main():
    # Initialize vector store
    store = VectorStore(
        storage_dir="demo_vector_store",
        dim=128,
        distance_metric="cosine"
    )
    
    print("1. Adding sample documents...")
    # Add some sample documents
    documents = {
        "document1.pdf": "This is the first document",
        "document2.pdf": "Second document with different content",
        "document3.pdf": "Third document for testing"
    }
    
    vectors = create_sample_vectors(len(documents), 128)
    for (doc_name, content), vector in zip(documents.items(), vectors):
        store.add_item(
            id=f"{doc_name}_1",
            vector=vector,
            metadata={"filename": doc_name, "content": content}
        )
    
    print("\n2. Creating first backup...")
    # Create a backup
    store.backup()
    
    print("\n3. Adding more documents...")
    # Add more documents
    new_documents = {
        "document4.pdf": "Fourth document added later",
        "document5.pdf": "Fifth document in second batch"
    }
    
    new_vectors = create_sample_vectors(len(new_documents), 128)
    for (doc_name, content), vector in zip(new_documents.items(), vectors):
        store.add_item(
            id=f"{doc_name}_1",
            vector=vector,
            metadata={"filename": doc_name, "content": content}
        )
    
    print("\n4. Creating second backup...")
    # Create another backup
    store.backup()
    
    print("\n5. Listing all backups...")
    # List all backups
    backups = store.list_backups()
    for i, backup in enumerate(backups, 1):
        print(f"\nBackup {i}:")
        print(f"- Time: {backup['timestamp']}")
        print(f"- Files: {', '.join(backup['files'])}")
        print(f"- Vectors: {backup['num_vectors']}")
    
    print("\n6. Loading specific files from first backup...")
    # Load specific files from first backup
    first_backup = backups[-1]  # Last in list is oldest
    store.load_specific_files(
        backup_dir=first_backup['backup_dir'],
        files_to_load=["document1.pdf", "document2.pdf"]
    )
    
    print("\n7. Searching with loaded files...")
    # Try a search with the loaded files
    query_vector = np.random.rand(128)  # Random query vector
    results = store.search(query_vector, k=2)
    
    print("\nSearch results:")
    for result in results:
        print(f"- Found: {result.metadata['filename']}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Content: {result.metadata['content']}")

if __name__ == "__main__":
    main()
