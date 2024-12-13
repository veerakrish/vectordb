# Mistral VectorDB

A high-performance vector database with HNSW indexing, designed specifically for RAG (Retrieval Augmented Generation) applications.

## Features

- **HNSW Indexing**: Fast and efficient similarity search
- **Persistence**: Automatic save and load functionality
- **ACID Compliance**: Transaction logging and recovery
- **Backup & Restore**: Built-in backup and restore capabilities
- **Metadata Support**: Store and retrieve metadata alongside vectors
- **Batch Operations**: Efficient batch insert and search operations
- **Type Safety**: Full type hints support

## Installation

```bash
pip install mistral-vectordb
```

## Quick Start

```python
from mistral_vectordb import VectorStore

# Initialize vector store
store = VectorStore(
    storage_dir="vector_data",
    dim=1024,
    distance_metric="cosine"
)

# Add vectors
store.add_item(
    id="doc1",
    vector=[0.1, 0.2, ...],  # Your vector here
    metadata={"text": "Document content", "source": "file1.txt"}
)

# Search vectors
results = store.search(
    query_vector=[0.3, 0.4, ...],  # Your query vector
    top_k=5
)

# Access results
for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")

# Backup and restore
store.backup("backup_20231213")
store.restore("backup_20231213")
```

## Advanced Usage

### Batch Operations

```python
# Batch insert
vectors = [
    ([0.1, 0.2, ...], "doc1", {"text": "Content 1"}),
    ([0.3, 0.4, ...], "doc2", {"text": "Content 2"}),
]
store.batch_add(vectors)

# Batch search
query_vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
results = store.batch_search(query_vectors, top_k=5)
```

### Custom Distance Metrics

```python
store = VectorStore(
    storage_dir="vector_data",
    dim=1024,
    distance_metric="inner_product"  # or "euclidean", "cosine"
)
```

### Transaction Logging

```python
# All operations are automatically logged
with store.transaction() as txn:
    txn.add_item("doc1", vector, metadata)
    txn.delete_item("doc2")
```

## Backup Management

The vector store includes comprehensive backup management features:

### Creating Backups

```python
# Create a backup with automatic naming (timestamp + files)
store.backup()

# Create a backup in a specific directory
store.backup(backup_dir="/path/to/backup")
```

### Listing Backups

```python
# Get list of all available backups
backups = store.list_backups()

# View backup information
for backup in backups:
    print(f"Backup Time: {backup['timestamp']}")
    print(f"Files: {backup['files']}")
    print(f"Vectors: {backup['num_vectors']}")
```

### Loading Specific Files

```python
# Load specific files from a backup
store.load_specific_files(
    backup_dir="/path/to/backup",
    files_to_load=["document1.pdf", "document2.pdf"]
)
```

### Full Restore

```python
# Restore entire vector store from backup
store.restore(backup_dir="/path/to/backup")
```

Each backup contains:
- `vector_store.pkl`: The vector store data
- `metadata.json`: File metadata and hashes
- `transaction.log`: Transaction history
- `backup_info.json`: Backup metadata

Backups are stored in timestamped directories:
```
backup_20241213_1200_document1_document2/
```

Features:
- Incremental backups
- File-level granularity
- Detailed backup metadata
- Safe restore operations
- Backup verification

## API Reference

### VectorStore

```python
class VectorStore:
    def __init__(
        self,
        storage_dir: str,
        dim: int,
        distance_metric: str = "cosine",
        ef_construction: int = 200,
        M: int = 16
    ):
        """Initialize vector store.
        
        Args:
            storage_dir: Directory for storing vectors and metadata
            dim: Dimension of vectors
            distance_metric: Distance metric ("cosine", "euclidean", "inner_product")
            ef_construction: HNSW index construction parameter
            M: HNSW index parameter
        """
        ...

    def add_item(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict] = None
    ) -> None:
        """Add single item to store."""
        ...

    def batch_add(
        self,
        items: List[Tuple[List[float], str, Optional[Dict]]]
    ) -> None:
        """Add multiple items in batch."""
        ...

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        ...

    def batch_search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10
    ) -> List[List[SearchResult]]:
        """Search for multiple query vectors in batch."""
        ...

    def backup(self, backup_dir: str) -> None:
        """Create backup of vector store."""
        ...

    def restore(self, backup_dir: str) -> None:
        """Restore vector store from backup."""
        ...

    def list_backups(self) -> List[Dict]:
        """Get list of all available backups."""
        ...

    def load_specific_files(
        self,
        backup_dir: str,
        files_to_load: List[str]
    ) -> None:
        """Load specific files from a backup."""
        ...
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
