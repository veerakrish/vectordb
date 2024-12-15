# Mistral VectorDB Usage Guide

## Table of Contents

1. [Document Processing](#document-processing)
2. [Secure Storage](#secure-storage)
3. [RAG Search](#rag-search)
4. [Configuration](#configuration)
5. [Maintenance](#maintenance)

## Document Processing

### Basic Usage

```python
from mistral_vectordb.document_processor import DocumentProcessor

processor = DocumentProcessor(
    language_model="en_core_web_sm",
    enable_ocr=True
)

# Process a single document
doc_info = processor.process_document(
    file_path="document.pdf",
    metadata={"category": "research"}
)

# Access chunks
for chunk in doc_info['chunks']:
    print(f"Chunk size: {chunk['size']}")
    print(f"Text: {chunk['text'][:100]}...")
```

### Supported Formats

- PDF (`.pdf`)
- Word (`.doc`, `.docx`)
- Excel (`.xls`, `.xlsx`)
- Text (`.txt`, `.md`)
- HTML (`.html`, `.htm`)
- Images (`.png`, `.jpg`, `.jpeg`) with OCR

### Chunking Options

```python
processor = DocumentProcessor(
    chunk_size=500,        # Characters per chunk
    chunk_overlap=0.1,     # 10% overlap
    enable_ocr=True,       # Enable OCR for images
    language_model="en_core_web_sm"
)
```

## Secure Storage

### Using B+ Tree Storage

```python
from mistral_vectordb.storage.btree import BPlusTree

# Initialize secure storage
tree = BPlusTree(
    order=4,
    storage_path="~/.mistral_vectordb/indices/my_index",
    encryption_key="secure-key"
)

# Store data
tree.insert("key1", {"vector": [0.1, 0.2], "metadata": {...}})

# Retrieve data
data = tree.get("key1")

# Range query
results = tree.range_query("key1", "key5")
```

### Security Features

```python
# Enable encryption
tree = BPlusTree(
    storage_path="index.db",
    encryption_key="your-secure-key"
)

# Verify file integrity
checksum = tree._calculate_checksum()
```

## RAG Search

### Basic Search

```python
from mistral_vectordb import VectorStore

store = VectorStore(
    storage_dir="~/.mistral_vectordb/my_vectors",
    dim=1024
)

# Simple search
results = store.search(
    query_vector=query_embedding,
    k=5
)
```

### Advanced Search Features

```python
# Hybrid search
results = store.search(
    query_vector=query_embedding,
    query_text="original query",
    k=5,
    hybrid_alpha=0.7,
    use_mmr=True,
    mmr_lambda=0.5,
    rerank=True
)

# With filtering
def filter_func(metadata):
    return metadata['category'] == 'research'

results = store.search(
    query_vector=query_embedding,
    k=5,
    filter_func=filter_func
)
```

## Configuration

### Custom Configuration

```python
from mistral_vectordb import Config

# Load custom config
config = Config("config.json")

# Access configuration
base_dir = config.base_dir
index_dir = config.index_dir

# Get storage path
path = config.get_storage_path("my_collection")
```

### Configuration Options

```json
{
    "storage": {
        "base_dir": "~/.mistral_vectordb",
        "max_file_size": 104857600
    },
    "processing": {
        "chunk_size": 500,
        "enable_ocr": true
    },
    "indexing": {
        "vector_dim": 1024,
        "distance_metric": "cosine"
    }
}
```

## Maintenance

### Backup and Restore

```python
# Create backup
store.backup("backup_name")

# List backups
backups = store.list_backups()

# Restore from backup
store.restore("backup_name")
```

### Optimization

```python
# Optimize storage
store.optimize()

# Clear old data
store.clear_old_data(days=30)

# Verify integrity
store.verify_integrity()
```

### Monitoring

```python
# Get statistics
stats = store.get_statistics()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Storage size: {stats['storage_size']}")

# Check health
health = store.check_health()
print(f"Status: {health['status']}")
```

## Error Handling

```python
from mistral_vectordb.exceptions import (
    StorageError,
    ProcessingError,
    SecurityError
)

try:
    result = store.search(query_vector)
except StorageError as e:
    print(f"Storage error: {e}")
except ProcessingError as e:
    print(f"Processing error: {e}")
except SecurityError as e:
    print(f"Security error: {e}")
```

## Best Practices

1. **Document Processing**:
   - Use appropriate chunk sizes for your use case
   - Enable OCR only when needed
   - Process documents in batches for better performance

2. **Storage**:
   - Regularly backup important data
   - Monitor storage usage
   - Use encryption for sensitive data

3. **Search**:
   - Cache frequent queries
   - Use hybrid search for better results
   - Implement proper error handling

4. **Maintenance**:
   - Regular backups
   - Monitor system health
   - Optimize storage periodically
