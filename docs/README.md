# Mistral VectorDB Documentation

A high-performance vector database with secure storage, advanced document processing, and RAG features.

## Features

- **Secure Storage**: B+ tree-based storage with encryption support
- **Advanced Document Processing**: Support for multiple document formats with semantic chunking
- **Hybrid Search**: Combine dense and sparse retrieval methods
- **Configurable**: Extensive configuration options for all components
- **Modular**: Easy to extend and maintain

## Installation

```bash
# Basic installation
pip install mistral-vectordb

# With GPU support
pip install mistral-vectordb[gpu]

# With development tools
pip install mistral-vectordb[dev]
```

## Quick Start

```python
from mistral_vectordb import Config, VectorStore
from mistral_vectordb.document_processor import DocumentProcessor

# Initialize configuration
config = Config()

# Initialize components
processor = DocumentProcessor()
store = VectorStore(
    storage_dir=config.get_storage_path("my_vectors"),
    dim=1024
)

# Process and store a document
doc_info = processor.process_document("document.pdf")
for chunk in doc_info['chunks']:
    store.add_item(
        id=f"doc1_chunk_{chunk['index']}",
        vector=get_embedding(chunk['text']),  # Your embedding function
        metadata={'text': chunk['text']}
    )

# Search
results = store.search(
    query_vector=query_embedding,
    k=3
)
```

## Module Structure

- **config.py**: Configuration management
- **document_processor.py**: Document processing and chunking
- **storage/**: Storage implementations
  - **btree.py**: B+ tree implementation
- **search/**: Search implementations
  - **hybrid.py**: Hybrid search methods
  - **reranking.py**: Re-ranking implementations
- **security/**: Security features
  - **encryption.py**: Encryption utilities
  - **access.py**: Access control

## Storage Location

All data is stored under the `~/.mistral_vectordb` directory by default:

```
~/.mistral_vectordb/
├── indices/          # Vector indices
├── documents/        # Processed documents
└── backups/          # Backup files
```

## Configuration

Create a custom configuration file:

```json
{
    "storage": {
        "base_dir": "~/.mistral_vectordb",
        "max_file_size": 104857600
    },
    "processing": {
        "chunk_size": 500,
        "chunk_overlap": 0.1,
        "enable_ocr": true
    },
    "security": {
        "enable_encryption": true
    }
}
```

Load custom configuration:

```python
config = Config("config.json")
```

## Examples

See the `examples/` directory for complete examples:

- `document_processing_example.py`: Document processing
- `secure_storage_example.py`: Secure storage operations
- `rag_search_example.py`: RAG search functionality

## Security

- Data encryption using Fernet
- Secure key derivation with PBKDF2
- File integrity verification
- Access control
- Size and type validation

## Performance Tips

1. **Indexing**:
   - Adjust `ef_construction` and `M` parameters
   - Use batch operations for bulk insertions

2. **Search**:
   - Enable caching for repeated queries
   - Use hybrid search with appropriate weights
   - Adjust MMR parameters for diversity

3. **Storage**:
   - Configure appropriate chunk sizes
   - Use SSD storage for better performance
   - Regular maintenance and optimization

## Debugging

Each module includes detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

See CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file.
