"""Example of RAG search with Mistral VectorDB."""

from pathlib import Path
import numpy as np
from mistral_vectordb import Config, VectorStore
from mistral_vectordb.document_processor import DocumentProcessor

def rag_search_example():
    # Initialize configuration
    config = Config()
    
    # Initialize vector store with advanced features
    store = VectorStore(
        storage_dir=config.get_storage_path("rag_example"),
        dim=1024,
        enable_hybrid_search=True,
        enable_cache=True,
        reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    # Initialize document processor
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=0.1
    )
    
    # Process and store some sample documents
    documents = [
        "sample_doc1.pdf",
        "sample_doc2.txt",
        "sample_doc3.docx"
    ]
    
    print("Processing and storing documents...")
    for doc_path in documents:
        try:
            # Process document
            doc_info = processor.process_document(
                file_path=Path(doc_path),
                metadata={"source": doc_path}
            )
            
            # Store each chunk with its embedding
            for i, chunk in enumerate(doc_info['chunks']):
                # In real application, use proper embedding model
                mock_embedding = np.random.rand(1024)  # Mock embedding
                
                store.add_item(
                    id=f"{doc_path}_chunk_{i}",
                    vector=mock_embedding,
                    metadata={
                        **doc_info['metadata'],
                        'chunk_text': chunk['text'],
                        'chunk_index': i
                    }
                )
            print(f"Processed and stored {doc_path}")
    
    # Perform different types of searches
    print("\nPerforming searches...")
    
    # 1. Basic vector search
    print("\n1. Basic vector search:")
    query_vector = np.random.rand(1024)  # Mock query vector
    results = store.search(
        query_vector=query_vector,
        k=3
    )
    for score, id, metadata in results:
        print(f"Score: {score:.4f}, Document: {metadata['source']}")
    
    # 2. Hybrid search with text
    print("\n2. Hybrid search:")
    results = store.search(
        query_vector=query_vector,
        query_text="sample query text",
        k=3,
        hybrid_alpha=0.7
    )
    for score, id, metadata in results:
        print(f"Score: {score:.4f}, Document: {metadata['source']}")
    
    # 3. MMR search for diversity
    print("\n3. MMR search for diversity:")
    results = store.search(
        query_vector=query_vector,
        k=3,
        use_mmr=True,
        mmr_lambda=0.5
    )
    for score, id, metadata in results:
        print(f"Score: {score:.4f}, Document: {metadata['source']}")
    
    # 4. Search with filtering
    print("\n4. Filtered search:")
    def filter_func(metadata):
        return metadata['source'].endswith('.pdf')
    
    results = store.search(
        query_vector=query_vector,
        k=3,
        filter_func=filter_func
    )
    for score, id, metadata in results:
        print(f"Score: {score:.4f}, Document: {metadata['source']}")

if __name__ == "__main__":
    rag_search_example()
