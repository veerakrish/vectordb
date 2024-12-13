"""Example demonstrating RAG with backup management in Mistral VectorDB."""

import numpy as np
from pathlib import Path
from typing import List, Dict
from mistral_vectordb import VectorStore

# Simulated embeddings function (replace with your actual embedding model)
def get_embeddings(texts: List[str]) -> np.ndarray:
    """Simulate text embeddings (replace with actual embedding model)."""
    return np.random.rand(len(texts), 128)  # 128-dim embeddings

class DocumentProcessor:
    def __init__(self, storage_dir: str):
        self.store = VectorStore(
            storage_dir=storage_dir,
            dim=128,
            distance_metric="cosine"
        )
        self.chunk_size = 500  # characters per chunk
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 for space
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_document(self, filename: str, content: str):
        """Process a document and add to vector store."""
        # Split into chunks
        chunks = self.chunk_text(content)
        
        # Get embeddings
        embeddings = get_embeddings(chunks)
        
        # Add to vector store
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.store.add_item(
                id=f"{filename}_{i}",
                vector=embedding,
                metadata={
                    "filename": filename,
                    "chunk_id": i,
                    "content": chunk,
                    "total_chunks": len(chunks)
                }
            )
    
    def create_backup(self, backup_dir: str = None):
        """Create a backup of the current state."""
        self.store.backup(backup_dir)
    
    def list_available_documents(self) -> List[str]:
        """List all available documents."""
        backups = self.store.list_backups()
        if not backups:
            return []
        
        # Get most recent backup
        latest_backup = backups[0]
        return latest_backup['files']
    
    def load_documents(self, backup_dir: str, filenames: List[str]):
        """Load specific documents from a backup."""
        self.store.load_specific_files(backup_dir, filenames)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant chunks."""
        # Get query embedding
        query_embedding = get_embeddings([query])[0]
        
        # Search vector store
        results = self.store.search(query_embedding, k=k)
        
        return [{
            'filename': r.metadata['filename'],
            'content': r.metadata['content'],
            'score': r.score
        } for r in results]

def main():
    # Initialize processor
    processor = DocumentProcessor("rag_demo_store")
    
    print("1. Processing sample documents...")
    # Sample documents
    documents = {
        "physics.txt": """
        Quantum mechanics is a fundamental theory in physics that provides a description 
        of the physical properties of nature at the scale of atoms and subatomic particles.
        It describes the behavior of matter and its interactions with energy on the scale 
        of atomic and subatomic particles.
        """,
        "chemistry.txt": """
        Chemical bonding is a lasting attraction between atoms, ions or molecules that 
        enables the formation of chemical compounds. The bond may result from the 
        electrostatic force between oppositely charged ions or through the sharing 
        of electrons.
        """
    }
    
    # Process documents
    for filename, content in documents.items():
        processor.process_document(filename, content)
    
    print("\n2. Creating first backup...")
    processor.create_backup()
    
    print("\n3. Adding new document...")
    # Add new document
    processor.process_document(
        "biology.txt",
        """
        DNA, or deoxyribonucleic acid, is the hereditary material in humans and 
        almost all other organisms. Nearly every cell in a person's body has the 
        same DNA. Most DNA is located in the cell nucleus.
        """
    )
    
    print("\n4. Creating second backup...")
    processor.create_backup()
    
    print("\n5. Available documents in latest backup:")
    available_docs = processor.list_available_documents()
    for doc in available_docs:
        print(f"- {doc}")
    
    print("\n6. Loading specific documents from first backup...")
    # Get first backup
    backups = processor.store.list_backups()
    first_backup = backups[-1]  # Last is oldest
    
    # Load only physics and chemistry
    processor.load_documents(
        first_backup['backup_dir'],
        ["physics.txt", "chemistry.txt"]
    )
    
    print("\n7. Searching in loaded documents...")
    # Try some searches
    queries = [
        "How do atoms bond together?",
        "What is quantum mechanics about?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = processor.search(query, k=2)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Document: {result['filename']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content']}")

if __name__ == "__main__":
    main()
