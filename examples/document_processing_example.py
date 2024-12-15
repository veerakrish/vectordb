"""Example of document processing with Mistral VectorDB."""

from pathlib import Path
from mistral_vectordb import Config
from mistral_vectordb.document_processor import DocumentProcessor

def process_documents_example():
    # Initialize configuration
    config = Config()
    
    # Initialize document processor with custom settings
    processor = DocumentProcessor(
        language_model="en_core_web_sm",
        enable_ocr=True,
        chunk_size=300,
        chunk_overlap=0.1
    )
    
    # Process different types of documents
    documents = [
        "sample.pdf",
        "document.docx",
        "spreadsheet.xlsx",
        "webpage.html",
        "image.png"
    ]
    
    for doc_name in documents:
        try:
            # Process document
            doc_info = processor.process_document(
                file_path=Path(doc_name),
                metadata={
                    "category": "samples",
                    "tags": ["example", "documentation"]
                }
            )
            
            # Print document information
            print(f"\nProcessed {doc_name}:")
            print(f"File type: {doc_info['metadata']['file_type']}")
            print(f"Size: {doc_info['metadata']['file_size']} bytes")
            print(f"Language: {doc_info['metadata']['language']}")
            print(f"Number of chunks: {len(doc_info['chunks'])}")
            
            # Print first chunk
            if doc_info['chunks']:
                first_chunk = doc_info['chunks'][0]
                print("\nFirst chunk preview:")
                print(f"Size: {first_chunk['size']} characters")
                print(f"Text: {first_chunk['text'][:200]}...")
        
        except Exception as e:
            print(f"Error processing {doc_name}: {e}")

if __name__ == "__main__":
    process_documents_example()
