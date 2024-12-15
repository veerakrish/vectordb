"""Document processing module for efficient text extraction and preprocessing."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime
import mimetypes
import logging
import threading

import spacy
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import docx
import pandas as pd
from tika import parser
import magic
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(
        self,
        language_model: str = "en_core_web_sm",
        enable_ocr: bool = True,
        chunk_size: int = 500,
        chunk_overlap: float = 0.1,
        max_file_size: int = 100 * 1024 * 1024  # 100MB
    ):
        """Initialize document processor.
        
        Args:
            language_model: Spacy language model to use
            enable_ocr: Whether to enable OCR for images
            chunk_size: Default chunk size in characters
            chunk_overlap: Overlap between chunks as fraction
            max_file_size: Maximum file size in bytes
        """
        self.nlp = spacy.load(language_model)
        self.enable_ocr = enable_ocr
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size = max_file_size
        self._lock = threading.Lock()
        
        # Initialize supported formats
        self.supported_formats = {
            'text': ['.txt', '.md', '.rst'],
            'pdf': ['.pdf'],
            'word': ['.doc', '.docx'],
            'spreadsheet': ['.csv', '.xls', '.xlsx'],
            'image': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
            'html': ['.html', '.htm'],
        }

    def process_document(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document and return its content and metadata."""
        try:
            with self._lock:
                # Check file size
                if os.path.getsize(file_path) > self.max_file_size:
                    raise ValueError(f"File size exceeds maximum limit of {self.max_file_size} bytes")
                
                # Get file type
                mime_type = magic.from_file(str(file_path), mime=True)
                ext = file_path.suffix.lower()
                
                # Extract text based on file type
                if ext in self.supported_formats['text']:
                    text = self._process_text_file(file_path)
                elif ext in self.supported_formats['pdf']:
                    text = self._process_pdf(file_path)
                elif ext in self.supported_formats['word']:
                    text = self._process_word(file_path)
                elif ext in self.supported_formats['spreadsheet']:
                    text = self._process_spreadsheet(file_path)
                elif ext in self.supported_formats['image'] and self.enable_ocr:
                    text = self._process_image(file_path)
                elif ext in self.supported_formats['html']:
                    text = self._process_html(file_path)
                else:
                    # Try Tika as fallback
                    text = self._process_with_tika(file_path)
                
                # Generate document metadata
                doc_metadata = {
                    'filename': file_path.name,
                    'file_type': mime_type,
                    'file_size': os.path.getsize(file_path),
                    'created_at': datetime.fromtimestamp(os.path.getctime(file_path)),
                    'modified_at': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'checksum': self._calculate_checksum(file_path),
                    'language': self._detect_language(text),
                    'processed_at': datetime.now().isoformat(),
                }
                
                if metadata:
                    doc_metadata.update(metadata)
                
                # Chunk the text
                chunks = self._chunk_text(text)
                
                return {
                    'content': text,
                    'chunks': chunks,
                    'metadata': doc_metadata
                }
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise

    def _process_text_file(self, file_path: Path) -> str:
        """Process plain text files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _process_pdf(self, file_path: Path) -> str:
        """Process PDF files."""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def _process_word(self, file_path: Path) -> str:
        """Process Word documents."""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _process_spreadsheet(self, file_path: Path) -> str:
        """Process spreadsheet files."""
        df = pd.read_excel(file_path) if file_path.suffix in ['.xls', '.xlsx'] else pd.read_csv(file_path)
        return df.to_string()

    def _process_image(self, file_path: Path) -> str:
        """Process images using OCR."""
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    def _process_html(self, file_path: Path) -> str:
        """Process HTML files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()

    def _process_with_tika(self, file_path: Path) -> str:
        """Process file using Apache Tika."""
        parsed = parser.from_file(str(file_path))
        return parsed["content"] if parsed["content"] else ""

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text into semantic units."""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_size = len(sent_text)
            
            if current_size + sent_size > self.chunk_size:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'size': current_size,
                    'sentences': len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_size = int(self.chunk_size * self.chunk_overlap)
                overlap_text = chunk_text[-overlap_size:] if overlap_size > 0 else ""
                current_chunk = [overlap_text, sent_text] if overlap_text else [sent_text]
                current_size = len(overlap_text) + sent_size
            else:
                current_chunk.append(sent_text)
                current_size += sent_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'size': current_size,
                'sentences': len(current_chunk)
            })
        
        return chunks

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _detect_language(self, text: str) -> str:
        """Detect document language."""
        doc = self.nlp(text[:1000])  # Use first 1000 chars for efficiency
        return doc.lang_
