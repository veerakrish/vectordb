"""Setup script for Mistral VectorDB."""

from setuptools import setup, find_packages
import os
from pathlib import Path
import sys
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

def _post_install():
    """Post-installation setup."""
    try:
        from mistral_vectordb.setup_tool import setup_mistral_vectordb
        
        # Check if API key is in environment
        api_key = os.environ.get('MISTRAL_API_KEY')
        
        # Run setup
        setup_mistral_vectordb(api_key)
        
    except Exception as e:
        print(f"\nNote: Initial setup failed: {e}")
        print("You can run setup later using: python -m mistral_vectordb.setup_tool")

class PostInstallCommand(install):
    """Post-installation setup command."""
    def run(self):
        install.run(self)
        self.execute(_post_install, [], msg="Running post-installation setup...")

class PostDevelopCommand(develop):
    """Post-development setup command."""
    def run(self):
        develop.run(self)
        self.execute(_post_install, [], msg="Running post-installation setup...")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mistral-vectordb",
    version="0.5.0",  # Updated version for document processing and security features
    description="High-performance vector database with secure storage, advanced document processing, and RAG features",
    author="viswanath veera krishna maddinala",
    author_email="veerukhannan@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/veerakrish/mistral-vectordb",
    packages=find_packages(),
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    install_requires=[
        "numpy>=1.19.0",
        "requests>=2.25.0",
        "tqdm>=4.65.0",
        "python-dotenv>=0.19.0",
        "streamlit>=1.24.0",
        "hnswlib>=0.7.0",
        "spacy>=3.0.0",
        "sentence-transformers>=2.2.0",
        "rank_bm25>=0.2.2",
        "scikit-learn>=1.0.0",
        "cachetools>=5.0.0",
        "mistralai>=0.0.7",
        # API Server
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-jose[cryptography]>=3.3.0",
        "python-multipart>=0.0.5",
        # Rate limiting and monitoring
        "slowapi>=0.1.4",
        "prometheus-client>=0.12.0",
        "psutil>=5.8.0",
        "authlib>=1.0.0",
        "starlette>=0.14.2",
        # Document processing
        "PyMuPDF>=1.18.0",  # PDF processing
        "python-docx>=0.8.11",  # Word documents
        "beautifulsoup4>=4.9.3",  # HTML processing
        "pandas>=1.3.0",  # Spreadsheet processing
        "openpyxl>=3.0.7",  # Excel support
        "pytesseract>=0.3.8",  # OCR
        "tika>=1.24",  # General document parsing
        "python-magic>=0.4.24",  # File type detection
        "Pillow>=8.0.0",  # Image processing
        # Security
        "cryptography>=35.0.0",  # Encryption
        "bcrypt>=3.2.0",  # Password hashing
        "keyring>=24.0.0",  # For secure API key storage
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security :: Cryptography",
        "Topic :: Text Processing :: Indexing",
        "Operating System :: OS Independent",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
            "flake8>=4.0.0",
        ],
        "gpu": [
            "torch>=1.8.0",
            "tensorflow>=2.4.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'mistral-vectordb-setup=mistral_vectordb.setup_tool:setup_mistral_vectordb',
        ],
    },
)
