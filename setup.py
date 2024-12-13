"""Setup script for Mistral VectorDB."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mistral-vectordb",
    version="0.3.0",  # Updated version for backup management features
    description="High-performance vector database with HNSW indexing and backup management",
    author="Your Name",
    author_email="your.email@example.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mistral-vectordb",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "requests>=2.25.0",
        "tqdm>=4.65.0",
        "python-dotenv>=0.19.0",
        "streamlit>=1.24.0",
        "hnswlib>=0.7.0"
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
    },
)
