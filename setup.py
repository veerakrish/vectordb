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
    version="1.0.0",  # Major version release with security features
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
        'numpy>=1.19.0',
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'python-jose[cryptography]>=3.3.0',
        'python-multipart>=0.0.5',
        'requests>=2.26.0',
        'pydantic>=1.8.0',
        'PyYAML>=5.4.1',
        'cryptography>=3.4.7',
        'python-dateutil>=2.8.2',
        'aiohttp>=3.8.0',
        'prometheus-client>=0.12.0',
        'psutil>=5.8.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'black>=21.9b0',
            'isort>=5.9.3',
            'flake8>=3.9.2',
            'mypy>=0.910',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Database',
        'Topic :: Security',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'vectordb-server=mistral_vectordb.server.api:run_server',
        ],
    },
)
