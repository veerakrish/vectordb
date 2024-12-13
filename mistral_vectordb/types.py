"""Type definitions for Mistral VectorDB."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TypeVar, Generic

T = TypeVar('T')

@dataclass
class VectorMetadata:
    """Metadata associated with a vector."""
    id: str
    data: Dict[str, Any]

@dataclass
class SearchResult:
    """Result from a vector search operation."""
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None

@dataclass
class Transaction(Generic[T]):
    """Transaction for vector store operations."""
    operations: List[T]
    timestamp: float
    status: str  # "committed", "rolled_back", "in_progress"
