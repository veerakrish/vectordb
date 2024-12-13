"""Type definitions for vector store."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class Node:
    """Node in the HNSW graph."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    neighbors: Dict[int, List[str]]  # layer -> list of neighbor ids

@dataclass
class SearchResult:
    """Search result from vector store."""
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
