"""Embedding providers for Mistral VectorDB."""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from mistralai.client import MistralClient
from mistralai.models.chat import ChatMessage
import torch

class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def __init__(self, **kwargs):
        """Initialize embedding provider."""
        pass
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for texts."""
        raise NotImplementedError

class MistralEmbedding(EmbeddingProvider):
    """Mistral AI embedding provider."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "mistral-embed",
        **kwargs
    ):
        """Initialize Mistral embedding provider.
        
        Args:
            api_key: Mistral API key
            model: Model name to use
        """
        super().__init__(**kwargs)
        self.client = MistralClient(api_key=api_key)
        self.model = model
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Mistral AI."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            response = self.client.embeddings(
                model=self.model,
                input=[text]
            )
            embeddings.append(response.data[0].embedding)
        
        return np.array(embeddings)

class SentenceTransformerEmbedding(EmbeddingProvider):
    """Sentence Transformer embedding provider."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize Sentence Transformer provider.
        
        Args:
            model_name: Model name to use
            device: Device to use (cpu/cuda)
        """
        super().__init__(**kwargs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Sentence Transformers."""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        **kwargs
    ):
        """Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        super().__init__(**kwargs)
        import openai
        openai.api_key = api_key
        self.model = model
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI."""
        import openai
        
        if isinstance(texts, str):
            texts = [texts]
        
        response = openai.Embedding.create(
            input=texts,
            model=self.model
        )
        
        embeddings = [data['embedding'] for data in response['data']]
        return np.array(embeddings)

class HuggingFaceEmbedding(EmbeddingProvider):
    """HuggingFace embedding provider."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize HuggingFace provider.
        
        Args:
            model_name: Model name to use
            device: Device to use (cpu/cuda)
        """
        super().__init__(**kwargs)
        from transformers import AutoTokenizer, AutoModel
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using HuggingFace model."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        
        return embeddings
