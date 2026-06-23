"""
Memory Retrieval System

Implements a lightweight vector-based knowledge store with cosine similarity search.
Retrieves relevant agricultural facts to augment model inference.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional


class MemoryStore:
    """Vector-based knowledge store for retrieval-augmented generation."""
    
    def __init__(self, embedding_dim: int = 256):
        """
        Initialize memory store.
        
        Args:
            embedding_dim: Dimension of fact embeddings.
        """
        self.embedding_dim = embedding_dim
        self.facts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
    
    def add_fact(self, fact: str, source: str = "unknown"):
        """
        Add a fact to memory store.
        
        Args:
            fact: Agricultural fact or knowledge piece.
            source: Source of the fact.
        """
        self.facts.append(fact)
        self.metadata.append({"source": source})
    
    def add_facts(self, facts: List[str], source: str = "unknown"):
        """
        Add multiple facts to memory store.
        
        Args:
            facts: List of facts to add.
            source: Source of all facts.
        """
        for fact in facts:
            self.add_fact(fact, source)
    
    def embed_fact(self, fact: str) -> np.ndarray:
        """
        Generate embedding for a fact (placeholder).
        
        In production, this would use a real embedding model (BERT, etc.).
        For now, returns a random embedding for testing.
        
        Args:
            fact: Fact text.
        
        Returns:
            Embedding vector.
        """
        # TODO: Replace with real embedding model
        np.random.seed(hash(fact) % (2**32))
        return np.random.randn(self.embedding_dim)
    
    def build_embeddings(self):
        """Build embeddings for all stored facts."""
        if not self.facts:
            self.embeddings = np.array([])
            return
        
        embeddings = []
        for fact in self.facts:
            embedding = self.embed_fact(fact)
            embeddings.append(embedding)
        
        self.embeddings = np.array(embeddings)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
        
        Returns:
            Cosine similarity score.
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Retrieve top-k most similar facts to query.
        
        Args:
            query: Query text.
            top_k: Number of facts to retrieve.
        
        Returns:
            List of (fact, similarity, source) tuples.
        """
        if self.embeddings is None or len(self.facts) == 0:
            return []
        
        # Embed query
        query_embedding = self.embed_fact(query)
        
        # Compute similarities
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, embedding)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for idx, sim in similarities[:top_k]:
            fact = self.facts[idx]
            source = self.metadata[idx]["source"]
            results.append((fact, sim, source))
        
        return results
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Get formatted context string from retrieved facts.
        
        Args:
            query: Query text.
            top_k: Number of facts to retrieve.
        
        Returns:
            Formatted context string.
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return "No relevant knowledge found."
        
        context = "Retrieved knowledge:\n"
        for i, (fact, sim, source) in enumerate(results, 1):
            context += f"{i}. {fact} (confidence: {sim:.2f}, source: {source})\n"
        
        return context
    
    def save(self, filepath: str):
        """Save memory store to file."""
        data = {
            "embedding_dim": self.embedding_dim,
            "facts": self.facts,
            "metadata": self.metadata,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load memory store from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.embedding_dim = data["embedding_dim"]
        self.facts = data["facts"]
        self.metadata = data["metadata"]
        
        # Rebuild embeddings
        self.build_embeddings()
