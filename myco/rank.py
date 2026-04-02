# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Topological health proxy for myco.

Computes the numerical rank of the Gram matrix of token embeddings.
A collapsing rank (below 0.4) signals structural compression.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RankAnalysis:
    """Result of rank analysis."""
    numerical_rank: int
    normalized_rank: float
    sample_size: int
    is_collapsing: bool
    condition_number: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "numerical_rank": self.numerical_rank,
            "normalized_rank": self.normalized_rank,
            "sample_size": self.sample_size,
            "is_collapsing": self.is_collapsing,
            "condition_number": self.condition_number,
        }
    
    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        status = "⚠️  COLLAPSING" if self.is_collapsing else "✓ Healthy"
        return (
            f"Rank Analysis: {status}\n"
            f"  Numerical rank: {self.numerical_rank}\n"
            f"  Normalized rank: {self.normalized_rank:.3f}\n"
            f"  Sample size: {self.sample_size}\n"
            f"  Condition number: {self.condition_number:.2f}"
        )


class RankCalculator:
    """Calculates the numerical rank of token embedding matrices.
    
    The Gram matrix G = X @ X.T where X is the matrix of token embeddings.
    The numerical rank is the number of singular values above a threshold.
    
    A collapsing rank (normalized < 0.4) indicates the model is losing
    representational diversity - a structural compression signal.
    """
    
    # Threshold for singular value significance
    SINGULAR_VALUE_THRESHOLD = 1e-6
    COLLAPSING_THRESHOLD = 0.4
    
    def __init__(self):
        """Initialize the rank calculator."""
        self._last_analysis: Optional[RankAnalysis] = None
    
    def analyze_embeddings(
        self,
        embeddings: np.ndarray,
        threshold: Optional[float] = None
    ) -> RankAnalysis:
        """Analyze the numerical rank of an embedding matrix.
        
        Args:
            embeddings: Matrix of shape (n_tokens, embedding_dim)
            threshold: Singular value threshold (uses default if None)
            
        Returns:
            RankAnalysis with results
        """
        if threshold is None:
            threshold = self.SINGULAR_VALUE_THRESHOLD
        
        n_tokens, embedding_dim = embeddings.shape
        
        # Compute Gram matrix: G = X @ X.T
        gram_matrix = embeddings @ embeddings.T
        
        # Compute singular values of Gram matrix
        try:
            singular_values = np.linalg.svd(gram_matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            # Matrix is singular - rank is 0
            analysis = RankAnalysis(
                numerical_rank=0,
                normalized_rank=0.0,
                sample_size=n_tokens,
                is_collapsing=True,
                condition_number=float('inf')
            )
            self._last_analysis = analysis
            return analysis
        
        # Count significant singular values
        significant_sv = np.sum(singular_values > threshold)
        
        # Calculate condition number (ratio of largest to smallest non-zero SV)
        non_zero_sv = singular_values[singular_values > threshold]
        if len(non_zero_sv) > 1:
            condition_number = float(non_zero_sv[0] / non_zero_sv[-1])
        else:
            condition_number = 1.0
        
        # Normalize rank by sample size
        normalized_rank = significant_sv / n_tokens if n_tokens > 0 else 0.0
        
        # Check if collapsing
        is_collapsing = normalized_rank < self.COLLAPSING_THRESHOLD
        
        analysis = RankAnalysis(
            numerical_rank=significant_sv,
            normalized_rank=normalized_rank,
            sample_size=n_tokens,
            is_collapsing=is_collapsing,
            condition_number=condition_number
        )
        
        self._last_analysis = analysis
        return analysis
    
    def analyze_token_logits(
        self,
        logits: np.ndarray,
        top_k: int = 100
    ) -> RankAnalysis:
        """Analyze rank from token logits (proxy for embeddings).
        
        Uses the top-k logits as a low-dimensional proxy for embeddings.
        This is useful when actual embeddings are not available.
        
        Args:
            logits: Array of shape (n_tokens, vocab_size) or (n_tokens, top_k)
            top_k: Number of top logits to use for dimensionality reduction
            
        Returns:
            RankAnalysis with results
        """
        n_tokens, vocab_size = logits.shape
        
        # Use top-k logits as proxy embeddings
        if vocab_size > top_k:
            # Get indices of top-k logits for each token
            top_indices = np.argsort(logits, axis=1)[:, -top_k:]
            
            # Gather top-k logits for each token
            reduced_logits = np.take_along_axis(logits, top_indices, axis=1)
        else:
            reduced_logits = logits
        
        # Normalize logits to create pseudo-embeddings
        # Subtract mean and divide by std for each token
        mean = np.mean(reduced_logits, axis=1, keepdims=True)
        std = np.std(reduced_logits, axis=1, keepdims=True) + 1e-8
        normalized = (reduced_logits - mean) / std
        
        return self.analyze_embeddings(normalized)
    
    def get_last_analysis(self) -> Optional[RankAnalysis]:
        """Get the last rank analysis.
        
        Returns:
            RankAnalysis or None if no analysis has been run
        """
        return self._last_analysis
    
    def analyze_diversity(
        self,
        texts: list[str],
        embedding_fn: Optional[callable] = None
    ) -> RankAnalysis:
        """Analyze the diversity of a set of texts.
        
        Args:
            texts: List of text strings
            embedding_fn: Optional function to get embeddings
                         (uses simple bag-of-words if not provided)
            
        Returns:
            RankAnalysis with results
        """
        if embedding_fn:
            # Use provided embedding function
            embeddings = np.array([embedding_fn(text) for text in texts])
        else:
            # Use bag-of-words as simple embedding
            embeddings = self._bag_of_words_embeddings(texts)
        
        return self.analyze_embeddings(embeddings)
    
    def _bag_of_words_embeddings(
        self,
        texts: list[str],
        max_features: int = 1000
    ) -> np.ndarray:
        """Create simple bag-of-words embeddings.
        
        Args:
            texts: List of texts
            max_features: Maximum vocabulary size
            
        Returns:
            Embedding matrix of shape (n_texts, vocab_size)
        """
        # Build vocabulary
        vocab = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word = word.strip('.,!?;:()[]{}"\'')
                if word and len(word) > 2:
                    if word not in vocab and len(vocab) < max_features:
                        vocab[word] = len(vocab)
        
        # Create embeddings
        n_texts = len(texts)
        n_features = len(vocab)
        embeddings = np.zeros((n_texts, n_features))
        
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                word = word.strip('.,!?;:()[]{}"\'')
                if word in vocab:
                    embeddings[i, vocab[word]] += 1
        
        return embeddings


# Global calculator instance
_calculator: Optional[RankCalculator] = None


def get_calculator() -> RankCalculator:
    """Get the global rank calculator.
    
    Returns:
        RankCalculator instance
    """
    global _calculator
    if _calculator is None:
        _calculator = RankCalculator()
    return _calculator


def analyze_rank(embeddings: np.ndarray) -> RankAnalysis:
    """Analyze the rank of an embedding matrix using the global calculator.
    
    Args:
        embeddings: Matrix of shape (n_tokens, embedding_dim)
        
    Returns:
        RankAnalysis with results
    """
    return get_calculator().analyze_embeddings(embeddings)


def analyze_logits(logits: np.ndarray, top_k: int = 100) -> RankAnalysis:
    """Analyze the rank from token logits using the global calculator.
    
    Args:
        logits: Array of shape (n_tokens, vocab_size)
        top_k: Number of top logits to use
        
    Returns:
        RankAnalysis with results
    """
    return get_calculator().analyze_token_logits(logits, top_k)


def check_diversity(texts: list[str]) -> RankAnalysis:
    """Check the diversity of a set of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        RankAnalysis with results
    """
    return get_calculator().analyze_diversity(texts)
