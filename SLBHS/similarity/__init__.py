"""
SLBHS.similarity — Time-series similarity computation module.

Provides cosine similarity-based hand pose clustering utilities:
- CosineSimilarity: Compute similarity matrix from transition counts
- HandLabeler: Classify L/R hand from orientation vectors
- TransitionCounter: Build token transition matrix
"""

from .cosine_similarity import CosineSimilarity
from .hand_labeler import HandLabeler
from .transition_counter import TransitionCounter

__all__ = [
    'CosineSimilarity',
    'HandLabeler',
    'TransitionCounter',
]