"""
CosineSimilarity — Cosine similarity matrix from transition counts.

Computes S = cosine(M_prob) where M_prob is row-normalized
transition count matrix with self-transitions excluded.

Example
-------
>>> import numpy as np
>>> from SLBHS.similarity import CosineSimilarity
>>> C = np.random.rand(1024, 1024)
>>> sim = CosineSimilarity()
>>> S = sim.compute(C)
>>> S.shape
(1024, 1024)
"""

import numpy as np
from typing import Optional


class CosineSimilarity:
    """
    Compute cosine similarity matrix from transition count matrix.

    Pipeline:
        1. Symmetrize: W = (M + M.T) / 2
        2. Remove self-transitions (diagonal = 0)
        3. Row normalize → M_prob
        4. S[i,j] = cosine(M_prob[i] excluding i/j, M_prob[j] excluding i/j)
    """

    def __init__(self):
        """Initialize with no parameters."""
        self.S: Optional[np.ndarray] = None

    def compute(self, M: np.ndarray, symmetrize: bool = True) -> np.ndarray:
        """
        Compute cosine similarity matrix S.

        Args:
            M (np.ndarray): (k, k) transition count matrix C
            symmetrize (bool): whether to symmetrize W = (M + M.T) / 2

        Returns:
            np.ndarray: (k, k) cosine similarity matrix S
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Step 1: Symmetrize
        W = (M + M.T) / 2.0 if symmetrize else M.copy()

        # Step 2: Remove self-transitions (diagonal = 0)
        np.fill_diagonal(W, 0)

        # Step 3: Row normalization → M_prob
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M_prob = W / row_sums

        # Step 4: Compute S[i,j] = cosine(M_prob[i] excluding i/j, M_prob[j] excluding i/j)
        k = M_prob.shape[0]
        S = np.zeros((k, k), dtype=np.float64)

        for i in range(k):
            for j in range(i + 1, k):
                mask = np.ones(k, dtype=bool)
                mask[i] = False
                mask[j] = False
                vec_i = M_prob[i][mask]
                vec_j = M_prob[j][mask]
                S[i, j] = S[j, i] = cosine_similarity([vec_i], [vec_j])[0, 0]

        np.fill_diagonal(S, 1.0)
        self.S = S
        return S

    def get_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: (k, k) cosine similarity matrix S
        """
        if self.S is None:
            raise RuntimeError("CosineSimilarity.compute() must be called first")
        return self.S