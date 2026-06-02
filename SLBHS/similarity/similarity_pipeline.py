"""
SimilarityPipeline — End-to-end similarity computation from H5 folder.

Loads a pre-trained clusterer, processes all H5 files to build
transition matrix, and computes cosine similarity matrix.

Example
-------
>>> from SLBHS.clustering import ThetaClusterer
>>> from SLBHS.similarity import SimilarityPipeline
>>> 
>>> # Load pre-trained clusterer
>>> clusterer = ThetaClusterer()
>>> clusterer.load('/path/to/model/')
>>> 
>>> # Create pipeline
>>> pipeline = SimilarityPipeline(
>>>     clusterer=clusterer,
>>>     k=1024,
>>>     delta_t=1
>>> )
>>> 
>>> # Run on all H5 files
>>> pipeline.run(h5_folder='/path/to/h5/')
>>> 
>>> # Save results
>>> pipeline.save('/path/to/results/')
>>> 
>>> print(f"Similarity matrix: {pipeline.similarity_matrix.shape}")

Attributes:
    similarity_matrix: (k, k) cosine similarity matrix
    transition_matrix: (k, k) transition count matrix
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union

from .hand_labeler import HandLabeler
from .transition_counter import TransitionCounter
from .cosine_similarity import CosineSimilarity

logger = logging.getLogger(__name__)


class SimilarityPipeline:
    """
    End-to-end similarity computation pipeline.

    Reads all H5 files from a folder, uses a clusterer to predict
    token labels, builds transition matrix, and computes cosine similarity.
    """

    def __init__(
        self,
        clusterer,
        k: int = 1024,
        delta_t: int = 1,
        min_transitions: int = 0,
    ):
        """
        Args:
            clusterer: Pre-trained clusterer with predict() method.
                      Must have predict(X) -> labels interface.
            k (int): Number of token classes (default 1024).
            delta_t (int): Steps to look back for transitions (default 1).
            min_transitions (int): Minimum transitions to keep (default 0).

        Raises:
            AttributeError: If clusterer does not have predict() method.
        """
        # Validate clusterer has predict method
        if not hasattr(clusterer, 'predict'):
            raise AttributeError(
                f"clusterer must have a 'predict' method. "
                f"Got {type(clusterer).__name__} which does not have predict."
            )

        self.clusterer = clusterer
        self.k = k
        self.delta_t = delta_t
        self.min_transitions = min_transitions

        self.hand_labeler = HandLabeler()
        self.transition_counter = TransitionCounter(
            k=k, delta_t=delta_t, min_transitions=min_transitions
        )
        self.cosine_similarity = CosineSimilarity()

        self.transition_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self._fitted = False

    def run(
        self,
        h5_folder: Union[str, Path],
        verbose: bool = True
    ) -> 'SimilarityPipeline':
        """
        Process all H5 files in folder and compute similarity matrix.

        Args:
            h5_folder (str or Path): Path to folder containing H5 files.
            verbose (bool): Print progress (default True).

        Returns:
            self (chainable)
        """
        h5_folder = Path(h5_folder)
        h5_files = list(h5_folder.glob('*.h5'))

        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {h5_folder}")

        if verbose:
            logger.info(f"[SimilarityPipeline] Found {len(h5_files)} H5 files")

        # Process each H5 file
        for h5_path in h5_files:
            if verbose:
                logger.info(f"Processing {h5_path.name}...")

            import h5py
            with h5py.File(h5_path, 'r') as f:
                # Get aligned_63d data
                if 'aligned_63d' not in f:
                    logger.warning(f"{h5_path.name}: No 'aligned_63d' key, skipping")
                    continue

                X = f['aligned_63d'][:]

                # Get orientation vectors
                x_vec = f['x_vec'][:] if 'x_vec' in f else None
                y_vec = f['y_vec'][:] if 'y_vec' in f else None
                z_vec = f['z_vec'][:] if 'z_vec' in f else None

                if x_vec is None or y_vec is None or z_vec is None:
                    logger.warning(f"{h5_path.name}: Missing orientation vectors, skipping")
                    continue

                # Predict token labels
                labels = self.clusterer.predict(X)  # (N,)

                # Classify L/R hand
                hand_labels = self.hand_labeler.fit_predict(x_vec, y_vec, z_vec)

                # Update transition matrix
                self.transition_counter.update(labels, hand_labels)

        # Compute similarity matrix from transition matrix
        C = self.transition_counter.get_matrix()
        self.similarity_matrix = self.cosine_similarity.compute(C)
        self.transition_matrix = C

        self._fitted = True

        if verbose:
            logger.info(
                f"[SimilarityPipeline] Done. "
                f"S.shape={self.similarity_matrix.shape}, "
                f"nnz={np.sum(C > 0)}"
            )

        return self

    def save(self, results_dir: Union[str, Path]) -> dict:
        """
        Save similarity matrix and related data.

        Args:
            results_dir (str or Path): Output directory path.

        Returns:
            dict: {'similarity_matrix': path, 'transition_matrix': path, 'meta': path}
        """
        if not self._fitted:
            raise RuntimeError("run() must be called before save()")

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save similarity matrix
        sim_path = results_dir / 'similarity_matrix.npy'
        np.save(sim_path, self.similarity_matrix)

        # Save transition matrix
        trans_path = results_dir / 'transition_matrix.npy'
        np.save(trans_path, self.transition_matrix)

        # Save metadata
        import json
        meta = {
            'k': self.k,
            'delta_t': self.delta_t,
            'min_transitions': self.min_transitions,
            'shape': self.similarity_matrix.shape,
        }
        meta_path = results_dir / 'pipeline_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[SimilarityPipeline] Results saved to {results_dir}")

        return {
            'similarity_matrix': str(sim_path),
            'transition_matrix': str(trans_path),
            'meta': str(meta_path),
        }

    def get_similarity_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: (k, k) cosine similarity matrix.
        """
        if not self._fitted:
            raise RuntimeError("run() must be called first")
        return self.similarity_matrix

    def get_transition_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: (k, k) transition count matrix.
        """
        if not self._fitted:
            raise RuntimeError("run() must be called first")
        return self.transition_matrix