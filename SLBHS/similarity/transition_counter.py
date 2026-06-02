"""
TransitionCounter — Build token transition matrix.

Traverses Left/Right tracks (sorted by time), for each token i,
looks back delta_t steps: i→i+1, i→i+2, ..., i→i+delta_t.
Each matching pair increments C[T_i, T_{i+k}] by 1.

Example
-------
>>> from SLBHS.similarity import TransitionCounter
>>> import numpy as np
>>> labels = np.array([0, 1, 2, 0, 1, 2])
>>> hand_labels = np.array(['L', 'L', 'R', 'R', 'L', 'R'])
>>> counter = TransitionCounter(k=1024, delta_t=1)
>>> counter.fit(labels, hand_labels)
>>> C = counter.get_matrix()
>>> C.shape
(1024, 1024)
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TransitionCounter:
    """
    Build token transition matrix C[k x k].

    Tracks left/right hand separately, accumulates transition counts
    between tokens across time steps.
    """

    def __init__(self, k: int = 1024, delta_t: int = 1, min_transitions: int = 0):
        """
        Args:
            k (int): Total number of tokens (default 1024).
            delta_t (int): Number of steps to look back (default 1).
            min_transitions (int): Minimum transitions to keep (default 0).
        """
        self.k = k
        self.delta_t = delta_t
        self.min_transitions = min_transitions
        self.C: Optional[np.ndarray] = None

    def update(self, labels_batch: np.ndarray,
               hand_labels_batch: np.ndarray) -> 'TransitionCounter':
        """
        Add a batch of labels to the transition matrix (cumulative).

        Args:
            labels_batch (np.ndarray): (N,) token IDs (0 to k-1)
            hand_labels_batch (np.ndarray): (N,) 'L'/'R' hand labels

        Returns:
            self (chainable)
        """
        delta_t = self.delta_t
        if self.C is None:
            self.C = np.zeros((self.k, self.k), dtype=np.float64)

        left_mask = hand_labels_batch == 'L'
        right_mask = hand_labels_batch == 'R'
        left_track = labels_batch[left_mask]
        right_track = labels_batch[right_mask]

        for track in (left_track, right_track):
            if track.shape[0] <= delta_t:
                continue
            from_tokens_list = [track[:-k] for k in range(1, delta_t + 1)]
            to_tokens_list = [track[k:] for k in range(1, delta_t + 1)]
            from_all = np.concatenate(from_tokens_list)
            to_all = np.concatenate(to_tokens_list)
            np.add.at(self.C, (from_all, to_all), 1)

        logger.info(f"[TransitionCounter.update] batch added, C.nnz={int(np.sum(self.C > 0))}")
        return self

    def fit(self, labels: np.ndarray, hand_labels: np.ndarray,
            delta_t: Optional[int] = None,
            min_transitions: Optional[int] = None) -> 'TransitionCounter':
        """
        Build transition matrix from labels and hand labels.

        Args:
            labels (np.ndarray): (N,) token IDs (0 to k-1)
            hand_labels (np.ndarray): (N,) 'L'/'R' hand labels
            delta_t (int): Steps to look back (default self.delta_t)
            min_transitions (int): Minimum transitions to keep (default self.min_transitions)

        Returns:
            self (chainable)
        """
        delta_t = delta_t if delta_t is not None else self.delta_t
        min_transitions = min_transitions if min_transitions is not None else self.min_transitions

        k = self.k
        C = np.zeros((k, k), dtype=np.float64)

        left_mask = hand_labels == 'L'
        right_mask = hand_labels == 'R'
        left_track = labels[left_mask]
        right_track = labels[right_mask]

        for track in (left_track, right_track):
            if track.shape[0] <= delta_t:
                continue
            from_tokens_list = [track[:-k] for k in range(1, delta_t + 1)]
            to_tokens_list = [track[k:] for k in range(1, delta_t + 1)]
            from_all = np.concatenate(from_tokens_list)
            to_all = np.concatenate(to_tokens_list)
            np.add.at(C, (from_all, to_all), 1)

        if min_transitions > 0:
            C[C < min_transitions] = 0

        self.C = C
        logger.info(f"[TransitionCounter] delta_t={delta_t}, min_transitions={min_transitions}, "
                    f"C.shape={C.shape}, nnz={int(np.sum(C > 0))}, max={C.max():.0f}")
        return self

    def get_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: (k, k) transition count matrix C
        """
        if self.C is None:
            raise RuntimeError("TransitionCounter must be fitted first")
        return self.C