"""
HandLabeler — Classify L/R hand from orientation vectors.

Uses vec_x × vec_y dot vec_z to determine handedness:
- dot < 0 → Left hand (L)
- dot >= 0 → Right hand (R)

Physical meaning: cross(x, y) produces a vector perpendicular to
the palm plane; dot product with z indicates handedness.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class HandLabeler:
    """
    Classify hand orientation as Left or Right.

    Example
    -------
    >>> import numpy as np
    >>> from SLBHS.similarity import HandLabeler
    >>> x_vec = np.random.rand(100, 3)
    >>> y_vec = np.random.rand(100, 3)
    >>> z_vec = np.random.rand(100, 3)
    >>> labeler = HandLabeler()
    >>> hand_labels = labeler.fit_predict(x_vec, y_vec, z_vec)
    >>> hand_labels.shape
    (100,)
    """

    def __init__(self):
        """Initialize."""
        self._fitted = False

    def fit_predict(self, x_vec: np.ndarray, y_vec: np.ndarray,
                    z_vec: np.ndarray) -> np.ndarray:
        """
        Fit and predict L/R hand labels.

        Args:
            x_vec (np.ndarray): (N, 3) x-axis vectors
            y_vec (np.ndarray): (N, 3) y-axis vectors
            z_vec (np.ndarray): (N, 3) z-axis vectors

        Returns:
            np.ndarray: (N,) '<U1' hand labels ('L' or 'R')
        """
        cross_xy = np.cross(x_vec, y_vec)  # (N, 3)
        dot_product = np.sum(cross_xy * z_vec, axis=1)  # (N,)

        self._fitted = True

        labels = np.empty(x_vec.shape[0], dtype='<U1')
        # Degenerate case: when x_vec and y_vec are collinear, cross_xy=0,
        # so dot_product=0 and the sample is classified as 'R' (dot >= 0).
        labels[dot_product < 0] = 'L'
        labels[dot_product >= 0] = 'R'

        logger.info(
            f"[HandLabeler] L={int(np.sum(labels=='L'))}  "
            f"R={int(np.sum(labels=='R'))}"
        )
        return labels

    def fit(self, x_vec: np.ndarray, y_vec: np.ndarray,
            z_vec: np.ndarray) -> None:
        """
        Fit (sklearn-style, results stored internally).

        Args:
            x_vec (np.ndarray): (N, 3) x-axis vectors
            y_vec (np.ndarray): (N, 3) y-axis vectors
            z_vec (np.ndarray): (N, 3) z-axis vectors
        """
        self.fit_predict(x_vec, y_vec, z_vec)

    def predict(self, x_vec: np.ndarray, y_vec: np.ndarray,
                z_vec: np.ndarray) -> np.ndarray:
        """
        Predict L/R hand labels (requires prior fit).

        Args:
            x_vec (np.ndarray): (N, 3) x-axis vectors
            y_vec (np.ndarray): (N, 3) y-axis vectors
            z_vec (np.ndarray): (N, 3) z-axis vectors

        Returns:
            np.ndarray: (N,) '<U1' hand labels ('L' or 'R')
        """
        if not self._fitted:
            raise RuntimeError("HandLabeler must be fitted before predict()")
        cross_xy = np.cross(x_vec, y_vec)
        dot_product = np.sum(cross_xy * z_vec, axis=1)
        labels = np.empty(x_vec.shape[0], dtype='<U1')
        labels[dot_product < 0] = 'L'
        labels[dot_product >= 0] = 'R'
        return labels