"""
HandPoseGeometrySwapper — Mirror-swap hand pose geometry across the X-axis.

This module provides a single-class transformer that:
  1. Reflects x_vec across the origin (x → -x), leaving y_vec and z_vec unchanged.
  2. Leaves the 63-D aligned pose vector completely untouched.
  3. Flips the handedness label ('L' ↔ 'R').

Physical basis
--------------
HandLabeler determines handedness via:
    dot = np.sum(np.cross(x_vec, y_vec) * z_vec)
    L if dot < 0, R if dot >= 0

When x_vec is negated, cross(-x_vec, y_vec) = -cross(x_vec, y_vec),
so the dot product flips sign and the label inverts.  This makes the swap
a true geometric reflection, not an arbitrary label change.

Example
-------
>>> import numpy as np
>>> from SLBHS.geometry import HandPoseGeometrySwapper
>>> swapper = HandPoseGeometrySwapper()
>>> feat = {
...     'label':   'L',
...     'aligned': np.ones(63, dtype=np.float32),
...     'x_vec':   np.array([1., 0., 0.], dtype=np.float32),
...     'y_vec':   np.array([0., 1., 0.], dtype=np.float32),
...     'z_vec':   np.array([0., 0., 1.], dtype=np.float32),
... }
>>> out = swapper.swap(feat)
>>> out['label']
'R'
>>> out['x_vec']
array([-1.,  0.,  0.])
>>> out['aligned'] is feat['aligned']          # original untouched
True
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any


class HandPoseGeometrySwapper:
    """
    Mirror-swap a hand-pose feature dict across the X-axis.

    Parameters
    ----------
    None.

    Methods
    -------
    swap(feature_dict: Dict[str, Any]) -> Dict[str, Any]
        Returns a NEW dict with mirrored geometry; original is unmodified.
    """

    # Expected keys in the input feature dict
    _KEY_LABEL   = 'label'
    _KEY_ALIGNED = 'aligned'
    _KEY_X       = 'x_vec'
    _KEY_Y       = 'y_vec'
    _KEY_Z       = 'z_vec'

    def __init__(self) -> None:
        """Initialize the swapper (no state needed)."""
        pass

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def swap(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a mirrored copy of the input hand-pose feature dict.

        Args:
            feature_dict: Must contain the keys
                - ``label``   (str, 'L' or 'R')
                - ``aligned`` (np.ndarray, shape=(63,))
                - ``x_vec``   (np.ndarray, shape=(3,))
                - ``y_vec``   (np.ndarray, shape=(3,))
                - ``z_vec``   (np.ndarray, shape=(3,))

        Returns:
            Dict[str, Any]: A **new** dict (original is never modified) with:
                - ``x_vec`` negated on the X component (X → -X)
                - ``y_vec`` and ``z_vec`` unchanged
                - ``aligned`` copied by reference (the 63-D vector is untouched)
                - ``label`` flipped ('L' → 'R', 'R' → 'L')

        Raises
        ------
        KeyError
            If any required key is missing from ``feature_dict``.
        ValueError
            If ``label`` is not 'L' or 'R'.
        """
        self._validate_keys(feature_dict)
        self._validate_label(feature_dict[self._KEY_LABEL])

        original = feature_dict

        # Negate X-axis — in-place negation of a copy; original array untouched
        x_vec_new = -original[self._KEY_X].copy()
        y_vec_new = original[self._KEY_Y].copy()   # unchanged
        z_vec_new = original[self._KEY_Z].copy()   # unchanged

        # Flip handedness
        label_map = {'L': 'R', 'R': 'L'}
        label_new = label_map[original[self._KEY_LABEL]]

        # aligned is passed through unchanged (same object, no copy made)
        return {
            self._KEY_LABEL:   label_new,
            self._KEY_ALIGNED: original[self._KEY_ALIGNED],
            self._KEY_X:       x_vec_new,
            self._KEY_Y:       y_vec_new,
            self._KEY_Z:       z_vec_new,
        }

    # ------------------------------------------------------------------ #
    # Validation helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_keys(fd: Dict[str, Any]) -> None:
        """Raise KeyError if any required key is absent."""
        required = [
            HandPoseGeometrySwapper._KEY_LABEL,
            HandPoseGeometrySwapper._KEY_ALIGNED,
            HandPoseGeometrySwapper._KEY_X,
            HandPoseGeometrySwapper._KEY_Y,
            HandPoseGeometrySwapper._KEY_Z,
        ]
        missing = [k for k in required if k not in fd]
        if missing:
            raise KeyError(
                f"feature_dict missing required key(s): {missing}"
            )

    @staticmethod
    def _validate_label(label: Any) -> None:
        """Raise ValueError if label is not 'L' or 'R'."""
        if label not in ('L', 'R'):
            raise ValueError(
                f"label must be 'L' or 'R', got {repr(label)}"
            )