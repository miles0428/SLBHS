"""
Cosine Feature Extraction for MediaPipe Hand Landmarks.

Transform (N, 21, 3) → (N, 15) cosine similarity features.
Each feature = cos(angle) between two consecutive bone vectors.

Reference landmark indices:
  - Thumb : [0, 1, 2, 3, 4]
  - Index : [0, 5, 6, 7, 8]
  - Middle: [0, 9, 10, 11, 12]
  - Ring  : [0, 13, 14, 15, 16]
  - Pinky : [0, 17, 18, 19, 20]

Feature indices 0-2:  Thumb  (0-1, 1-2, 2-3 vs 1-2, 2-3, 3-4)
Feature indices 3-5:  Index  (0-5, 5-6, 6-7 vs 5-6, 6-7, 7-8)
Feature indices 6-8:  Middle (0-9, 9-10, 10-11 vs 9-10, 10-11, 11-12)
Feature indices 9-11: Ring   (0-13, 13-14, 14-15 vs 13-14, 14-15, 15-16)
Feature indices 12-14: Pinky  (0-17, 17-18, 18-19 vs 17-18, 18-19, 19-20)
"""

import numpy as np


def compute_cosine_features(X, verbose=True):
    """
    Convert aligned_63d hand pose data to 15-dim cosine similarity features.

    Args:
        X: np.ndarray of shape (N, 63) — flat MediaPipe landmark data
           OR shape (N, 21, 3) — already reshaped
        verbose: bool, print progress info

    Returns:
        features: np.ndarray of shape (N, 15), dtype float32
    """
    # Auto-detect input shape
    if len(X.shape) == 2 and X.shape[1] == 63:
        # Flat (N, 63) → reshape to (N, 21, 3)
        landmarks = X.reshape(X.shape[0], 21, 3)
    elif len(X.shape) == 3 and X.shape[1] == 21 and X.shape[2] == 3:
        # Already (N, 21, 3)
        landmarks = X
    else:
        raise ValueError(f'Expected (N, 63) or (N, 21, 3), got {X.shape}')

    N = landmarks.shape[0]
    if verbose:
        print(f'[CosineFeatures] Input: {N} frames, shape={landmarks.shape}')

    # Landmark index sets: each entry is (p, m, d) for one joint
    # p=proximal, m=mid, d=distal landmark index
    # For 3 features per finger: (L0,L1,L2), (L1,L2,L3), (L2,L3,L4)
    finger_sets = [
        [(0, 1, 2), (1, 2, 3), (2, 3, 4)],   # Thumb  → 3 features
        [(0, 5, 6), (5, 6, 7), (6, 7, 8)],   # Index  → 3 features
        [(0, 9, 10), (9, 10, 11), (10, 11, 12)],  # Middle → 3 features
        [(0, 13, 14), (13, 14, 15), (14, 15, 16)], # Ring   → 3 features
        [(0, 17, 18), (17, 18, 19), (18, 19, 20)], # Pinky  → 3 features
    ]

    features = np.zeros((N, 15), dtype=np.float32)

    feat_idx = 0
    for finger_idx, joints in enumerate(finger_sets):
        for joint_idx, (p, m, d) in enumerate(joints):
            # Vector 1: mid - proximal
            u = landmarks[:, m] - landmarks[:, p]
            # Vector 2: distal - mid
            v = landmarks[:, d] - landmarks[:, m]

            # Cosine similarity: (u · v) / (||u|| * ||v||)
            u_norm = np.linalg.norm(u, axis=1, keepdims=True)
            v_norm = np.linalg.norm(v, axis=1, keepdims=True)

            # Avoid division by zero
            denom = u_norm * v_norm
            denom = np.where(denom > 1e-8, denom, 1.0)

            cos_vals = np.sum(u * v, axis=1) / denom.squeeze(-1)
            cos_vals = np.clip(cos_vals, -1.0, 1.0)  # Guard against roundoff overflow
            features[:, feat_idx] = cos_vals

            if verbose and feat_idx < 15:
                print(f'  Feature {feat_idx:2d}: finger={finger_idx} joint={joint_idx} '
                      f'({p}-{m}, {m}-{d})')
            feat_idx += 1

    if verbose:
        print(f'[CosineFeatures] Output: {features.shape}, dtype={features.dtype}')
        print(f'  Value range: [{features.min():.4f}, {features.max():.4f}]')
        # Check for NaN
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            print(f'  WARNING: {nan_count} NaN values (typically from zero-length bones)')

<<<<<<< HEAD
    return features
=======
    return features
>>>>>>> 18323399d01ff675f6ffb995d73819bde7128695
