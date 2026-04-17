"""
SuperClusterer: Hierarchical clustering of K-Means centers into super clusters.
"""
import numpy as np
import os
import json
from sklearn.cluster import AgglomerativeClustering


class SuperClusterer:
    """
    Take K-Means cluster centers and group them into super clusters.

    Usage:
        sc = SuperClusterer(kmeans_labels, kmeans_centers)
        sc.fit(n_super=20, linkage='ward')
        sc.save()
        sc.load()
    """

    def __init__(self, kmeans_labels=None, kmeans_centers=None, results_dir=None):
        """
        Args:
            kmeans_labels: np.ndarray (N,) — per-frame K-Means labels
            kmeans_centers: np.ndarray (k, 63) — K-Means cluster centers
            results_dir: str, save location
        """
        self.kmeans_labels = kmeans_labels
        self.kmeans_centers = kmeans_centers
        self.results_dir = results_dir or self._default_results_dir()
        os.makedirs(self.results_dir, exist_ok=True)

        self.n_super_ = None
        self.super_labels_ = None   # (k,) — which super cluster each center belongs to
        self.frame_super_ = None    # (N,) — which super cluster each frame belongs to

    @staticmethod
    def _default_results_dir():
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'results')

    # --------------------------------------------------------------------------
    # Fit
    # --------------------------------------------------------------------------

    def fit(self, n_super=20, linkage='ward', metric='euclidean'):
        """
        Agglomerative clustering on centers.

        Args:
            n_super: number of super clusters
            linkage: 'ward', 'complete', 'average', 'single'
            metric: distance metric
        Returns:
            super_labels: np.ndarray (k,) super cluster per center
            frame_super: np.ndarray (N,) super cluster per frame
        """
        if self.kmeans_centers is None:
            raise ValueError('kmeans_centers must be set')

        print(f'[SuperClusterer] Hierarchical clustering {len(self.kmeans_centers)} centers '
              f'into {n_super} super clusters (linkage={linkage}) ...')
        hc = AgglomerativeClustering(
            n_clusters=n_super,
            linkage=linkage,
            metric=metric,
        )
        self.super_labels_ = hc.fit_predict(self.kmeans_centers).astype(int)
        self.n_super_ = n_super

        # Map each frame to its super cluster
        self.frame_super_ = self.super_labels_[self.kmeans_labels]
        print(f'[SuperClusterer] Done.')
        self._print_distribution()
        return self.super_labels_, self.frame_super_

    def _print_distribution(self):
        for sc in range(self.n_super_):
            n_frames = np.sum(self.frame_super_ == sc)
            n_centers = np.sum(self.super_labels_ == sc)
            print(f'  SC {sc:2d}: {n_frames:5d} frames, {n_centers:3d} centers')

    # --------------------------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------------------------

    def save(self, results_dir=None):
        """Save super_labels and meta.json."""
        if self.super_labels_ is None:
            raise RuntimeError('Must fit() before save()')

        out_dir = results_dir or self.results_dir
        os.makedirs(out_dir, exist_ok=True)

        super_labels_path = os.path.join(out_dir, 'super_labels.npy')
        np.save(super_labels_path, self.super_labels_)

        meta = {
            'n_super': int(self.n_super_),
            'n_centers': int(len(self.kmeans_centers)),
            'linkage': 'ward',
        }
        meta_path = os.path.join(out_dir, 'super_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'[SuperClusterer] Saved to {out_dir}')
        return {'super_labels': super_labels_path, 'meta': meta_path}

    def load(self, results_dir=None):
        """Load super_labels from disk."""
        in_dir = results_dir or self.results_dir
        super_labels_path = os.path.join(in_dir, 'super_labels.npy')
        meta_path = os.path.join(in_dir, 'super_meta.json')

        self.super_labels_ = np.load(super_labels_path).astype(int)
        self.n_super_ = int(len(np.unique(self.super_labels_)))

        with open(meta_path) as f:
            meta = json.load(f)

        # Reconstruct frame_super from kmeans_labels
        if self.kmeans_labels is not None:
            self.frame_super_ = self.super_labels_[self.kmeans_labels]

        print(f'[SuperClusterer] Loaded {self.n_super_} super clusters from {in_dir}')
        return self.super_labels_
