"""
UMAPReducer / PCAReducer: Dimensionality reduction with persistent cache.
"""
import numpy as np
import os
import json
import hashlib


class UMAPReducer:
    """
    UMAP dimensionality reduction with cache.

    Usage:
        reducer = UMAPReducer(X_scaled, cache_dir='results/umap_cache')
        overview = reducer.transform_overview(n=10000, seed=42)
        sc_umap  = reducer.transform_sc(sc_id=5, n=2000, seed=42)
    """

    def __init__(self, X_scaled, super_labels=None, cache_dir=None):
        """
        Args:
            X_scaled: np.ndarray (N, D) — standardized data
            super_labels: np.ndarray (N,) — super cluster per frame
            cache_dir: str, where to cache UMAP results
        """
        self.X = X_scaled
        self.super_labels = super_labels
        self.cache_dir = cache_dir or self._default_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _default_cache_dir():
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'results', 'umap_cache')

    def _cache_key(self, kind, **kwargs):
        """Generate a unique cache key from parameters."""
        key = json.dumps({'kind': kind, **kwargs}, sort_keys=True)
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f'{kind}_{digest}.npy')

    def _load_cache(self, kind, **kwargs):
        path = self._cache_key(kind, **kwargs)
        if os.path.exists(path):
            print(f'  [UMAPReducer] Cache hit: {os.path.basename(path)}')
            return np.load(path)
        return None

    def _save_cache(self, arr, kind, **kwargs):
        path = self._cache_key(kind, **kwargs)
        np.save(path, arr)
        print(f'  [UMAPReducer] Cached to {os.path.basename(path)}')
        return arr

    def transform_overview(self, n=10000, seed=42, n_neighbors=30, min_dist=0.1):
        """
        UMAP on subsampled data (for overview plot).

        Returns:
            result: np.ndarray (n_actual, 2) UMAP coordinates
            indices: np.ndarray (n_actual,) — the original indices used (for slicing labels)
        """
        import umap
        cache = self._load_cache('overview', n=n, seed=seed,
                                 n_neighbors=n_neighbors, min_dist=min_dist)
        if cache is not None:
            # Cache stores only coords; return with synthetic indices
            n_actual = len(cache)
            np.random.seed(seed)
            all_idx = np.random.choice(len(self.X), min(n, len(self.X)), replace=False)
            return cache, all_idx[:n_actual]

        np.random.seed(seed)
        idx = np.random.choice(len(self.X), min(n, len(self.X)), replace=False)

        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        result = reducer.fit_transform(self.X[idx])
        self._save_cache(result, 'overview', n=n, seed=seed,
                        n_neighbors=n_neighbors, min_dist=min_dist)
        return result, idx

    def transform_sc(self, sc_id, n=2000, seed=42, n_neighbors=30, min_dist=0.1):
        """
        UMAP on frames belonging to a specific super cluster.

        Returns:
            result: np.ndarray (n_actual, 2) UMAP coordinates
            indices: np.ndarray (n_actual,) — indices into X_sc (not original X)
        """
        import umap
        if self.super_labels is None:
            raise ValueError('super_labels required for SC-specific UMAP')

        cache = self._load_cache(f'sc{sc_id:02d}', n=n, seed=seed,
                                 n_neighbors=n_neighbors, min_dist=min_dist)
        if cache is not None:
            n_actual = len(cache)
            np.random.seed(seed)
            X_sc_len = np.sum(self.super_labels == sc_id)
            sc_idx = np.random.choice(X_sc_len, n_actual, replace=False)
            return cache, sc_idx

        mask = self.super_labels == sc_id
        X_sc = self.X[mask]
        n_actual = min(n, len(X_sc))

        if n_actual < 5:
            return np.zeros((0, 2)), np.array([], dtype=int)

        np.random.seed(seed)
        idx = np.random.choice(len(X_sc), n_actual, replace=False)

        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=min(n_neighbors, n_actual - 1),
            min_dist=min_dist,
        )
        result = reducer.fit_transform(X_sc[idx])
        self._save_cache(result, f'sc{sc_id:02d}', n=n, seed=seed,
                         n_neighbors=n_neighbors, min_dist=min_dist)
        return result, idx

    def transform_pca(self):
        """PCA 2D fallback (always fast, no cache needed)."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        return pca.fit_transform(self.X)


class PCAReducer:
    """Simple PCA 2D wrapper with cache."""

    def __init__(self, X_scaled, cache_dir=None):
        self.X = X_scaled
        self.cache_dir = cache_dir

    def transform(self):
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(self.X)
