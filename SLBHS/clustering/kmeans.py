"""
KMeansClusterer: K-Means clustering of hand pose vectors with caching.
"""
import numpy as np
import os
import json
import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class KMeansClusterer:
    """
    Fit K-Means on aligned_63d data, save/load labels and centers.

    Usage:
        kc = KMeansClusterer(X, results_dir='TWSLT/results')
        kc.fit(k=512, seed=42)
        labels, centers = kc.fit(k=512, seed=42)   # also returns
        kc.save()
        kc.load()
    """

    def __init__(self, X=None, results_dir=None, k=None, seed=42):
        """
        Args:
            X: np.ndarray (N, 63), optional. If None, call load().
            results_dir: str, where to save results.
            k: int, number of clusters. Default 512.
            seed: int, random seed. Default 42.
        """
        self.X = X
        self.results_dir = results_dir or self._default_results_dir()
        os.makedirs(self.results_dir, exist_ok=True)

        self.k = k
        self.seed = seed
        self.labels_ = None
        self.centers_ = None
        self.scaler = StandardScaler()
        self.X_scaled_ = None

    @staticmethod
    def _default_results_dir():
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'results')

    # --------------------------------------------------------------------------
    # Fit
    # --------------------------------------------------------------------------

    def fit(self, k=None, seed=None, X=None,
            init='k-means++', n_init=10, max_iter=300,
            algorithm='lloyd'):
        """
        Run K-Means clustering.

        Args:
            k: number of clusters
            seed: random seed
            X: data array (N, 63). If None, uses self.X.
            init, n_init, max_iter, algorithm: passed to sklearn KMeans
        Returns:
            labels: np.ndarray (N,) cluster labels
            centers: np.ndarray (k, 63) cluster centers
        """
        if k is not None:
            self.k = k
        if seed is not None:
            self.seed = seed
        if X is not None:
            self.X = X

        if self.X is None:
            raise ValueError('X must be set before fit()')

        print(f'[KMeansClusterer] Scaling {self.X.shape} ...')
        self.X_scaled_ = self.scaler.fit_transform(self.X)

        print(f'[KMeansClusterer] Fitting K-Means k={self.k} seed={self.seed} ...')
        self.km = KMeans(
            n_clusters=self.k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.seed,
            
        )
        self.km.fit(self.X_scaled_)
        self.labels_ = self.km.labels_
        # Ensure centers are float64 to avoid dtype mismatch in predict()
        self.km.cluster_centers_ = self.km.cluster_centers_.astype(np.float64)
        self.centers_ = self.km.cluster_centers_

        print(f'[KMeansClusterer] Done. Inertia: {self.km.inertia_:.0f}')
        return self.labels_, self.centers_

    def fit_transform(self, X=None, **kwargs):
        """Alias for fit() then return (labels, centers)."""
        return self.fit(X=X, **kwargs)

    # --------------------------------------------------------------------------

    def elbow(self, k_range, n_init=10, max_iter=300, seed=None):
        """
        Compute inertia for a range of k values.

        Args:
            k_range: list[int] of k values to try
            n_init, max_iter, seed: passed to KMeans
        Returns:
            list of (k, inertia)
        """
        results = []
        X = self.scaler.fit_transform(self.X) if self.X_scaled_ is None else self.X_scaled_
        for k in k_range:
            km = KMeans(n_clusters=k, init='k-means++', n_init=n_init,
                        max_iter=max_iter, random_state=seed or self.seed)
            km.fit(X)
            results.append((k, float(km.inertia_)))
            print(f'  k={k:4d}  inertia={km.inertia_:.0f}')
        return results

    def silhouette(self, k_range, n_samples=5000, n_init=10, max_iter=300, seed=None):
        """
        Compute average silhouette score for a range of k values.
        Subsamples n_samples frames for speed.
        """
        from sklearn.metrics import silhouette_score
        results = []
        X = self.scaler.fit_transform(self.X) if self.X_scaled_ is None else self.X_scaled_
        if len(X) > n_samples:
            idx = np.random.RandomState(seed or self.seed).choice(len(X), n_samples, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        for k in k_range:
            km = KMeans(n_clusters=k, init='k-means++', n_init=n_init,
                        max_iter=max_iter, random_state=seed or self.seed)
            labels = km.fit_predict(X_sub)
            score = silhouette_score(X_sub, labels)
            results.append((k, float(score)))
            print(f'  k={k:4d}  silhouette={score:.4f}')
        return results

    # --------------------------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------------------------

    def save(self, results_dir=None, prefix='kmeans'):
        """Save labels, centers, and meta.json."""
        if self.labels_ is None or self.centers_ is None:
            raise RuntimeError('Must fit() before save()')

        out_dir = results_dir or self.results_dir
        os.makedirs(out_dir, exist_ok=True)

        labels_path = os.path.join(out_dir, 'labels.npy')
        centers_path = os.path.join(out_dir, 'centers.npy')
        meta_path = os.path.join(out_dir, 'kmeans_meta.json')

        np.save(labels_path, self.labels_)
        np.save(centers_path, self.centers_)

        meta = {
            'k': int(self.k),
            'seed': int(self.seed),
            'n_frames': int(len(self.labels_)),
            'inertia': float(
                np.sum((self.X_scaled_[self.labels_] - self.centers_[self.labels_])**2)
            ) if self.X_scaled_ is not None else None,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'[KMeansClusterer] Saved to {out_dir}')
        return {'labels': labels_path, 'centers': centers_path, 'meta': meta_path}

    def load(self, results_dir=None):
        """Load labels and centers from disk. Supports both meta.json and legacy config.json."""
        in_dir = results_dir or self.results_dir
        labels_path = os.path.join(in_dir, 'labels.npy')
        centers_path = os.path.join(in_dir, 'centers.npy')
        meta_path = os.path.join(in_dir, 'kmeans_meta.json')
        legacy_meta_path = os.path.join(in_dir, 'config.json')

        self.labels_ = np.load(labels_path)
        self.centers_ = np.load(centers_path)

        # Try new format first
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.k = meta['k']
            self.seed = meta['seed']
        elif os.path.exists(legacy_meta_path):
            # Legacy: k stored in config.json
            with open(legacy_meta_path) as f:
                cfg = json.load(f)
            self.k = cfg.get('k', len(self.centers_))
            self.seed = cfg.get('seed', 42)
            print(f'[KMeansClusterer] Loaded (legacy config.json) k={self.k} from {in_dir}')
        else:
            # Fallback: infer from shapes
            self.k = len(self.centers_)
            self.seed = 42
            print(f'[KMeansClusterer] No meta file, inferred k={self.k} from {in_dir}')
            return self.labels_, self.centers_

        print(f'[KMeansClusterer] Loaded k={self.k} from {in_dir}')
        return self.labels_, self.centers_

    # --------------------------------------------------------------------------
    # MiniBatch K-Means
    # --------------------------------------------------------------------------

    def fit_minibatch(self, k=None, seed=None, X=None,
                  init='k-means++', n_init=3, max_iter=300,
                  batch_size=5000, reassignment_ratio=0.01, max_no_improvement=10,
                  algorithm='lloyd'):
        """
        Run MiniBatch K-Means clustering (faster for large datasets).

        Args:
            k: number of clusters
            seed: random seed
            X: data array (N, 63). If None, uses self.X.
            init, n_init, max_iter: passed to sklearn MiniBatchKMeans
            batch_size: size of mini-batches
            reassignment_ratio: fraction of samples to reassign per iteration
            max_no_improvement: stop if inertia doesn't improve for this many iterations
            algorithm: 'lloyd' or 'elkan'
        Returns:
            labels: np.ndarray (N,) cluster labels
            centers: np.ndarray (k, 63) cluster centers
        """
        if k is not None:
            self.k = k
        if seed is not None:
            self.seed = seed
        if X is not None:
            self.X = X

        if self.X is None:
            raise ValueError('X must be set before fit_minibatch()')

        print(f'[MiniBatchKMeans] Scaling {self.X.shape} ...')
        self.X_scaled_ = self.scaler.fit_transform(self.X)

        print(f'[MiniBatchKMeans] Fitting MiniBatchKMeans k={self.k} seed={self.seed} batch_size={batch_size} n_init={n_init} max_iter={max_iter} ...')
        self.km = MiniBatchKMeans(
            n_clusters=self.k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            batch_size=batch_size,
            reassignment_ratio=reassignment_ratio,
            max_no_improvement=max_no_improvement,
            random_state=self.seed,
            verbose=1,
        )
        self.km.fit(self.X_scaled_)
        self.labels_ = self.km.labels_
        # Ensure centers are float64 to avoid dtype mismatch in predict()
        self.km.cluster_centers_ = self.km.cluster_centers_.astype(np.float64)
        self.centers_ = self.km.cluster_centers_

        print(f'[MiniBatchKMeans] Done. Inertia: {self.km.inertia_:.0f}')
        return self.labels_, self.centers_

    # --------------------------------------------------------------------------
    # Model Save / Load (for inference)
    # --------------------------------------------------------------------------

    def save_model(self, results_dir=None, prefix='kmeans'):
        """
        Save sklearn model (KMeans + StandardScaler) for inference.
        Use load_model() to load and predict() to classify new data.
        """
        if not hasattr(self, 'km') or self.km is None:
            raise RuntimeError('Must fit() with store_model=True before save_model()')

        # Ensure centers are float64 to avoid dtype mismatch in predict()
        if self.km.cluster_centers_.dtype != np.float64:
            self.km.cluster_centers_ = self.km.cluster_centers_.astype(np.float64)

        out_dir = results_dir or self.results_dir
        os.makedirs(out_dir, exist_ok=True)

        model_path = os.path.join(out_dir, f'{prefix}_model.joblib')
        scaler_path = os.path.join(out_dir, f'{prefix}_scaler.joblib')

        joblib.dump(self.km, model_path)
        joblib.dump(self.scaler, scaler_path)

        meta = {
            'k': int(self.k),
            'seed': int(self.seed),
            'model_type': type(self.km).__name__,
        }
        meta_path = os.path.join(out_dir, f'{prefix}_model_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'[KMeansClusterer] Saved model to {out_dir}')
        return {'model': model_path, 'scaler': scaler_path}

    def load_model(self, results_dir=None, prefix='kmeans'):
        """
        Load sklearn model for inference. After loading, use predict().
        """
        in_dir = results_dir or self.results_dir
        model_path = os.path.join(in_dir, f'{prefix}_model.joblib')
        scaler_path = os.path.join(in_dir, f'{prefix}_scaler.joblib')

        self.km = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.k = self.km.n_clusters
        self.model_loaded = True

        meta_path = os.path.join(in_dir, f'{prefix}_model_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.seed = meta.get('seed', 42)

        print(f'[KMeansClusterer] Loaded model (k={self.k}) from {in_dir}')

    def predict(self, X_new):
        """
        Predict cluster labels for new data.
        Requires load_model() OR fit()/fit_minibatch() first.

        Args:
            X_new: np.ndarray (M, 63), new hand pose vectors
        Returns:
            labels: np.ndarray (M,), cluster labels
        """
        # Check if model is available (from load_model or fit/fit_minibatch)
        has_model = (
            getattr(self, 'model_loaded', False) or
            (hasattr(self, 'km') and self.km is not None and hasattr(self, 'scaler') and self.scaler is not None)
        )
        if not has_model:
            raise RuntimeError('Must call load_model() or fit()/fit_minibatch() before predict()')

        X_new_scaled = self.scaler.transform(X_new)
        # Ensure dtype matches for sklearn
        if X_new_scaled.dtype != np.float64:
            X_new_scaled = X_new_scaled.astype(np.float64)
        return self.km.predict(X_new_scaled)
