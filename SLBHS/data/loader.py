"""
DataLoader: Load and cache aligned_63d data from HDF5 files.

Supports three modes:
    1. Single file : data_dir='/path/to/file_crop---xxx' (exact file)
    2. Glob pattern: data_dir='/dir/*_crop---*'          (all matching)
    3. Directory   : data_dir='/dir/'                    (all _crop---* files in dir)
"""
import numpy as np
import h5py
import glob
import os
import hashlib
import json


class DataLoader:
    """Load TWSLT hand pose data from HDF5 and cache as compressed numpy."""

    def __init__(self, data_dir=None, cache_dir=None):
        """
        Args:
            data_dir: str.
                - Path ending in _crop---*  : treated as exact file path
                - Contains wildcard *        : treated as glob pattern
                - Otherwise                  : treated as directory → load all *_crop---* files
              Default: ~/.openclaw/media/inbound/
            cache_dir: str, where to store compressed cache.
        """
        if data_dir is None:
            data_dir = '/home/ubuntu/.openclaw/media/inbound'
        self.data_dir = data_dir
        # Default cache: <data_dir>/.cache/
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(data_dir.rstrip('/'))), '.cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _find_h5_files(self):
        """
        Return list of H5 file paths based on self.data_dir:
            - If path has no wildcard and is a file → [path]
            - If path has wildcard (* or ?)           → glob results
            - Otherwise                              → all *_crop---* in directory
        """
        d = self.data_dir

        # Has shell wildcard → glob
        if '*' in d or '?' in d:
            files = sorted(glob.glob(d))
            if not files:
                raise FileNotFoundError(f'No files match glob: {d}')
            return files

        # Exact file path (ends with _crop---...)
        if os.path.isfile(d):
            return [d]

        # Directory → all *_crop---* files inside
        pattern = os.path.join(d, '*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f'No h5 files found in directory: {d}')
        return files

    def _cache_path(self, file_list):
        """Return persistent cache path based on all source files."""
        # Sort for deterministic order
        names = '|'.join(sorted(os.path.basename(f) for f in file_list))
        digest = hashlib.md5(names.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f'aligned_63d_multi_{digest}.npz')

    def load(self, force_reload=False):
        """
        Load aligned_63d from one or more H5 files and concatenate.
        Caches per unique set of source files.

        Returns:
            X: np.ndarray of shape (N, 63), float32
            meta: dict with keys:
                - 'n_frames': total frame count
                - 'files': list of source file paths
                - 'n_files': number of files
                - 'per_file_frames': list of frame counts per file
        """
        files = self._find_h5_files()
        cache_path = self._cache_path(files)

        if not force_reload and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            X = data['X']
            meta = dict(data['meta'].item())
            print(f'[DataLoader] Loaded from cache: {X.shape} ({meta["n_files"]} files)')
            return X, meta

        print(f'[DataLoader] Reading {len(files)} file(s)...')
        X_list = []
        per_file_frames = []

        for fp in files:
            print(f'  {os.path.basename(fp)} ...')
            with h5py.File(fp, 'r') as f:
                Xi = f['aligned_63d'][:].astype(np.float32)
                has_mirror = 'is_mirror' in f

            if has_mirror:
                is_mirror = f['is_mirror'][:]
                n_mirror = int(is_mirror.sum())
                Xi = Xi[~is_mirror]
                print(f'    {f["aligned_63d"].shape[0]} frames, '
                      f'filtered {n_mirror} mirrored → {Xi.shape[0]}')
            else:
                print(f'    {Xi.shape[0]} frames')

            X_list.append(Xi)
            per_file_frames.append(len(Xi))

        X = np.concatenate(X_list, axis=0)
        meta = {
            'n_frames': len(X),
            'files': files,
            'n_files': len(files),
            'per_file_frames': per_file_frames,
        }

        np.savez_compressed(cache_path, X=X, meta=np.array([meta], dtype=object))
        print(f'[DataLoader] Total: {X.shape}, cached to {cache_path}')

        return X, meta
