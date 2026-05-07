#!/usr/bin/env python3
"""
run_pipeline.py — Super Cluster Pipeline CLI 入口

用於執行 SLBHS Phase 2 Super Cluster Pipeline（時序轉移相似度分析）。

用法：
    python run_pipeline.py --h5 /path/to/file.h5 --k 512 --tau 0.9
    python run_pipeline.py --h5 /path/to/file.h5 --k 512 --tau 0.9 --cosine-features --results-dir results
    python run_pipeline.py --h5 /path/to/file.h5 --skip-kmeans --results-dir results
"""

import argparse
import h5py
import logging
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SLBHS Super Cluster Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python run_pipeline.py --h5 /path/to/file.h5 --k 512 --tau 0.9
  python run_pipeline.py --folder /path/to/h5_folder --k 512 --tau 0.9 --cosine-features --results-dir results
  python run_pipeline.py --h5 /path/to/file.h5 --skip-kmeans --results-dir results
        """
    )
    parser.add_argument(
        '--h5',
        type=str,
        default=None,
        help='Path to H5 file (or glob pattern with *). '
             'Default: reads from data_dir.'
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Path to folder containing multiple H5 files. '
             'Batch processes all *.h5 files, computing S and BigClusterer at the end.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/ubuntu/.openclaw/media/inbound',
        help='Data directory containing H5 files. Default: /home/ubuntu/.openclaw/media/inbound'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results output directory. Default: results'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1024,
        help='K-Means number of clusters. Default: 1024'
    )
    parser.add_argument(
        '--tau',
        type=float,
        default=0.9,
        help='Similarity threshold (0.0-1.0). S_ij > tau → edge. Default: 0.9'
    )
    parser.add_argument(
        '--delta-t',
        type=int,
        default=10,
        help='Transition interval (n → n+delta_t). Default: 10'
    )
    parser.add_argument(
        '--min-transitions',
        type=int,
        default=0,
        help='Minimum transition count (below this, token pairs are zeroed). Default: 0'
    )
    parser.add_argument(
        '--symmetrize',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='Symmetrize transition matrix. Default: True'
    )
    parser.add_argument(
        '--cosine-features',
        action='store_true',
        help='Use 78D cosine features (63D scaled + 15D bone-angle) instead of raw 63D'
    )
    parser.add_argument(
        '--skip-kmeans',
        action='store_true',
        help='Skip K-Means prediction, use pre-computed labels.npy from results-dir'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed. Default: 42'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    parser.add_argument(
        '--save-only',
        action='store_true',
        help='Only run pipeline, skip verification plots'
    )
    return parser.parse_args()


def load_h5_data(h5_path):
    """Load data from a single H5 file or glob pattern."""
    import glob

    if h5_path is None:
        return None, None, None, None, None

    # Resolve glob patterns
    if '*' in h5_path or '?' in h5_path:
        files = sorted(glob.glob(h5_path))
        if not files:
            raise FileNotFoundError(f"No files match: {h5_path}")
        # For now, just use the first file (multi-file support can be added later)
        h5_path = files[0]
        logger.info(f"Using first file from glob: {h5_path}")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    logger.info(f"Loading H5: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        X = f['aligned_63d'][:].astype(np.float32)
        x_vec = f['x_vec'][:].astype(np.float32)
        y_vec = f['y_vec'][:].astype(np.float32)
        z_vec = f['z_vec'][:].astype(np.float32)

    logger.info(f"  X.shape={X.shape}, x_vec.shape={x_vec.shape}")

    # Try to load pre-computed labels
    labels = None
    return X, x_vec, y_vec, z_vec, labels


def load_from_results_dir(results_dir, X_shape):
    """Try to load pre-computed labels from results directory."""
    labels_path = os.path.join(results_dir, 'labels.npy')
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        logger.info(f"Loaded labels from {labels_path}: shape={labels.shape}, "
                    f"n_unique={len(np.unique(labels))}")
        return labels
    return None


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    if args.folder:
        # Folder mode: batch process all H5 files, finalize at the end
        h5_files = sorted(Path(args.folder).glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No *.h5 files found in: {args.folder}")
        logger.info(f"Found {len(h5_files)} H5 files in {args.folder}")

        pipeline = BigClusterPipeline(
            k=args.k,
            tau=args.tau,
            delta_t=args.delta_t,
            min_transitions=args.min_transitions,
            symmetrize=args.symmetrize,
            cosine_features=args.cosine_features,
            results_dir=args.results_dir,
        )

        for h5_path in h5_files:
            logger.info(f"  Processing {h5_path.name} ...")
            with h5py.File(h5_path) as f:
                X = f['aligned_63d'][:].astype(np.float32)
                x_vec = f['x_vec'][:].astype(np.float32)
                y_vec = f['y_vec'][:].astype(np.float32)
                z_vec = f['z_vec'][:].astype(np.float32)
            pipeline.update(X, x_vec, y_vec, z_vec)

        # Finalize: compute S + BigClusterer
        pipeline.finalize(tau=args.tau)

    elif args.h5:
        # Single H5 mode: use existing pipeline.fit()
        X, x_vec, y_vec, z_vec, labels = load_h5_data(args.h5)

        if args.skip_kmeans or X is None:
            labels = load_from_results_dir(args.results_dir, getattr(X, 'shape', None))
            if labels is None:
                raise ValueError("--skip-kmeans requires labels.npy in results-dir")

        if X is None or x_vec is None:
            raise ValueError("H5 data required for pipeline (provide --h5)")

        pipeline = BigClusterPipeline(
            k=args.k,
            tau=args.tau,
            delta_t=args.delta_t,
            min_transitions=args.min_transitions,
            symmetrize=args.symmetrize,
            cosine_features=args.cosine_features,
            results_dir=args.results_dir,
        )

        logger.info(f"Starting pipeline: k={args.k}, tau={args.tau}, "
                    f"cosine_features={args.cosine_features}, delta_t={args.delta_t}")

        pipeline.fit(
            X, x_vec, y_vec, z_vec,
            labels=labels if not args.skip_kmeans else None,
            k=args.k,
            tau=args.tau,
            cosine_features=args.cosine_features,
            min_transitions=args.min_transitions,
            delta_t=args.delta_t,
            symmetrize=args.symmetrize,
            results_dir=args.results_dir,
        )
    else:
        # Use DataLoader
        from SLBHS.data.loader import DataLoader
        loader = DataLoader(data_dir=args.data_dir)
        X, meta = loader.load()
        logger.info("DataLoader mode: results_dir must contain labels.npy")
        x_vec = y_vec = z_vec = None

        labels = load_from_results_dir(args.results_dir, getattr(X, 'shape', None))
        if labels is None:
            raise ValueError("labels.npy not found in results-dir")

        pipeline = BigClusterPipeline(
            k=args.k,
            tau=args.tau,
            delta_t=args.delta_t,
            min_transitions=args.min_transitions,
            symmetrize=args.symmetrize,
            cosine_features=args.cosine_features,
            results_dir=args.results_dir,
        )

        pipeline.fit(
            X, x_vec, y_vec, z_vec,
            labels=labels,
            k=args.k,
            tau=args.tau,
            cosine_features=args.cosine_features,
            min_transitions=args.min_transitions,
            delta_t=args.delta_t,
            symmetrize=args.symmetrize,
            results_dir=args.results_dir,
        )

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    pipeline.save(args.results_dir)

    # Summary
    n_clusters = pipeline.big_clusterer.n_clusters
    cluster_map = pipeline.big_clusterer.cluster_map

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully")
    logger.info(f"  k={pipeline._k_used}")
    logger.info(f"  tau={args.tau}")
    logger.info(f"  delta_t={args.delta_t}")
    logger.info(f"  cosine_features={pipeline.cosine_features}")
    logger.info(f"  N Super Clusters={n_clusters}")
    logger.info(f"  tokens in clusters={len(cluster_map)}")
    logger.info(f"  results saved to: {args.results_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())