#!/usr/bin/env python3
"""
run_pipeline.py — Super Cluster Pipeline CLI (v4)

Simplified pipeline: K-Means → Transition Matrix → Cosine Similarity → (Optional) SuperClusterer
BigClusterer removed.

用法：
    # 批次模式（多個 H5）
    python run_pipeline.py --folder /path/to/h5/folder --model-dir /path/to/kmeans/model/ --output results/

    # 單一 H5 模式
    python run_pipeline.py --h5 /path/to/file.h5 --model-dir /path/to/kmeans/model/ --output results/

    # 加上 SuperClusterer（n_super 群）
    python run_pipeline.py --h5 /path/to/file.h5 --model-dir /path/to/kmeans/model/ --output results/ --n-super 20

    # 詳細輸出
    python run_pipeline.py --folder /path/to/h5/folder --model-dir /path/to/kmeans/model/ -v
"""

import argparse
import h5py
import logging
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)
# Keep chunk size bounded to avoid loading entire large H5 datasets into RAM at once.
H5_CHUNK_SIZE = 200_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="SLBHS Super Cluster Pipeline v4 — K-Means + Cosine Similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 基本模式
  python run_pipeline.py --folder /path/to/h5/folder --model-dir /path/to/kmeans/model/ --output results/

  # 加 SuperClusterer
  python run_pipeline.py --h5 /path/to/file.h5 --model-dir /path/to/kmeans/model/ --output results/ --n-super 20

  # 詳細輸出
  python run_pipeline.py --folder /path/to/h5/folder --model-dir /path/to/kmeans/model/ -v
        """
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Path to folder containing multiple H5 files. '
             'Batch processes all *.h5 files via update()+finalize().'
    )
    parser.add_argument(
        '--h5',
        type=str,
        default=None,
        help='Path to a single H5 file. Uses fit() directly.'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='KMeans model directory containing kmeans_model.joblib + kmeans_scaler.joblib.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Results output directory. Default: results'
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
        '--n-super',
        type=int,
        default=None,
        help='Number of super clusters for SuperClusterer (Agglomerative Hierarchical). '
             'If not specified, SuperClusterer is skipped.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.folder and not args.h5:
        raise ValueError("Must specify either --folder or --h5")

    if args.folder and args.h5:
        raise ValueError("Cannot specify both --folder and --h5. Use one or the other.")

    os.makedirs(args.output, exist_ok=True)

    # Common parameters
    common_kwargs = dict(
        delta_t=args.delta_t,
        min_transitions=args.min_transitions,
        symmetrize=args.symmetrize,
        model_dir=args.model_dir,
        results_dir=args.output,
        n_super=args.n_super,
    )

    if args.folder:
        # ── 批次模式（多個 H5）：走 update() + finalize() ──
        h5_files = sorted(Path(args.folder).glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No *.h5 files found in: {args.folder}")
        logger.info(f"Found {len(h5_files)} H5 files in {args.folder}")

        pipeline = BigClusterPipeline(**common_kwargs)

        for h5_path in h5_files:
            logger.info(f"  Processing {h5_path.name} ...")
            with h5py.File(h5_path) as f:
                total = f["aligned_63d"].shape[0]
                for start in range(0, total, H5_CHUNK_SIZE):
                    end = min(start + H5_CHUNK_SIZE, total)
                    X = f['aligned_63d'][start:end].astype(np.float32)
                    x_vec = f['x_vec'][start:end].astype(np.float32)
                    y_vec = f['y_vec'][start:end].astype(np.float32)
                    z_vec = f['z_vec'][start:end].astype(np.float32)
                    pipeline.update(X, x_vec, y_vec, z_vec)

        # Finalize: compute S
        pipeline.finalize()

    else:
        # ── 單一 H5 模式：走 fit()（直接完成）──
        h5_path = args.h5
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

        logger.info(f"Loading H5: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            X = f['aligned_63d'][:].astype(np.float32)
            x_vec = f['x_vec'][:].astype(np.float32)
            y_vec = f['y_vec'][:].astype(np.float32)
            z_vec = f['z_vec'][:].astype(np.float32)
        logger.info(f"  X.shape={X.shape}")

        pipeline = BigClusterPipeline(**common_kwargs)

        logger.info(f"Starting pipeline: "
                    f"model_dir={args.model_dir}, delta_t={args.delta_t}, n_super={args.n_super}")

        pipeline.fit(
            X, x_vec, y_vec, z_vec,
            min_transitions=args.min_transitions,
            delta_t=args.delta_t,
            symmetrize=args.symmetrize,
            model_dir=args.model_dir,
            results_dir=args.output,
        )

    # Save results
    pipeline.save(args.output)

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully")
    logger.info(f"  k={pipeline._k_used}")
    logger.info(f"  delta_t={args.delta_t}")
    logger.info(f"  symmetrize={args.symmetrize}")
    logger.info(f"  model_dir={args.model_dir}")
    if args.n_super is not None:
        logger.info(f"  n_super={args.n_super}")
        logger.info(f"  n_clusters={pipeline._n_clusters}")
    logger.info(f"  S.shape={pipeline.similarity_matrix.S.shape}")
    logger.info(f"  S_nan={int(np.sum(np.isnan(pipeline.similarity_matrix.S)))}")
    logger.info(f"  results saved to: {args.output}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())