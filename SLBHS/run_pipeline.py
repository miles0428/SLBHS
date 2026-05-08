#!/usr/bin/env python3
"""
run_pipeline.py — Super Cluster Pipeline CLI 入口（v3）

完全吃 K-Means model，不碰 training，只做：
    H5 → predict → C → S → SuperCluster

用法：
    # 批次模式（多個 H5）：走 update() + finalize()
    python run_pipeline.py \
        --folder /path/to/h5/folder \
        --model-dir /path/to/kmeans/model/ \
        --delta-t 10 \
        --tau 0.9 \
        --output results/

    # 單一 H5 模式（debug）：走 fit()
    python run_pipeline.py \
        --h5 /path/to/single.h5 \
        --model-dir /path/to/kmeans/model/ \
        --delta-t 10 \
        --tau 0.9 \
        --output results/

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
        description="SLBHS Super Cluster Pipeline v3 — 吃 K-Means model，不 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 批次模式（多個 H5）
  python run_pipeline.py --folder /path/to/h5/folder --model-dir /path/to/kmeans/model/ --delta-t 10 --tau 0.9 --output results/

  # 單一 H5 模式
  python run_pipeline.py --h5 /path/to/file.h5 --model-dir /path/to/kmeans/model/ --delta-t 10 --tau 0.9 --output results/

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
        default=None,
        help='KMeans model directory containing kmeans_model.joblib + kmeans_scaler.joblib. '
             'Pipeline only loads model (no training). Required when using --folder.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Results output directory. Default: results'
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

    # Validate model_dir
    if args.model_dir is None:
        raise ValueError(
            "--model-dir is required. MiniBatchKMeans fallback is not implemented "
            "by BigClusterPipeline.fit()/update(), so the CLI cannot run without a "
            "pre-trained model directory."
        )

    # Common parameters
    common_kwargs = dict(
        tau=args.tau,
        delta_t=args.delta_t,
        min_transitions=args.min_transitions,
        symmetrize=args.symmetrize,
        model_dir=args.model_dir,
        results_dir=args.output,
    )

    os.makedirs(args.output, exist_ok=True)

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

        # Finalize: compute S + BigClusterer
        pipeline.finalize(tau=args.tau)

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

        logger.info(f"Starting pipeline: tau={args.tau}, "
                    f"model_dir={args.model_dir}, delta_t={args.delta_t}")

        pipeline.fit(
            X, x_vec, y_vec, z_vec,
            tau=args.tau,
            min_transitions=args.min_transitions,
            delta_t=args.delta_t,
            symmetrize=args.symmetrize,
            model_dir=args.model_dir,
            results_dir=args.output,
        )

    # Save results
    pipeline.save(args.output)

    # Summary
    n_clusters = pipeline.big_clusterer.n_clusters
    cluster_map = pipeline.big_clusterer.cluster_map

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully")
    logger.info(f"  k={pipeline._k_used}")
    logger.info(f"  tau={args.tau}")
    logger.info(f"  delta_t={args.delta_t}")
    logger.info(f"  model_dir={args.model_dir}")
    logger.info(f"  N Super Clusters={n_clusters}")
    logger.info(f"  tokens in clusters={len(cluster_map)}")
    logger.info(f"  results saved to: {args.output}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
