#!/usr/bin/env python3
"""
run_compute_similarity.py — Compute Cosine Similarity Pipeline

K-Means → Transition Matrix → Cosine Similarity → (Optional) SuperClusterer
One-line CLI. BigClusterer removed.

用法：
    # 單一 H5 模式
    python -m SLBHS.run_compute_similarity \
        --h5 /path/to/file.h5 \
        --model-dir /path/to/kmeans/model/ \
        --output results/

    # 批次模式（多個 H5）
    python -m SLBHS.run_compute_similarity \
        --folder /path/to/h5/folder \
        --model-dir /path/to/kmeans/model/ \
        --output results/

    # 加上 SuperClusterer（Hierarchical Clustering for visualization）
    python -m SLBHS.run_compute_similarity \
        --h5 /path/to/file.h5 \
        --model-dir /path/to/kmeans/model/ \
        --output results/ \
        --n-super 20
"""

import argparse
import h5py
import logging
import numpy as np
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SLBHS.clustering.super_cluster_pipeline import (
    HandLabeler,
    TransitionCounter,
    SimilarityMatrix,
)
from SLBHS.clustering.super_cluster import SuperClusterer
from SLBHS.clustering.kmeans import KMeansClusterer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Keep chunk size bounded to avoid loading entire large H5 datasets into RAM at once.
H5_CHUNK_SIZE = 200_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="SLBHS Compute Similarity Pipeline — K-Means → Cosine Similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 基本模式（只算 Similarity Matrix）
  python -m SLBHS.run_compute_similarity --h5 /path/to/file.h5 --model-dir /path/to/model/ --output results/

  # 加上 SuperClusterer（Hierarchical，n_super 群）
  python -m SLBHS.run_compute_similarity --h5 /path/to/file.h5 --model-dir /path/to/model/ --output results/ --n-super 20

  # 批次模式
  python -m SLBHS.run_compute_similarity --folder /path/to/h5/folder --model-dir /path/to/model/ --output results/ --n-super 20
        """
    )
    parser.add_argument(
        '--h5',
        type=str,
        default=None,
        help='Path to a single H5 file.'
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Path to folder containing multiple H5 files. '
             'Batch processes all *.h5 files via update()+finalize().'
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


def run_single_h5(h5_path, model_dir, output_dir, delta_t, min_transitions, symmetrize, n_super, logger):
    """Process a single H5 file."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    logger.info(f"Loading H5: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        X = f['aligned_63d'][:].astype(np.float32)
        x_vec = f['x_vec'][:].astype(np.float32)
        y_vec = f['y_vec'][:].astype(np.float32)
        z_vec = f['z_vec'][:].astype(np.float32)
    logger.info(f"  X.shape={X.shape}")

    # Step 1: HandLabeler
    hand_labeler = HandLabeler()
    hand_labels = hand_labeler.fit_predict(x_vec, y_vec, z_vec)
    logger.info(f"[Pipeline] HandLabeler: L={int(np.sum(hand_labels=='L'))} "
                f"R={int(np.sum(hand_labels=='R'))}")

    # Step 2: KMeans predict
    logger.info(f"[Pipeline] KMeansClusterer.load_model from {model_dir}")
    kc = KMeansClusterer(results_dir=model_dir)
    kc.load_model(model_dir)
    labels = kc.predict(X)
    k = kc.k
    logger.info(f"[Pipeline] K-Means k={k}, labels.shape={labels.shape}")

    # Step 3: TransitionCounter
    transition_counter = TransitionCounter(k=k, delta_t=delta_t, min_transitions=min_transitions)
    transition_counter.fit(labels, hand_labels, delta_t=delta_t, min_transitions=min_transitions)
    C = transition_counter.get_matrix()
    logger.info(f"[Pipeline] TransitionCounter: C.shape={C.shape}, nnz={int(np.sum(C > 0))}")

    # Step 4: SimilarityMatrix
    similarity_matrix = SimilarityMatrix()
    S = similarity_matrix.compute(C, symmetrize=symmetrize)
    logger.info(f"[Pipeline] SimilarityMatrix: S.shape={S.shape}, S_nan={int(np.sum(np.isnan(S)))}")

    # Step 5: SuperClusterer (optional, Hierarchical Clustering on K-Means centers)
    super_labels = None
    if n_super is not None:
        logger.info(f"[Pipeline] SuperClusterer: n_super={n_super}")
        # Get K-Means centers from the loaded model
        centers = kc.km.cluster_centers_
        sc = SuperClusterer(kmeans_labels=labels, kmeans_centers=centers)
        super_labels, frame_super = sc.fit(n_super=n_super, linkage='ward')
        logger.info(f"[Pipeline] SuperClusterer done: {n_super} super clusters")

    return {
        'k': k,
        'C': C,
        'S': S,
        'labels': labels,
        'super_labels': super_labels,
        'transition_counter': transition_counter,
        'similarity_matrix': similarity_matrix,
    }


def run_batch(folder, model_dir, output_dir, delta_t, min_transitions, symmetrize, n_super, logger):
    """Process multiple H5 files in batch mode."""
    h5_files = sorted(Path(folder).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No *.h5 files found in: {folder}")
    logger.info(f"Found {len(h5_files)} H5 files in {folder}")

    # Load KMeans model once
    logger.info(f"[Batch] Loading KMeans model from {model_dir}")
    kc = KMeansClusterer(results_dir=model_dir)
    kc.load_model(model_dir)
    k = kc.k
    logger.info(f"[Batch] K-Means k={k}")

    # Initialize pipeline components
    hand_labeler = HandLabeler()
    transition_counter = TransitionCounter(k=k, delta_t=delta_t, min_transitions=min_transitions)
    _kmeans_clusterer = kc  # keep reference to avoid GC

    # Process each H5 file in chunks
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

                # HandLabeler
                hand_labels = hand_labeler.fit_predict(x_vec, y_vec, z_vec)

                # KMeans predict
                labels = kc.predict(X)

                # Accumulate into TransitionCounter
                transition_counter.update(labels, hand_labels)

        logger.info(f"  {h5_path.name}: done, C.nnz={int(np.sum(transition_counter.C > 0))}")

    # Compute SimilarityMatrix
    C = transition_counter.get_matrix()
    similarity_matrix = SimilarityMatrix()
    S = similarity_matrix.compute(C, symmetrize=symmetrize)
    logger.info(f"[Batch] SimilarityMatrix: S.shape={S.shape}, S_nan={int(np.sum(np.isnan(S)))}")

    # SuperClusterer (optional)
    super_labels = None
    if n_super is not None:
        logger.info(f"[Batch] SuperClusterer: n_super={n_super}")
        # Need to accumulate labels for all data to get centers
        # Re-run to get labels for centers - this is a limitation of batch mode
        # For simplicity, we skip centers-based SuperClusterer in batch mode
        logger.warning(f"[Batch] SuperClusterer not supported in batch mode (would need full labels). Skipping.")
        super_labels = None

    return {
        'k': k,
        'C': C,
        'S': S,
        'labels': None,  # not stored in batch mode
        'super_labels': super_labels,
        'transition_counter': transition_counter,
        'similarity_matrix': similarity_matrix,
        '_kmeans_clusterer': _kmeans_clusterer,
    }


def save_results(result, output_dir, n_super, logger):
    """Save pipeline outputs to output_dir."""
    results_path = Path(output_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    k = result['k']
    S = result['S']
    C = result['C']

    # Save similarity matrix
    np.save(results_path / "similarity_matrix.npy", S)
    logger.info(f"[Output] similarity_matrix.npy saved: {S.shape}")

    # Save transition matrix
    np.save(results_path / "transition_matrix.npy", C)
    logger.info(f"[Output] transition_matrix.npy saved: {C.shape}")

    # Save symmetrized matrix
    W = (C + C.T) / 2.0
    np.save(results_path / "symmetrized_matrix.npy", W)
    logger.info(f"[Output] symmetrized_matrix.npy saved: {W.shape}")

    # Save super cluster labels if available
    if result['super_labels'] is not None:
        super_labels = result['super_labels']
        np.save(results_path / "super_cluster_labels.npy", super_labels)
        logger.info(f"[Output] super_cluster_labels.npy saved: {super_labels.shape}")

        # Save super cluster meta
        meta = {
            'n_super': n_super,
            'n_centers': k,
            'linkage': 'ward',
        }
        with open(results_path / "super_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[Output] super_meta.json saved")

    # Save pipeline summary
    summary = {
        'phase': 'similarity',
        'k': k,
        'delta_t': result.get('delta_t', 'N/A'),
        'min_transitions': result.get('min_transitions', 0),
        'symmetrize': result.get('symmetrize', True),
        'n_super': n_super,
        'transition_matrix': {
            'shape': list(C.shape),
            'nnz': int(np.sum(C > 0)),
            'max': float(C.max()),
        },
        'similarity_matrix': {
            'shape': list(S.shape),
            'nan_count': int(np.sum(np.isnan(S))),
            'min': float(S.min()),
            'max': float(S.max()),
        },
    }
    with open(results_path / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"[Output] pipeline_summary.json saved")


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate
    if not args.h5 and not args.folder:
        raise ValueError("Must specify either --h5 or --folder")
    if args.h5 and args.folder:
        raise ValueError("Cannot specify both --h5 and --folder. Use one or the other.")
    if not args.model_dir:
        raise ValueError("--model-dir is required")

    os.makedirs(args.output, exist_ok=True)

    common_kwargs = dict(
        delta_t=args.delta_t,
        min_transitions=args.min_transitions,
        symmetrize=args.symmetrize,
        model_dir=args.model_dir,
    )

    if args.h5:
        # Single H5 mode
        result = run_single_h5(
            h5_path=args.h5,
            output_dir=args.output,
            n_super=args.n_super,
            logger=logger,
            **common_kwargs
        )
    else:
        # Batch mode
        result = run_batch(
            folder=args.folder,
            output_dir=args.output,
            n_super=args.n_super if args.h5 else None,  # no SuperClusterer in batch mode
            logger=logger,
            **common_kwargs
        )

    # Save outputs
    result['delta_t'] = args.delta_t
    result['min_transitions'] = args.min_transitions
    result['symmetrize'] = args.symmetrize
    save_results(result, args.output, args.n_super, logger)

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully")
    logger.info(f"  k={result['k']}")
    logger.info(f"  delta_t={args.delta_t}")
    logger.info(f"  symmetrize={args.symmetrize}")
    logger.info(f"  S.shape={result['S'].shape}")
    logger.info(f"  S_nan={int(np.sum(np.isnan(result['S'])))}")
    if args.n_super is not None:
        logger.info(f"  n_super={args.n_super}")
    logger.info(f"  results saved to: {args.output}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())