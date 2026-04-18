#!/usr/bin/env python3
"""
Generate sample images for each cluster.
Run from project root or as module: python -m SLBHS.viz.gen_samples --help

Usage:
    python -m SLBHS.viz.gen_samples --k 1024 --samples-per-cluster 10
    python -m SLBHS.viz.gen_samples --results-dir results --k 1024 --samples-per-cluster 10
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile


# MediaPipe hand landmark connections
MEDIAPIPE_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate sample images for each cluster')
    parser.add_argument('--k', type=int, default=1024, help='Number of clusters (k-means k)')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    parser.add_argument('--samples-per-cluster', type=int, default=10, help='Number of samples per cluster')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-zip', action='store_true', help='Create ZIP archive after generation')
    return parser.parse_args()


def generate_samples(k, results_dir, samples_per_cluster=10, seed=42, create_zip=False):
    """Generate sample images for each cluster."""
    
    # Load results
    labels = np.load(os.path.join(results_dir, 'labels.npy'))
    centers = np.load(os.path.join(results_dir, 'centers.npy'))
    data = np.load(os.path.join(results_dir, 'aligned_63d_multi_ddbdc3ed24df9eba.npz'), allow_pickle=True)
    X = data['X']

    print(f"Labels: {len(labels)}, X: {X.shape}, Centers: {centers.shape}")

    # Output directory
    output_dir = Path(results_dir) / f'clusters_{samples_per_cluster}samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing files
    existing = set()
    for f in output_dir.glob('*.png'):
        try:
            existing.add(int(f.stem.split('_')[1]))
        except:
            pass

    print(f"Existing: {len(existing)}/{k} clusters")
    
    # Random seed for reproducibility
    np.random.seed(seed)
    
    # Generate samples for each cluster
    for cluster_id in range(k):
        if cluster_id in existing:
            continue
            
        cluster_indices = np.where(labels == cluster_id)[0]
        n_samples = min(samples_per_cluster, len(cluster_indices))
        if n_samples == 0:
            # Create empty placeholder
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.text(0.5, 0.5, f'Cluster {cluster_id}\n(No samples)', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            plt.suptitle(f'Cluster {cluster_id} (0 frames)', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / f'cluster_{cluster_id:04d}.png', dpi=80, bbox_inches='tight')
            plt.close()
            continue
        
        # Random sample without replacement
        sample_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        
        # Calculate grid size
        n_rows = (n_samples + 4) // 5  # Ceiling division
        fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            hand_pose = X[idx].reshape(21, 3)
            x = hand_pose[:, 0]
            y = -hand_pose[:, 1]  # Flip Y for image coordinates
            
            for j, k_conn in MEDIAPIPE_CONNECTIONS:
                ax.plot([x[j], x[k_conn]], [y[j], y[k_conn]], 'b-', linewidth=2)
            
            ax.scatter(x, y, c='red', s=20)
            ax.set_title(f'Sample {i+1}')
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Cluster {cluster_id} ({len(cluster_indices)} frames)', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(output_dir / f'cluster_{cluster_id:04d}.png', dpi=80, bbox_inches='tight')
        plt.close()
        
        if cluster_id % 100 == 0:
            print(f"Processed {cluster_id}/{k} clusters")
    
    print(f"Done: {len(list(output_dir.glob('*.png')))} images in {output_dir}")
    
    # Create ZIP archive if requested
    if create_zip:
        output_zip = Path(results_dir) / f'clusters_{k}_{samples_per_cluster}samples.zip'
        with zipfile.ZipFile(output_zip, 'w') as zf:
            for f in sorted(output_dir.glob('*.png')):
                zf.write(f, f.name)
        print(f"Created {output_zip}")


def main():
    args = parse_args()
    
    # Get results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Default to results in current directory or SLBHS package
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    generate_samples(
        k=args.k,
        results_dir=results_dir,
        samples_per_cluster=args.samples_per_cluster,
        seed=args.seed,
        create_zip=args.output_zip
    )


if __name__ == '__main__':
    main()