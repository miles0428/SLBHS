#!/usr/bin/env python3
"""
Generate 10 sample images for each of 1024 clusters.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
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

def main():
    # Load results
    labels = np.load('results/labels.npy')
    centers = np.load('results/centers.npy')
    data = np.load('results/aligned_63d_multi_ddbdc3ed24df9eba.npz', allow_pickle=True)
    X = data['X']

    print(f"Labels: {len(labels)}, X: {X.shape}, Centers: {centers.shape}")

    # Check existing files
    output_dir = Path('results/clusters_10samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing = set()
    for f in output_dir.glob('*.png'):
        try:
            existing.add(int(f.stem.split('_')[1]))
        except:
            pass

    print(f"Existing: {len(existing)}/1024")
    
    # Continue from where it left off
    np.random.seed(42)
    
    for cluster_id in range(len(centers)):
        if cluster_id in existing:
            continue
            
        cluster_indices = np.where(labels == cluster_id)[0]
        n_samples = min(10, len(cluster_indices))
        if n_samples == 0:
            continue
        
        sample_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            hand_pose = X[idx].reshape(21, 3)
            x = hand_pose[:, 0]
            y = -hand_pose[:, 1]
            
            for j, k in MEDIAPIPE_CONNECTIONS:
                ax.plot([x[j], x[k]], [y[j], y[k]], 'b-', linewidth=2)
            
            ax.scatter(x, y, c='red', s=20)
            ax.set_title(f'Sample {i+1}')
            ax.set_aspect('equal')
            ax.axis('off')
        
        for i in range(n_samples, 10):
            axes[i].axis('off')
        
        plt.suptitle(f'Cluster {cluster_id} ({len(cluster_indices)} frames)', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(f'results/clusters_10samples/cluster_{cluster_id:04d}.png', dpi=80, bbox_inches='tight')
        plt.close()
        
        if cluster_id % 100 == 0:
            print(f"Processed {cluster_id}/1024 clusters")
    
    print("Done creating images")
    
    # Create zip
    output_zip = 'results/clusters_1024_10samples.zip'
    with zipfile.ZipFile(output_zip, 'w') as zf:
        for f in sorted(output_dir.glob('*.png')):
            zf.write(f, f.name)
    
    print(f"Created {output_zip}")

if __name__ == '__main__':
    main()