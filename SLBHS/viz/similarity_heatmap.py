"""
Similarity Heatmap Visualization
=================================
Computes S[i,j] = cosine(M_prob[i] excluding i/j, M_prob[j] excluding i/j)
and renders a heatmap with magma colormap.

Usage:
    cd /home/ubuntu/.openclaw/workspace-coding/SLBHS && source venv/bin/activate
    python SLBHS/viz/similarity_heatmap.py --input /path/to/symmetrized_matrix.npy --output heatmap.png

Defaults:
    --input  : /home/ubuntu/.openclaw/media/inbound/symmetrized_matrix---dec21f33-b895-4377-a969-a4a6b7f7493c.npy
    --output : /home/ubuntu/.openclaw/media/inbound/similarity_heatmap.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

INBOUND = Path('/home/ubuntu/.openclaw/media/inbound')
DEFAULT_INPUT = INBOUND / 'symmetrized_matrix---dec21f33-b895-4377-a969-a4a6b7f7493c.npy'
DEFAULT_OUTPUT = INBOUND / 'similarity_heatmap.png'


def compute_similarity_matrix(W: np.ndarray) -> np.ndarray:
    """
    Compute S[i,j] = cosine(M_prob[i] excluding i/j, M_prob[j] excluding i/j).

    Step 1: Ensure symmetric (already symmetrized, but safety check)
    Step 2: Row-normalize → M_prob (probability matrix)
    Step 3: For each pair (i, j), exclude dimensions i and j, compute cosine

    Parameters
    ----------
    W : (k, k) symmetrized transition matrix

    Returns
    -------
    S : (k, k) cosine similarity matrix (diagonal = 1.0)
    """
    # Ensure symmetric
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0)

    # Row normalize → M_prob
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M_prob = W / row_sums

    k = M_prob.shape[0]
    S = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        for j in range(i + 1, k):
            mask = np.ones(k, dtype=bool)
            mask[i] = False
            mask[j] = False
            vec_i = M_prob[i][mask]
            vec_j = M_prob[j][mask]
            S[i, j] = S[j, i] = cosine_similarity([vec_i], [vec_j])[0, 0]

    np.fill_diagonal(S, 1.0)
    return S


def draw_heatmap(S: np.ndarray, output_path: Path, title: str = None) -> None:
    """
    Render S matrix as a heatmap using magma colormap.

    Parameters
    ----------
    S : (k, k) similarity matrix
    output_path : where to save the PNG
    title : optional title string
    """
    k = S.shape[0]
    vmin = max(S.min(), 0.0)  # similarity scores are >= 0
    vmax = min(S.max(), 1.0)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(S, cmap='magma', vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_title(title or f'Similarity Heatmap (k={k})', fontsize=14, color='#cccccc')
    ax.set_xlabel('Token ID', fontsize=11, color='#cccccc')
    ax.set_ylabel('Token ID', fontsize=11, color='#cccccc')
    ax.tick_params(axis='both', colors='#cccccc')

    # Color bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=11, color='#cccccc')
    cbar.ax.tick_params(colors='#cccccc')

    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Heatmap saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Similarity Heatmap Visualization')
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT),
                        help=f'Path to symmetrized matrix (.npy). Default: {DEFAULT_INPUT}')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help=f'Output PNG path. Default: {DEFAULT_OUTPUT}')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading W from: {input_path}")
    W = np.load(input_path)
    print(f"W shape: {W.shape}, dtype: {W.dtype}")
    print(f"W range: [{W.min():.4f}, {W.max():.4f}]")
    print(f"W symmetric: {np.allclose(W, W.T)}")

    print("\nComputing similarity matrix S (excluding i/j dimensions)...")
    S = compute_similarity_matrix(W)
    print(f"S shape: {S.shape}")
    print(f"S range: [{S.min():.6f}, {S.max():.6f}]")
    print(f"S NaN count: {np.isnan(S).sum()}")

    # Statistics
    i_idx, j_idx = np.triu_indices(S.shape[0], k=1)
    upper_tri = S[i_idx, j_idx]
    stats = {
        'min': float(S.min()),
        'max': float(S.max()),
        'mean': float(S.mean()),
        'std': float(S.std()),
        'median': float(np.median(upper_tri)),
    }
    print(f"\n=== S Statistics ===")
    print(f"  min    : {stats['min']:.6f}")
    print(f"  max    : {stats['max']:.6f}")
    print(f"  mean   : {stats['mean']:.6f}")
    print(f"  std    : {stats['std']:.6f}")
    print(f"  median : {stats['median']:.6f}")

    # Save S matrix
    S_path = output_path.parent / 'similarity_matrix_heatmap.npy'
    np.save(S_path, S)
    print(f"S matrix saved: {S_path}")

    # Draw heatmap
    title = f'Similarity Heatmap (k={S.shape[0]}) | min={stats["min"]:.4f}, max={stats["max"]:.4f}, mean={stats["mean"]:.4f}'
    draw_heatmap(S, output_path, title=title)

    print("\nDone.")
    return stats


if __name__ == '__main__':
    main()