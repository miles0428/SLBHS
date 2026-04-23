"""
TWSLT Visualization Pipeline — Example usage.

Run:
    python run_visualization.py --k 512 --n-super 20 --dpi 200 --format png

Full pipeline:
    1. DataLoader  → load H5 data
    2. KMeansClusterer → fit/save k-means
    3. SuperClusterer  → hierarchical super clusters
    4. UMAPReducer     → UMAP (cached)
    5. SLBHSViz        → plot + save
"""
import argparse
import os
import sys
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .data.loader import DataLoader
from .clustering.kmeans import KMeansClusterer
from .clustering.feature_transform import compute_cosine_features
from .clustering.super_cluster import SuperClusterer
from .clustering.reducer import UMAPReducer
from .viz.visualizer import SLBHSViz
from .viz.plot_config import KMEANS_K, KMEANS_SEED, N_SUPER, UMAP_OVERVIEW_N, UMAP_SC_N


def parse_args():
    parser = argparse.ArgumentParser(description='TWSLT visualization pipeline')
    parser.add_argument('--k', type=int, default=KMEANS_K, help='K-Means k')
    parser.add_argument('--n-super', type=int, default=N_SUPER, help='Number of super clusters')
    parser.add_argument('--seed', type=int, default=KMEANS_SEED, help='Random seed')
    parser.add_argument('--data-dir', type=str, default=None, help='H5 data directory')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    parser.add_argument('--dpi', type=int, default=200, help='PNG DPI')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'svg', 'both'])
    parser.add_argument('--batch-size', type=int, default=5000, help='MiniBatch K-Means batch size')
    parser.add_argument('--skip-kmeans', action='store_true', help='Skip K-Means (load existing)')
    parser.add_argument('--skip-super', action='store_true', help='Skip super clustering')
    parser.add_argument('--skip-umap', action='store_true', help='Skip UMAP computation')
    parser.add_argument('--n-neighbors', type=int, default=30, help='UMAP n_neighbors')
    parser.add_argument('--overview-umap-n', type=int, default=UMAP_OVERVIEW_N, help='UMAP overview sample size')
    parser.add_argument('--sc-umap-n', type=int, default=UMAP_SC_N, help='UMAP per-supercluster sample size')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Suppress K-Means progress')
    parser.add_argument('--cosine-features', action='store_true', help='Use combined features: 63d scaled raw coordinates plus 15-dim cosine similarity features (78d total)')
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ---- 1. Load data ----
    print('=== Step 1: Load data ===')
    loader = DataLoader(data_dir=args.data_dir, cache_dir=results_dir)
    X, meta = loader.load()
    print(f'  Loaded {X.shape[0]} frames, {X.shape[1]} dims')

    # ---- 2. K-Means ----
    if args.skip_kmeans:
        print('=== Step 2: Load existing K-Means ===')
        kc = KMeansClusterer(results_dir=results_dir)
        kc.load()
    else:
        kc = KMeansClusterer(results_dir=results_dir)
        if args.cosine_features:
            from sklearn.preprocessing import StandardScaler
            scaler_63d = StandardScaler()
            X_scaled_63d = scaler_63d.fit_transform(X)
            X_cosine_15d = compute_cosine_features(X, verbose=True)
            X_combined = np.hstack([X_scaled_63d, X_cosine_15d])
            print(f'=== Step 2: Cosine+Raw MiniBatch K-Means k={args.k} input_dim={X_combined.shape[1]} ===')
            kc.fit_cosine_minibatch(k=args.k, seed=args.seed, X_combined=X_combined, scaler=scaler_63d, batch_size=args.batch_size, verbose_progress=args.verbose)
        else:
            print(f'=== Step 2: MiniBatch K-Means k={args.k} batch_size={args.batch_size} ===')
            kc.fit_minibatch(k=args.k, seed=args.seed, X=X, batch_size=args.batch_size, verbose_progress=args.verbose)
        kc.save()
        kc.save_model()

    labels = kc.labels_
    centers = kc.centers_
    print(f'  K-Means done: {len(labels)} labels, {len(centers)} centers')

    # ---- 3. Super Clustering ----
    if args.skip_super:
        print('=== Step 3: Load existing Super Clusters ===')
        sc = SuperClusterer(kmeans_labels=labels, kmeans_centers=centers,
                            results_dir=results_dir)
        sc.load()
    else:
        print(f'=== Step 3: Super Clustering n={args.n_super} ===')
        sc = SuperClusterer(kmeans_labels=labels, kmeans_centers=centers,
                            results_dir=results_dir)
        sc.fit(n_super=args.n_super)
        sc.save()

    frame_super = sc.frame_super_
    super_labels = sc.super_labels_
    print(f'  Super clusters done: {sc.n_super_} super clusters')

    # ---- 4. Scale X + Build UMAP Reducer ----
    print('=== Step 4: Scale data ===')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    del X

    # ---- 5. UMAP ----
    if args.skip_umap:
        print('=== Step 5: Skip UMAP ===')
        overview_umap = None
        overview_labels = None
        sc_umaps = {}
        # X_scaled no longer needed — free it
        del X_scaled
    else:
        print(f'=== Step 5: UMAP (overview n={args.overview_umap_n}, sc n={args.sc_umap_n}) ===')
        reducer = UMAPReducer(X_scaled, super_labels=frame_super, cache_dir=results_dir)
        overview_umap, ov_idx = reducer.transform_overview(n=args.overview_umap_n, seed=args.seed, n_neighbors=args.n_neighbors)
        overview_labels = frame_super[ov_idx]
        sc_umaps = {}
        for s in range(args.n_super):
            sc_umap, sc_umap_idx = reducer.transform_sc(sc_id=s, n=args.sc_umap_n, seed=args.seed, n_neighbors=args.n_neighbors)
            sc_mask = frame_super == s
            sc_indices = np.where(sc_mask)[0]
            sc_frame_labels = labels[sc_indices[sc_umap_idx]]
            sc_umaps[s] = (sc_umap, sc_frame_labels)
            print(f'  SC {s}: {len(sc_umap)} UMAP points')
        # Free X_scaled and reducer immediately — not needed after UMAP
        del X_scaled, reducer

    # ---- 6. Visualize ----
    print('=== Step 6: Visualize ===')
    viz = SLBHSViz(
        kmeans_labels=labels, kmeans_centers=centers,
        frame_super=frame_super, super_labels=super_labels,
        kmeans_meta={'k': args.k, 'seed': args.seed},
        super_meta={'n_super': args.n_super},
    )
    viz.plot(overview_umap=overview_umap, overview_labels=overview_labels, sc_umaps=sc_umaps)

    if args.format in ('png', 'both'):
        png_path = os.path.join(results_dir, f'twslt_k{args.k}_super{args.n_super}.png')
        viz.save_png(png_path, dpi=args.dpi)

    if args.format in ('svg', 'both'):
        svg_path = os.path.join(results_dir, f'twslt_k{args.k}_super{args.n_super}.svg')
        viz.save_svg(svg_path)

    print('=== Done ===')


if __name__ == '__main__':
    main()
