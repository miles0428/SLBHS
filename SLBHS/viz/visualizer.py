"""
visualizer.py — SLBHSViz: main plotting class for TWSLT UMAP + super cluster visualizations.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from .layout import GridLayout
from .plot_config import (
    SUPER_CMAP, CLUSTER_CMAP,
    SCATTER_SIZE_OVERVIEW, SCATTER_ALPHA_OVERVIEW,
    SCATTER_SIZE_SC, SCATTER_ALPHA_SC,
    TITLE_SIZE, AXIS_LABEL_SIZE, SC_TITLE_SIZE,
    FIG_WIDTH, FIG_HEIGHT,
)


class SLBHSViz:
    """
    Main visualization class for TWSLT data.

    Usage:
        viz = SLBHSViz(
            X=aligned_63d,                     # (N, 63)
            kmeans_labels=labels,               # (N,) cluster labels
            kmeans_centers=centers,             # (k, 63)
            super_labels=super_labels,          # (N,) or (k,) super cluster per frame
        )
        viz.plot()
        viz.save_svg('/tmp/out.svg')
        viz.save_png('/tmp/out.png', dpi=300)
    """

    def __init__(self, kmeans_labels=None, kmeans_centers=None,
                 super_labels=None, frame_super=None,
                 kmeans_meta=None, super_meta=None,
                 fig_width=FIG_WIDTH, fig_height=FIG_HEIGHT):
        """
        Args:
            kmeans_labels: np.ndarray (N,) — K-Means cluster ID per frame
            kmeans_centers: np.ndarray (k, 63) — K-Means centers
            super_labels: np.ndarray (k,) — super cluster per center, OR
            frame_super:  np.ndarray (N,) — super cluster per frame
            kmeans_meta: dict with 'k', 'seed'
            super_meta: dict with 'n_super'
            fig_width, fig_height: figure dimensions in inches
        """
        self.kmeans_labels = kmeans_labels
        self.kmeans_centers = kmeans_centers
        self.kmeans_meta = kmeans_meta or {}
        self.super_meta = super_meta or {}

        # Resolve frame_super
        if frame_super is not None:
            self.frame_super = frame_super
        elif super_labels is not None:
            # super_labels is per-center (k,) → expand to per-frame
            if len(super_labels) == len(kmeans_centers):
                self.frame_super = super_labels[kmeans_labels]
            else:
                self.frame_super = super_labels
        else:
            self.frame_super = None

        self.n_clusters = kmeans_meta.get('k', len(kmeans_centers)) if kmeans_centers is not None else 512
        self.n_super = super_meta.get('n_super', 20)
        self.fig_width = fig_width
        self.fig_height = fig_height

        self.fig = None
        self.axes = {}
        self._overview_ax = None
        self._sc_axes = {}

    # --------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------

    def plot(self,
             overview_umap=None,       # np.ndarray (n_ov, 2)
             overview_labels=None,     # np.ndarray (n_ov,) — labels for overview scatter (must match overview_umap rows)
             sc_umaps=None,            # dict {sc_id: np.ndarray (n, 2)}
             cluster_cmap=CLUSTER_CMAP,
             super_cmap=SUPER_CMAP,
             scatter_size_ov=SCATTER_SIZE_OVERVIEW,
             scatter_alpha_ov=SCATTER_ALPHA_OVERVIEW,
             scatter_size_sc=SCATTER_SIZE_SC,
             scatter_alpha_sc=SCATTER_ALPHA_SC,
             title='TWSLT — K-Means + Super Clusters, UMAP 2D',
             show_empty=True,
             n_rows=9, n_cols=5,
             height_ratios=None,
             wspace=0.1, hspace=0.1):
        """
        Draw the full TWSLT overview + super cluster grid.

        Args:
            overview_umap: 2D UMAP coords for overview (n_ov, 2)
            overview_labels: labels for overview scatter (n_ov,), e.g. super cluster ID.
                              If None, uses frame_super[random_sample].
            sc_umaps: dict {sc_id: 2D UMAP coords (n, 2)}
            show_empty: if True, draw N/A for empty super clusters
        """
        # height_ratios: e.g. [4,1,1,1,1,1,1,1,1] means overview=row 0 (4x height),
        # then rows 1-8 are 1x each for SC panels (4 rows x 5 cols = 20 SC slots)
        if height_ratios is None:
            height_ratios = [4, 1, 1, 1, 1, 1, 1, 1, 1]   # overview=4, SC rows=4, total=9

        self.fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        gs = self.fig.add_gridspec(
            n_rows, n_cols,
            height_ratios=height_ratios,
            width_ratios=[1.0] * n_cols,
            wspace=wspace, hspace=hspace,
        )

        # Overview spans the first N rows (ov_height = height_ratios[0] rows)
        ov_height = height_ratios[0]   # number of grid rows for overview
        sc_row_offset = ov_height     # first grid row for SC panels

        self._overview_ax = self.fig.add_subplot(gs[:ov_height, :])
        self._draw_overview(self._overview_ax, overview_umap, overview_labels, super_cmap,
                            scatter_size_ov, scatter_alpha_ov)
        self.axes['overview'] = self._overview_ax

        # ---- Super cluster panels ----
        self._sc_axes = {}
        for sc in range(self.n_super):
            sc_row = (sc // n_cols)       # 0-indexed within SC grid
            sc_col = sc % n_cols
            ax = self.fig.add_subplot(gs[sc_row_offset + sc_row, sc_col])
            self._sc_axes[sc] = ax
            self._draw_sc_panel(ax, sc, sc_umaps, cluster_cmap,
                                scatter_size_sc, scatter_alpha_sc,
                                show_empty)
            self.axes[f'sc_{sc}'] = ax

        self.fig.suptitle(title, fontsize=TITLE_SIZE, y=0.90)
        self.fig.tight_layout(rect=[0, 0, 1, 0.90])
        return self.fig

    def _draw_overview(self, ax, umap_coords, overview_labels, cmap_name,
                       scatter_size, scatter_alpha):
        """Draw the overview scatter plot."""
        if umap_coords is None:
            ax.set_title('Overview (no UMAP data)', fontsize=SC_TITLE_SIZE)
            return

        if overview_labels is None:
            overview_labels = self.frame_super

        sc_unique = np.unique(overview_labels)
        n_colors = max(len(sc_unique), 20)
        cmap = cm.get_cmap(cmap_name, n_colors)

        # Ensure labels match umap points
        n_points = min(len(umap_coords), len(overview_labels))
        ax.scatter(
            umap_coords[:n_points, 0], umap_coords[:n_points, 1],
            c=[cmap(int(l) % n_colors) for l in overview_labels[:n_points]],
            s=scatter_size, alpha=scatter_alpha,
        )
        ax.set_title('Overview — 20 Super Clusters', fontsize=TITLE_SIZE - 2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

    def _draw_sc_panel(self, ax, sc_id, sc_umaps, cmap_name,
                       scatter_size, scatter_alpha, show_empty):
        """Draw a single super cluster panel."""
        n_frames_total = int(np.sum(self.frame_super == sc_id)) if self.frame_super is not None else 0

        if sc_umaps is not None and sc_id in sc_umaps and sc_umaps[sc_id] is not None:
            entry = sc_umaps[sc_id]
            # Support both old format (just coords) and new format (coords, labels)
            if isinstance(entry, tuple) and len(entry) == 2:
                umap_sc, labels_sc = entry
                # Ensure UMAP points and labels have matching lengths
                min_len = min(len(umap_sc), len(labels_sc))
                umap_sc = umap_sc[:min_len]
                labels_sc = labels_sc[:min_len]
            else:
                umap_sc = entry
                labels_sc = self.kmeans_labels[self.frame_super == sc_id]
                # Ensure UMAP points and labels have matching lengths
                min_len = min(len(umap_sc), len(labels_sc))
                umap_sc = umap_sc[:min_len]
                labels_sc = labels_sc[:min_len]

            if len(umap_sc) == 0:
                ax.set_title(f'SC {sc_id} ({n_frames_total} fr)', fontsize=SC_TITLE_SIZE)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
            else:
                cmap = cm.get_cmap(cmap_name, self.n_clusters)
                ax.scatter(
                    umap_sc[:, 0], umap_sc[:, 1],
                    c=[cmap(int(l) % self.n_clusters) for l in labels_sc],
                    s=scatter_size, alpha=scatter_alpha,
                )
                ax.set_title(f'SC {sc_id} ({len(umap_sc)} fr)', fontsize=SC_TITLE_SIZE)
        else:
            if show_empty:
                ax.set_title(f'SC {sc_id} ({n_frames_total} fr)', fontsize=SC_TITLE_SIZE)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
            else:
                ax.set_title(f'SC {sc_id}', fontsize=SC_TITLE_SIZE)
                ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------

    def save_fig(self, path, dpi=200, bbox_inches='tight'):
        """Save matplotlib figure directly."""
        if self.fig is None:
            raise RuntimeError('Must call plot() first')
        self.fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        print(f'[SLBHSViz] Saved figure to {path}')

    def save_svg(self, path):
        """Save as SVG."""
        self.save_fig(path, dpi=72)

    def save_png(self, path, dpi=200):
        """Save as PNG. Uses cairosvg if available, else matplotlib."""
        try:
            import cairosvg
            import tempfile, os
            # Save SVG to temp, convert
            tmp_svg = path.replace('.png', '_tmp.svg')
            self.save_svg(tmp_svg)
            cairosvg.svg2png(url=tmp_svg, write_to=path, dpi=dpi)
            os.remove(tmp_svg)
            size_mb = os.path.getsize(path) / 1024**2
            print(f'[SLBHSViz] Saved PNG via cairosvg to {path} ({size_mb:.1f} MB)')
        except ImportError:
            print('[SLBHSViz] cairosvg not found, using matplotlib direct save')
            self.save_fig(path, dpi=dpi)

    # --------------------------------------------------------------------------
    # Factory (load all pre-computed data) — convenient one-liner
    # --------------------------------------------------------------------------

    @classmethod
    def from_results(cls, results_dir, data_loader=None, compute_umap=True,
                     n_super=20, seed=42, overview_n=10000, sc_n=2000):
        """
        Load everything from results_dir and compute UMAP.

        Args:
            results_dir: path to TWSLT/results/
            data_loader: DataLoader instance (for X_scaled)
            compute_umap: if True, run UMAP; if False, skip
            n_super, seed, overview_n, sc_n: UMAP parameters

        Returns:
            SLBHSViz instance (with plot() called but not yet saved)
        """
        import json
        import os, sys
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _base not in sys.path:
            sys.path.insert(0, _base)
        from clustering.kmeans import KMeansClusterer
        from clustering.super_cluster import SuperClusterer
        from clustering.reducer import UMAPReducer

        kc = KMeansClusterer(results_dir=results_dir)
        kc.load()
        labels = kc.labels_
        centers = kc.centers_

        with open(os.path.join(results_dir, 'kmeans_meta.json')) as f:
            kmeans_meta = json.load(f)
        with open(os.path.join(results_dir, 'super_meta.json')) as f:
            super_meta = json.load(f)

        # Super cluster
        sc = SuperClusterer(kmeans_labels=labels, kmeans_centers=centers,
                            results_dir=results_dir)
        sc.fit(n_super=n_super)
        frame_super = sc.frame_super_
        super_labels = sc.super_labels_

        # Scale X for UMAP
        from sklearn.preprocessing import StandardScaler
        X_raw, _ = data_loader.load()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # UMAP
        reducer = UMAPReducer(X_scaled, super_labels=frame_super)
        overview_umap = None
        overview_labels = None
        sc_umaps = {}
        if compute_umap:
            overview_umap, ov_idx = reducer.transform_overview(n=overview_n, seed=seed)
            overview_labels = frame_super[ov_idx]
            for s in range(n_super):
                sc_umap, sc_umap_idx = reducer.transform_sc(sc_id=s, n=sc_n, seed=seed)
                sc_mask = frame_super == s
                sc_indices = np.where(sc_mask)[0]
                sc_frame_labels = labels[sc_indices[sc_umap_idx]]
                sc_umaps[s] = (sc_umap, sc_frame_labels)

        viz = cls(
            X=X_scaled, kmeans_labels=labels, kmeans_centers=centers,
            frame_super=frame_super, super_labels=super_labels,
            kmeans_meta=kmeans_meta, super_meta=super_meta,
        )
        viz.plot(overview_umap=overview_umap, overview_labels=overview_labels, sc_umaps=sc_umaps)
        return viz
