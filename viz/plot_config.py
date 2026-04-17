"""
plot_config.py — Default constants for TWSLT visualization.
"""
import matplotlib.cm as cm
import numpy as np

# ---- UMAP / clustering defaults ----
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_OVERVIEW_N = 10000
UMAP_SC_N = 2000

# ---- K-Means defaults ----
KMEANS_K = 512
KMEANS_SEED = 42

# ---- Super cluster defaults ----
N_SUPER = 20
SUPER_LINKAGE = 'ward'

# ---- Scatter plot defaults ----
SCATTER_SIZE_OVERVIEW = 2
SCATTER_ALPHA_OVERVIEW = 0.5
SCATTER_SIZE_SC = 1
SCATTER_ALPHA_SC = 0.6

# ---- Colormaps ----
SUPER_CMAP = 'tab20'           # 20 distinct colors for super clusters
CLUSTER_CMAP = 'gist_rainbow'  # 512 distinct colors for cluster IDs

def get_cluster_colors(labels, cmap_name='gist_rainbow', n_clusters=512):
    """Return list of RGBA colors for a list/array of cluster labels."""
    cmap = cm.get_cmap(cmap_name, n_clusters)
    return [cmap(int(l) % n_clusters) for l in labels]

# ---- Grid layout ----
GRID_ROWS = 9
GRID_COLS = 5
GRID_HEIGHT_RATIOS = [4, 1, 1, 1, 1, 1, 1, 1, 1]  # Overview=4 of 9 rows

# ---- Figure size ----
FIG_WIDTH = 20       # inches
FIG_HEIGHT = 30      # inches

# ---- DPI ----
DPI_PNG = 200
DPI_HD = 300

# ---- Font sizes ----
TITLE_SIZE = 14
AXIS_LABEL_SIZE = 10
SC_TITLE_SIZE = 8
