# SLBHS — Sign Language Basic Handshapes

Hand pose clustering and UMAP visualization pipeline for sign language basic handshapes.

## Installation

### Install directly from GitHub (no local clone needed)

```bash
pip install git+https://github.com/miles0428/SLBHS.git
```

### Clone and install locally (recommended when developing)

```bash
git clone https://github.com/miles0428/SLBHS.git
cd SLBHS
pip install .
```

## Quick Start

```bash
# Full pipeline: load data → K-Means → SuperCluster → UMAP → plot → save
python run_visualization.py --k 512 --n-super 20 --format both

# Skip K-Means and SuperCluster (load cached results)
python run_visualization.py --skip-kmeans --skip-super

# Customize UMAP sampling
python run_visualization.py --k 512 --n-super 20 --overview-n 10000 --sc-n 2000
```

## Architecture

```
SLBHS/
├── data/loader.py          DataLoader — H5 reader (multi-file concat + cache)
├── clustering/
│   ├── kmeans.py           KMeansClusterer — fit/save/elbow/silhouette
│   ├── super_cluster.py    SuperClusterer — hierarchical clustering on centers
│   └── reducer.py          UMAPReducer — with persistent cache
├── viz/
│   ├── layout.py           GridLayout — gridspec parameters
│   ├── plot_config.py      plot_config — constants
│   └── visualizer.py       TWSLTViz — main plotter
└── run_visualization.py    pipeline entry point
```

## Key Classes

```python
from SLBHS.data.loader import DataLoader
from SLBHS.clustering.kmeans import KMeansClusterer
from SLBHS.clustering.super_cluster import SuperClusterer
from SLBHS.clustering.reducer import UMAPReducer
from SLBHS.viz.visualizer import TWSLTViz

# 1. Load data
X, meta = DataLoader('/path/to/h5/files/').load()

# 2. K-Means
kc = KMeansClusterer(X, results_dir='results')
kc.fit(k=512); kc.save()

# 3. SuperCluster
sc = SuperClusterer(kc.labels_, kc.centers_)
sc.fit(n_super=20); sc.save()

# 4. UMAP (cached)
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
reducer = UMAPReducer(X_scaled, super_labels=sc.frame_super_)
ov, ov_idx = reducer.transform_overview(n=10000)

# 5. Plot
viz = TWSLTViz(X=X_scaled, kmeans_labels=kc.labels_,
                kmeans_centers=kc.centers_,
                frame_super=sc.frame_super_)
viz.plot(overview_umap=ov, overview_labels=sc.frame_super_[ov_idx])
viz.save_svg('output.svg')
viz.save_png('output.png', dpi=300)
```

## Author
Yu-Cheng Chung

## License
See `LICENSE` file. Commercial use (including training commercial AI models) requires explicit prior authorization from the copyright holder. Educational and personal use do not require authorization.
