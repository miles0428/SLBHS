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
python -m SLBHS.run_visualization --k 512 --n-super 20 --format both

# Skip K-Means and SuperCluster (load cached results)
python -m SLBHS.run_visualization --skip-kmeans --skip-super

# Customize UMAP sampling
python -m SLBHS.run_visualization --k 512 --n-super 20 --overview-n 10000 --sc-n 2000

# Customize n_neighbors for UMAP (default: 30)
python -m SLBHS.run_visualization --k 512 --n-neighbors 10
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | 512 | K-Means number of clusters |
| `--n-super` | 20 | Number of super clusters |
| `--batch-size` | 5000 | MiniBatch K-Means batch size |
| `--n-neighbors` | 30 | UMAP n_neighbors parameter |
| `--seed` | 42 | Random seed |
| `--data-dir` | . | H5 data directory |
| `--results-dir` | results | Results directory |
| `--dpi` | 300 | PNG DPI |
| `--format` | png | Output format: png, svg, or both |
| `--skip-kmeans` | - | Skip K-Means (load cached) |
| `--skip-super` | - | Skip SuperCluster (load cached) |
| `--skip-umap` | - | Skip UMAP computation (load cached) |

## Architecture

```
SLBHS/
├── __init__.py           Package init (exports all classes)
├── version.py            Single source of truth for version
├── data/
│   └── loader.py         DataLoader — H5 reader (multi-file concat + cache)
├── clustering/
│   ├── kmeans.py         KMeansClusterer — MiniBatchKMeans fit/save
│   ├── super_cluster.py  SuperClusterer — hierarchical clustering on centers
│   └── reducer.py        UMAPReducer — with persistent cache
├── viz/

from SLBHS.data.loader import DataLoader
from SLBHS.clustering.kmeans import KMeansClusterer
from SLBHS.clustering.super_cluster import SuperClusterer
from SLBHS.clustering.reducer import UMAPReducer
from SLBHS.viz.visualizer import SLBHSViz


# 1. Load data
loader = DataLoader('/path/to/h5/files/')
X, meta = loader.load()

# 2. K-Means (MiniBatch)
kc = KMeansClusterer(X, results_dir='results')
kc.fit(k=512, batch_size=5000); kc.save()

# 3. SuperCluster
sc = SuperClusterer(kc.labels_, kc.centers_)
sc.fit(n_super=20); sc.save()

# 4. UMAP (cached)
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
reducer = UMAPReducer(X_scaled, super_labels=sc.frame_super_)
ov, ov_idx = reducer.transform_overview(n=10000)

# 5. Plot
viz = SLBHSViz(X=X_scaled, kmeans_labels=kc.labels_,
                kmeans_centers=kc.centers_,
                frame_super=sc.frame_super_)
viz.plot(overview_umap=ov, overview_labels=sc.frame_super_[ov_idx])
viz.save_png('output.png', dpi=300)
```

## Inference (classify new data)

```python
from SLBHS.clustering.kmeans import KMeansClusterer

# Load trained model
kc = KMeansClusterer()
kc.load_model('results')  # loads kmeans_model.joblib + kmeans_scaler.joblib

# Classify new hand pose data
new_data = np.load('new_handposes.npz')['X']  # shape (N, 63)
labels = kc.predict(new_data)
```

## Author
Yu-Cheng Chung

## License
See `LICENSE` file. 

This project is released for research purposes only.

You are allowed to:
- use the code to train your own models

You are NOT allowed to:
- use any pretrained model or outputs to train other models
- use this work in commercial settings

Commercial licensing is available upon request.
