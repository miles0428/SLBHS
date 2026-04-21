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
| `--no-verbose` | - | Suppress K-Means iteration progress |



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
