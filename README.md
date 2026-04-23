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

## Feature Extraction

The pipeline uses a **78-dimensional combined feature space**:

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Raw landmarks (scaled) | 63 | Flat MediaPipe 21×3 coordinates, standardized per axis |
| Cosine similarity | 15 | Bone-angle cosine similarities per finger joint |

### Cosine Feature Extraction (`feature_transform.py`)

The `compute_cosine_features(X)` function transforms aligned 63D hand pose data into 15 bone-angle cosine similarity features — one per finger joint triplet:

```
Feature  0-2:  Thumb   — angles (0-1, 1-2) vs (1-2, 2-3), (1-2, 2-3) vs (2-3, 3-4), (2-3, 3-4) vs (3-4, 4-?)
Feature  3-5:  Index   — (0-5, 5-6) vs (5-6, 6-7), ...
Feature  6-8:  Middle  — (0-9, 9-10) vs (9-10, 10-11), ...
Feature  9-11: Ring    — (0-13, 13-14) vs (13-14, 14-15), ...
Feature 12-14: Pinky   — (0-17, 17-18) vs (17-18, 18-19), ...
```

Each feature = cos(angle) between two consecutive bone vectors. Values are clipped to [-1, 1] for numerical robustness.

```python
from SLBHS.clustering.feature_transform import compute_cosine_features

X_aligned = np.load('data.npz')['aligned_63d']  # (N, 63)
cos_features = compute_cosine_features(X_aligned)  # (N, 15)
```

## Visualization (`gen_samples.py`)

Generate per-cluster PNG visualizations showing representative frames:

```bash
python -m SLBHS.viz.gen_samples --results-dir results
```

Each cluster PNG contains a 5×N grid of hand skeleton overlays with MediaPipe landmark connections.

## Inference (classify new data)

```python
from SLBHS.clustering.kmeans import KMeansClusterer

# Load trained model (pipeline uses 78D = 63D scaled + 15D cosine features)
kc = KMeansClusterer()
kc.load_model('results')  # loads kmeans_model.joblib + kmeans_scaler.joblib

# Classify new hand pose data (63D raw → auto-computed 78D)
new_data = np.load('new_handposes.npz')['X']  # shape (N, 63)
labels = kc.predict(new_data, cosine=False)  # cosine=False: 63D scaled only
labels = kc.predict(new_data, cosine=True)   # cosine=True:  63D scaled + 15D cosine → 78D
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