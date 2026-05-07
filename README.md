# SLBHS — Sign Language Basic Handshapes

Hand pose clustering pipelines for sign language basic handshapes.

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

### Train K-Means
```bash
python -m SLBHS.run_visualization --k 512 --cosine-features
```

### Run Big Cluster Pipeline
```bash
python -m SLBHS.run_pipeline --h5 /path/to/file.h5 --model-dir /path/to/model_dir --delta-t 10 --tau 0.9 --output results
```

## Input Format (H5)

```
file_crop---xxxxxxxxxx.h5
├── aligned_63d    (N, 63)   float32  — 21 landmarks × 3 axes
├── x_vec, y_vec, z_vec (N, 3) float32 — hand orientation vectors
└── is_mirror     (N,) bool   — mirror flag (optional)
```

## Output Files

| File | Shape | Description |
|------|-------|-------------|
| `similarity_matrix.npy` | (k, k) | Cosine similarity S |
| `transition_matrix.npy` | (k, k) | Transition counts C |
| `super_cluster_map.json` | JSON | `{token_id: super_cluster_id}` |

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--h5` | — | Single H5 path |
| `--folder` | — | Folder of H5 files (batch mode) |
| `--model-dir` | — | Pretrained KMeans model directory (required) |
| `--delta-t` | 10 | Transition interval |
| `--tau` | 0.9 | Similarity threshold |
| `--min-transitions` | 0 | Minimum transition count filter |
| `--symmetrize` | true | Symmetrize transition matrix |
| `--output` | results | Output directory |

## Documentation

## Author

Yu-Cheng Chung

## License

This project is for research use only. See `LICENSE` file for details.

Permitted:
- Use code to train your own models

Prohibited:
- Use pretrained models or outputs to train other models
- Commercial use

For commercial licensing, contact the author.
