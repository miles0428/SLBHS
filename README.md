# SLBHS — Sign Language Big Cluster Pipeline

Train K-Means on hand-pose tokens, then collapse noisy classes into Super Clusters via temporal transition similarity.

## Quick Start

### Train K-Means
```bash
python -m SLBHS.run_visualization --k 512 --cosine-features
```

### Run Big Cluster Pipeline
```bash
python -m SLBHS.run_visualization --k 512 --n-super 20 --format both
```

Use cached results:
```bash
python -m SLBHS.run_visualization --skip-kmeans --skip-super
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
| `--k` | 512 | K-Means cluster count |
| `--n-super` | 20 | Super Cluster count for visualization |
| `--cosine-features` | — | Use 78D cosine features (63D + 15D bone-angle) |
| `--skip-kmeans` | — | Use cached K-Means results |
| `--skip-super` | — | Use cached Super Cluster results |
| `--results-dir` | results | Output directory |

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