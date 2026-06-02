# BLUEPRINT.md — SLBHS Super Cluster Pipeline v3

**Goal:** Identify physically-equivalent hand poses (Super Clusters) via temporal transition similarity.

**Pipeline v3: Fully relies on K-Means model (.joblib), no training.**

---

## Input

| Source | Content |
|--------|---------|
| H5 raw file | `aligned_63d`(N,63) + `x_vec`/`y_vec`/`z_vec`(N,3) |
| `kmeans_model.joblib` | Pre-trained KMeans model (for prediction) |
| `kmeans_scaler.joblib` | Pre-trained StandardScaler |

---

## Pipeline Flow (v3)

```
H5 → KMeansClusterer.load_model(model_dir) → predict() → labels → C → S → SuperCluster
```

### Step 1: K-Means Prediction (load only, no training)
- `KMeansClusterer.load_model(model_dir)` — load pre-trained model
- `KMeansClusterer.predict(X)` → Token_ID (N,) 0~k-1

### Step 2: Hand Labeling
- `cross(vec_x, vec_y) dot vec_z`
- dot < 0 → "L" (left hand)
- dot >= 0 → "R" (right hand)

### Step 3: Build Transition Matrix C[k×k]
- Left/right handled separately: `n → n+delta_t` transitions
- `C = C_left + C_right`

### Step 4: Symmetrize + Row Normalize
- `W = (C + C.T) / 2`
- `M_ij = W_ij / Σ_k(W_ik)` → probability matrix

### Step 5: Similarity Matrix S[k×k]
- `S_ij = cos(M_i, M_j)`

### Step 6: Super Cluster Extraction
- `S_ij > tau` → create edge
- Connected Components → N Super_Clusters

---

## Two Modes

### `--h5` mode (single H5, debug)
```bash
python run_pipeline.py \
  --h5 /path/to/single.h5 \
  --model-dir /path/to/kmeans/model/ \
  --k 1024 \
  --delta-t 10 \
  --tau 0.9 \
  --output results/
```
Runs `fit()` end-to-end.

### `--folder` mode (batch multiple H5s)
```bash
python run_pipeline.py \
  --folder /path/to/h5/folder \
  --model-dir /path/to/kmeans/model/ \
  --k 1024 \
  --delta-t 10 \
  --tau 0.9 \
  --output results/
```
Each H5 runs `update()` (accumulates C), then `finalize()` computes S + BigClusterer.

---

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--folder` | H5 folder (batch) | - |
| `--h5` | Single H5 (debug) | - |
| `--model-dir` | KMeans model dir (k from meta.json) | - |
| `--output` | Output folder | results |
| `--k` | ~~K-Means cluster count~~ (removed, k from model meta.json) | ~~1024~~ |
| `--tau` | Similarity threshold | 0.9 |
| `--delta-t` | Transition interval | 10 |
| `--min-transitions` | Minimum transition count | 0 |
| `--symmetrize` | Symmetrize matrix | True |

---

## OOP Architecture

### Class Overview

| Class | Module | Responsibility |
|-------|--------|----------------|
| `HandLabeler` | `clustering.super_cluster_pipeline` | vec_x×vec_y dot vec_z → L/R |
| `TransitionCounter` | `clustering.super_cluster_pipeline` | Build C[k×k] transition matrix (L+R combined) |
| `SimilarityMatrix` | `clustering.super_cluster_pipeline` | Compute S = cosine(M_prob) |
| `BigClusterer` | `clustering.super_cluster_pipeline` | Super Cluster Extraction (Connected Components) |
| `BigClusterPipeline` | `clustering.super_cluster_pipeline` | Chain everything, one-shot output |
| `KMeansClusterer` | `clustering.kmeans` | K-Means prediction + model load/save |

---

## BigClusterPipeline API (v3)

```python
class BigClusterPipeline:
    def __init__(self, k=1024, tau=0.9, delta_t=10,
                 min_transitions=0, symmetrize=True,
                 model_dir=None, results_dir=None):
        """
        model_dir: KMeans model dir (pre-trained only, no training)
        results_dir: pipeline output path
        """

    def fit(self, X, x_vec, y_vec, z_vec,
            labels=None, k=None, tau=None,
            min_transitions=None, delta_t=None,
            symmetrize=None, model_dir=None,
            results_dir=None) -> self:
        """Single H5 end-to-end"""

    def update(self, X, x_vec, y_vec, z_vec) -> self:
        """Process one H5, accumulate C. Model loaded once."""

    def finalize(self, tau=None) -> self:
        """Call after all H5s done, compute S + BigClusterer"""

    def save(self, results_dir) -> None:
        """Save outputs"""
```

---

## Output (Validation Criteria)

| Output | Path | Success Criteria |
|--------|------|------------------|
| Super Cluster mapping | `results/super_cluster_map.json` | JSON, contains N Super_Clusters |
| S matrix | `results/similarity_matrix.npy` | (k, k), no NaN |
| C matrix | `results/transition_matrix.npy` | (k, k) |
| W matrix | `results/symmetrized_matrix.npy` | (k, k), symmetric |
| Pipeline summary | `results/pipeline_phase2.json` | JSON summary |

**Failure indicators:** "Error" / "NaN" in output → failure

---

_Last updated: 2026-05-07_