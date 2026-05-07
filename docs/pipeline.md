# Super Cluster Pipeline — Detailed Flow

## Overview

Super Cluster Pipeline is the core module of SLBHS Phase 2, which identifies physically-equivalent hand-pose clusters via **temporal transition similarity**.

**Core assumption**: If two Tokens tend to appear in similar contexts (surrounding Tokens) across time sequences, they may represent the same hand pose.

---

## Mathematical Principles

### Step 1: Hand Labeling

Determines left/right hand using `cross(vec_x, vec_y) dot vec_z`:

```
dot = sum(cross(vec_x, vec_y) * vec_z)
```

- `dot < 0` → **Left hand (L)**
- `dot >= 0` → **Right hand (R)**

Physics: cross(x, y) produces a vector perpendicular to the palm plane; dot product with z is positive for right hand, negative for left hand.

### Step 2: Left/Right Track Separation

Token sequences are split into two time series:
- `Left_Track`: Token_ID sequence where hand_label == "L"
- `Right_Track`: Token_ID sequence where hand_label == "R"

### Step 3: Build Transition Matrix C[1024×1024]

$$C_{ij} = \sum_{\text{track} \in \{L,R\}} \sum_{\delta=1}^{\Delta} \mathbb{1}[Token_{t}=i \wedge Token_{t+\delta}=j]$$

Computed separately for left and right hands, then summed.

### Step 4: Symmetrization + Row Normalization

$$W = \frac{C + C^T}{2}$$
$$M_{ij} = \frac{W_{ij}}{\sum_k W_{ik}}$$

M is a probability matrix; each row sums to 1.

### Step 5: Cosine Similarity

$$S_{ij} = \frac{M_i \cdot M_j}{\|M_i\| \|M_j\|}$$

S is the similarity matrix. Larger S_ij means Token i and Token j have more similar transition behavior.

### Step 6: Super Cluster Extraction

- Set threshold τ
- If S_ij > τ, create an edge between Token i and Token j
- Use Connected Components algorithm to form clusters

---

## Pipeline Flowchart

```
Input H5
  │
  ▼
[HandLabeler] ──→ hand_labels (L/R)
  │
  ▼
[BigClusterPipeline]
  │
  └── [KMeansClusterer.predict] ──→ labels
  │
  ▼
[TransitionCounter] ──→ C (1024, 1024)
  │
  ▼
[SimilarityMatrix] ──→ S (1024, 1024)
  │
  ▼
[BigClusterer] ──→ super_cluster_map
  │
  ▼
Output: S.npy, C.npy, W.npy, super_cluster_map.json, pipeline_phase2.json
```

---

## Cold-Start Handling

Tokens with zero outgoing transitions (cold-start rows) receive an **epsilon-uniform smoothing** before row normalization to avoid division-by-zero and NaN propagation.

**Mechanism**:
1. Detect rows where `row_sum == 0` in the symmetrized matrix W
2. Assign `epsilon = 1 / n_cols` to each entry in that row
3. Proceed with standard row normalization

**Why preserve S_ii = 1**: After row normalization, a uniform distribution has entropy n·log(n). Its cosine self-similarity is exactly 1.0 (since the row vector equals its own unit vector), so S_ii = 1 is naturally preserved — no special logic is needed.

---

## Parameter Tuning Guide

### tau (Similarity Threshold)

| tau | Expected # of Clusters | Description |
|-----|------------------------|-------------|
| 0.95 | Many (small clusters) | High threshold; only very similar Tokens are grouped together |
| 0.90 | Medium | Default; balance of precision and recall |
| 0.85 | Few (large clusters) | Low threshold; more Tokens grouped together |

### delta_t (Transition Interval)

| delta_t | Use Case |
|---------|---------|
| 1 | Standard; only consecutive frames |
| 2-3 | For low frame-rate or slow-motion footage |
| >3 | Usually not recommended; may introduce too much noise |

### min_transitions (Minimum Transition Count)

- Default 0 means no filtering
- Set >0 to eliminate low-frequency noise
- Recommended values: 1-5

---

## Validation Methods

### Check S Matrix

```python
import numpy as np
S = np.load('results/similarity_matrix.npy')
assert not np.isnan(S).any(), "S contains NaN"
assert np.allclose(S, S.T), "S is not symmetric"
assert np.all(S <= 1.0), "S values > 1"
```

### Check C Matrix

```python
C = np.load('results/transition_matrix.npy')
assert C.shape == (1024, 1024), "C shape mismatch"
assert not np.isnan(C).any(), "C contains NaN"
```

### Check Super Cluster Map

```python
import json
with open('results/super_cluster_map.json') as f:
    sc_map = json.load(f)
print(f"Tokens in clusters: {len(sc_map)}")
print(f"Unique super clusters: {len(set(sc_map.values()))}")
```

---

## Difference from Hierarchical Clustering

| Item | BigClusterPipeline (Phase 2) | SuperClusterer (Visualization) |
|------|------------------------------|--------------------------------|
| Method | Connected Components | Agglomerative Hierarchical |
| Input | Temporal transition similarity matrix S | K-Means centers |
| Purpose | Find physically-equivalent hand poses | Visualization clustering |
| Use Case | Actual classification | Chart presentation |

---

_Last updated: 2026-05-07_
