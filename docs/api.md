# API Reference

## Main Module: `SLBHS.clustering.super_cluster_pipeline`

---

## `HandLabeler`

Determines left/right hand using `cross(vec_x, vec_y) dot vec_z`.

### Methods

#### `fit_predict(x_vec, y_vec, z_vec) -> np.ndarray`

Fit and predict hand labels.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `x_vec` | (N, 3) float32 | x-axis vector |
| `y_vec` | (N, 3) float32 | y-axis vector |
| `z_vec` | (N, 3) float32 | z-axis vector |

**Returns**
- `(N,) '<U1'` — 'L' or 'R'

**Examples**

```python
from SLBHS.clustering.super_cluster_pipeline import HandLabeler

labeler = HandLabeler()
hand_labels = labeler.fit_predict(x_vec, y_vec, z_vec)
print(f"L={np.sum(hand_labels=='L')}, R={np.sum(hand_labels=='R')}")
```

---

## `TransitionCounter`

Builds Token transition matrix C[1024×1024].

### Constructor

```python
TransitionCounter(k=1024, delta_t=1, min_transitions=0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 1024 | Total token count |
| `delta_t` | int | 1 | Frames to look ahead |
| `min_transitions` | int | 0 | Minimum transition count |

### Methods

#### `fit(token_ids, hand_labels, delta_t=None, min_transitions=None) -> self`

Build transition matrix.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `token_ids` | (N,) int | Token_ID, 0-1023 |
| `hand_labels` | (N,) '<U1' | 'L' or 'R' |
| `delta_t` | int or None | Override constructor's delta_t |
| `min_transitions` | int or None | Override constructor's min_transitions |

**Returns**
- `self`

#### `get_matrix() -> np.ndarray`

Get transition matrix.

**Returns**
- `(k, k)` float64 — raw transition count matrix (unnormalized)

---

## `SimilarityMatrix`

Computes similarity matrix S from transition matrix.

### Methods

#### `compute(M, symmetrize=True) -> np.ndarray`

Compute similarity matrix.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `M` | (k, k) float64 | Raw transition count matrix C |
| `symmetrize` | bool | Symmetrize matrix (default True) |

**Returns**
- `(k, k)` float64 — cosine similarity matrix S

**Process**
1. Symmetrize: `W = (M + M.T) / 2`
2. Row normalize: `M_ij = W_ij / Σ_k(W_ik)`
3. Cosine similarity: `S_ij = cos(M_i, M_j)`

---

## `BigClusterer`

Extracts Super Clusters based on similarity threshold.

### Constructor

```python
BigClusterer(tau=0.9)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.9 | Similarity threshold (0.0-1.0) |

### Methods

#### `fit(S, tau=None) -> self`

Extract Super Clusters from similarity matrix S.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `S` | (k, k) float64 | Cosine similarity matrix |
| `tau` | float or None | Override constructor's tau |

**Returns**
- `self`

#### `get_clusters() -> dict`

Get Super Cluster mapping.

**Returns**
- `dict` — `{token_id: super_cluster_id}`

---

## `BigClusterPipeline`

Chains all steps, one-shot Phase 2 output.

### Constructor

```python
BigClusterPipeline(
    k=512,
    tau=0.9,
    delta_t=1,
    cosine_features=True,
    min_transitions=0,
    symmetrize=True,
    results_dir=None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 512 | K-Means cluster count |
| `tau` | float | 0.9 | Similarity threshold |
| `delta_t` | int | 1 | Transition interval |
| `cosine_features` | bool | True | Use cosine feature (78D) |
| `min_transitions` | int | 0 | Minimum transition count |
| `symmetrize` | bool | True | Symmetrize matrix |
| `results_dir` | str or None | None | KMeansClusterer model path |

### Methods

#### `fit(X, x_vec, y_vec, z_vec, labels=None, k=None, tau=None, cosine_features=None, min_transitions=None, delta_t=None, symmetrize=None, results_dir=None) -> self`

Run full pipeline.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `X` | (N, 63) float32 | aligned_63d hand pose data |
| `x_vec` | (N, 3) float32 | x-axis vector |
| `y_vec` | (N, 3) float32 | y-axis vector |
| `z_vec` | (N, 3) float32 | z-axis vector |
| `labels` | (N,) int or None | Token_ID; if None, generated based on cosine_features mode |
| `k` | int or None | Override constructor's k |
| `tau` | float or None | Override constructor's tau |
| `cosine_features` | bool or None | Override constructor's cosine_features |
| `min_transitions` | int or None | Override constructor's min_transitions |
| `delta_t` | int or None | Override constructor's delta_t |
| `symmetrize` | bool or None | Override constructor's symmetrize |
| `results_dir` | str or None | Override constructor's results_dir |

**Returns**
- `self`

#### `save(results_dir) -> None`

Save outputs to results_dir.

**Output Files**

| File | Format | Description |
|------|--------|-------------|
| `similarity_matrix.npy` | (k, k) float64 | Cosine similarity matrix S |
| `transition_matrix.npy` | (k, k) float64 | Raw transition count matrix C |
| `symmetrized_matrix.npy` | (k, k) float64 | Symmetrized matrix W |
| `super_cluster_map.json` | JSON | `{token_id: super_cluster_id}` |
| `pipeline_phase2.json` | JSON | Phase 2 full summary |

---

## `SLBHS.clustering.kmeans`

### `KMeansClusterer`

K-Means clusterer with inference support.

#### Constructor

```python
KMeansClusterer(X=None, results_dir=None, k=None, seed=42)
```

#### Methods

##### `load_model(results_dir=None, prefix='kmeans') -> None`

Load pre-trained model (joblib format).

##### `predict(X_new) -> np.ndarray`

Classify new data.

**Parameters**
- `X_new`: `(M, 63)` float32 — new hand pose data

**Returns**
- `(M,)` int — cluster labels (0 to k-1)

**Example**

```python
from SLBHS.clustering.kmeans import KMeansClusterer

kc = KMeansClusterer()
kc.load_model('results')
labels = kc.predict(new_data)
```

---

## `SLBHS.data.loader`

### `DataLoader`

H5 file reading and caching.

#### Constructor

```python
DataLoader(data_dir=None, cache_dir=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_dir` | str | H5 file path, directory or glob pattern |
| `cache_dir` | str | Cache directory |

#### Methods

##### `load(force_reload=False) -> Tuple[np.ndarray, dict]`

Load aligned_63d data.

**Returns**
- `X`: `(N, 63)` float32
- `meta`: dict (contains `n_frames`, `files`, `n_files`, `per_file_frames`)

**Example**

```python
from SLBHS.data.loader import DataLoader

loader = DataLoader(data_dir='/path/to/h5/')
X, meta = loader.load()
print(f"Loaded {meta['n_frames']} frames from {meta['n_files']} files")
```

---

_Last updated: 2026-05-07_