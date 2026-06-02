# API Reference

---

## `SLBHS.clustering.feature_transform`

### `compute_cosine_features`

Convert aligned_63d hand pose data to 15-dim cosine similarity features.
Each feature = cos(angle) between two consecutive bone vectors.

```python
compute_cosine_features(X, verbose=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | np.ndarray | — | `(N, 63)` flat MediaPipe landmark data **or** `(N, 21, 3)` already reshaped |
| `verbose` | bool | True | Print progress info |

**Returns**
- `features`: `(N, 15)` float32 — cosine similarity features

**Feature layout**

| Indices | Finger |
|---------|--------|
| 0–2 | Thumb |
| 3–5 | Index |
| 6–8 | Middle |
| 9–11 | Ring |
| 12–14 | Pinky |

**Examples**

```python
from SLBHS.clustering.feature_transform import compute_cosine_features

# From flat (N, 63)
features = compute_cosine_features(X)  # X.shape = (N, 63)

# From reshaped (N, 21, 3)
landmarks = X.reshape(N, 21, 3)
features = compute_cosine_features(landmarks)  # shape (N, 15)
```

---

## `SLBHS.clustering.super_cluster`

### `SuperClusterer`

Hierarchical clustering of K-Means centers into super clusters via agglomerative clustering.

#### Constructor

```python
SuperClusterer(kmeans_labels=None, kmeans_centers=None, results_dir=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kmeans_labels` | np.ndarray (N,) | None | Per-frame K-Means labels |
| `kmeans_centers` | np.ndarray (k, 63) | None | K-Means cluster centers |
| `results_dir` | str | auto | Save location |

#### Methods

##### `fit(n_super=20, linkage='ward', metric='euclidean') -> Tuple[np.ndarray, np.ndarray]`

Run agglomerative clustering on centers.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_super` | int | 20 | Number of super clusters |
| `linkage` | str | 'ward' | Linkage method: 'ward', 'complete', 'average', 'single' |
| `metric` | str | 'euclidean' | Distance metric |

**Returns**
- `super_labels`: `(k,)` int — super cluster ID per center
- `frame_super`: `(N,)` int — super cluster ID per frame

##### `save(results_dir=None) -> dict`

Save `super_labels.npy` and `super_meta.json`.

**Returns**
- `dict` — paths of saved files

##### `load(results_dir=None) -> np.ndarray`

Load `super_labels.npy` from disk.

**Returns**
- `super_labels`: `(k,)` int

**Examples**

```python
from SLBHS.clustering.super_cluster import SuperClusterer

sc = SuperClusterer(kmeans_labels=labels, kmeans_centers=centers)
super_labels, frame_super = sc.fit(n_super=20, linkage='ward')
sc.save()
```

---

## `SLBHS.clustering.reducer`

### `UMAPReducer`

UMAP dimensionality reduction with persistent MD5-cache.

#### Constructor

```python
UMAPReducer(X_scaled, super_labels=None, cache_dir=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_scaled` | np.ndarray (N, D) | — | Standardized data |
| `super_labels` | np.ndarray (N,) | None | Super cluster per frame |
| `cache_dir` | str | auto | Cache directory |

#### Methods

##### `transform_overview(n=10000, seed=42, n_neighbors=30, min_dist=0.1) -> Tuple[np.ndarray, np.ndarray]`

UMAP on subsampled data for overview plot.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n` | int | 10000 | Number of samples |
| `seed` | int | 42 | Random seed |
| `n_neighbors` | int | 30 | UMAP n_neighbors |
| `min_dist` | float | 0.1 | UMAP min_dist |

**Returns**
- `coords`: `(n_actual, 2)` float64 — 2D UMAP coordinates
- `indices`: `(n_actual,)` int — original indices used

##### `transform_sc(sc_id, n=2000, seed=42, n_neighbors=30, min_dist=0.1) -> Tuple[np.ndarray, np.ndarray]`

UMAP on frames belonging to a specific super cluster.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sc_id` | int | — | Super cluster ID |
| `n` | int | 2000 | Max samples |
| `seed` | int | 42 | Random seed |
| `n_neighbors` | int | 30 | UMAP n_neighbors |
| `min_dist` | float | 0.1 | UMAP min_dist |

**Returns**
- `coords`: `(n_actual, 2)` float64 — 2D UMAP coordinates
- `indices`: `(n_actual,)` int — indices into the subsample

##### `transform_pca() -> np.ndarray`

PCA 2D fallback (always fast, no cache).

**Returns**
- `(N, 2)` float64 — PCA coordinates

**Examples**

```python
from SLBHS.clustering.reducer import UMAPReducer

reducer = UMAPReducer(X_scaled, super_labels=frame_super, cache_dir='results/umap_cache')
overview_umap, ov_idx = reducer.transform_overview(n=10000, seed=42)
sc_umap, sc_idx = reducer.transform_sc(sc_id=5, n=2000, seed=42)
```

---

### `PCAReducer`

Simple PCA 2D wrapper.

#### Constructor

```python
PCAReducer(X_scaled, cache_dir=None)
```

#### Methods

##### `transform() -> np.ndarray`

**Returns**
- `(N, 2)` float64 — PCA coordinates

**Examples**

```python
from SLBHS.clustering.reducer import PCAReducer

pca = PCAReducer(X_scaled)
coords = pca.transform()
```

---

## `SLBHS.viz.visualizer`

### `SLBHSViz`

Main visualization class for TWSLT UMAP + super cluster grid plots.

#### Constructor

```python
SLBHSViz(kmeans_labels=None, kmeans_centers=None, super_labels=None, frame_super=None, kmeans_meta=None, super_meta=None, fig_width=20, fig_height=28)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kmeans_labels` | np.ndarray (N,) | None | K-Means cluster ID per frame |
| `kmeans_centers` | np.ndarray (k, 63) | None | K-Means centers |
| `super_labels` | np.ndarray (k,) | None | Super cluster per center |
| `frame_super` | np.ndarray (N,) | None | Super cluster per frame |
| `kmeans_meta` | dict | None | Dict with 'k', 'seed' |
| `super_meta` | dict | None | Dict with 'n_super' |
| `fig_width` | float | 20 | Figure width in inches |
| `fig_height` | float | 28 | Figure height in inches |

#### Methods

##### `plot(overview_umap=None, overview_labels=None, sc_umaps=None, ...) -> matplotlib.figure.Figure`

Draw the full overview + super cluster grid.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `overview_umap` | np.ndarray (n, 2) | None | 2D UMAP coords for overview |
| `overview_labels` | np.ndarray (n,) | None | Labels for overview scatter |
| `sc_umaps` | dict | None | `{sc_id: (coords, labels)}` for each SC |

**Returns**
- `matplotlib.figure.Figure`

##### `save_svg(path) -> None`

Save figure as SVG.

##### `save_png(path, dpi=200) -> None`

Save figure as PNG (uses cairosvg if available).

##### `save_fig(path, dpi=200, bbox_inches='tight') -> None`

Save matplotlib figure directly.

##### `from_results(cls, results_dir, data_loader=None, ...) -> SLBHSViz`

Factory: load all pre-computed data from `results_dir` and run UMAP.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results_dir` | str | — | Path to TWSLT/results/ |
| `data_loader` | DataLoader | None | DataLoader instance for X_scaled |
| `compute_umap` | bool | True | Whether to compute UMAP |
| `n_super` | int | 20 | Number of super clusters |
| `seed` | int | 42 | Random seed |
| `overview_n` | int | 10000 | Overview UMAP samples |
| `sc_n` | int | 2000 | Per-SC UMAP samples |

**Returns**
- `SLBHSViz` instance with `plot()` already called

**Examples**

```python
from SLBHS.viz.visualizer import SLBHSViz

viz = SLBHSViz(
    kmeans_labels=labels,
    kmeans_centers=centers,
    frame_super=frame_super,
)
viz.plot(overview_umap=overview_umap, overview_labels=overview_labels, sc_umaps=sc_umaps)
viz.save_png('/tmp/out.png', dpi=300)

# Or load from results
viz = SLBHSViz.from_results('results/', data_loader=loader)
viz.save_svg('/tmp/out.svg')
```

---

## `SLBHS.viz.layout`

### `GridLayout`

Manages matplotlib gridspec parameters for TWSLT overview + super cluster grid plots.

#### Constructor

```python
GridLayout(n_rows=7, n_cols=5, height_ratios=None, width_ratio=1.0, wspace=0.15, hspace=0.15, fig_width=20, fig_height=28)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_rows` | int | 7 | Total grid rows |
| `n_cols` | int | 5 | Total grid columns |
| `height_ratios` | list | [3,1,1,1,1,1,1] | Row height ratios |
| `width_ratio` | float | 1.0 | Width ratio per column (**currently unused**, reserved for future) |
| `wspace` | float | 0.15 | Horizontal subplot spacing |
| `hspace` | float | 0.15 | Vertical subplot spacing |
| `fig_width` | float | 20 | Figure width in inches |
| `fig_height` | float | 28 | Figure height in inches |

#### Methods

##### `create_figure() -> Tuple[plt.Figure, matplotlib.gridspec.GridSpec]`

Create figure and gridspec.

**Returns**
- `fig`: `matplotlib.figure.Figure`
- `gs`: `matplotlib.gridspec.GridSpec`

##### `create_subplots() -> Tuple[plt.Figure, list]`

Create figure and axes using plt.subplots.

**Returns**
- `fig`, `axes` list

##### `sc_index_to_rc(sc_idx) -> Tuple[int, int]`

Convert SC index to (row, col) within SC grid.

##### `get_sc_gs(gs, sc_idx) -> matplotlib.gridspec.GridSpec`

Get gridspec slice for a given SC index.

**Examples**

```python
from SLBHS.viz.layout import GridLayout

layout = GridLayout(n_rows=7, n_cols=5)
fig, gs = layout.create_figure()
ax_ov = fig.add_subplot(gs[:3, :])       # overview (top 3 rows)
ax_sc = fig.add_subplot(gs[3, 0])        # first SC panel
```

---

## `SLBHS` Top-Level

### `SLBHS.__version__`

```python
from SLBHS import __version__
print(__version__)  # '0.1.10'
```

**Returns**
- `str` — current version string (e.g. `"0.1.10"`)

---


## CLI Scripts

### `run_pipeline.py`

Super Cluster Pipeline CLI — loads a pre-trained K-Means model and processes H5 files to produce super clusters.

#### `parse_args() -> argparse.Namespace`

Parse command-line arguments for the pipeline.

**Returns**
- `argparse.Namespace` with the following attributes:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `folder` | str | None | Path to folder containing multiple H5 files (batch mode via `update()+finalize()`) |
| `h5` | str | None | Path to a single H5 file (single-file mode via `fit()`) |
| `model_dir` | str | None | KMeans model directory containing `kmeans_model.joblib` + `kmeans_scaler.joblib` |
| `output` | str | 'results' | Results output directory |
| `tau` | float | 0.9 | Similarity threshold (0.0–1.0). `S_ij > tau` → edge. Default: 0.9 |
| `delta_t` | int | 10 | Transition interval (`n → n+delta_t`). Default: 10 |
| `min_transitions` | int | 0 | Minimum transition count. Pairs below this are zeroed. Default: 0 |
| `symmetrize` | bool | True | Symmetrize transition matrix |
| `verbose` | bool | False | Enable verbose (DEBUG) logging |

**Modes**

| Mode | Flag | Method called |
|------|------|---------------|
| Batch (multiple H5) | `--folder` | `pipeline.update()` + `pipeline.finalize()` |
| Single H5 | `--h5` | `pipeline.fit()` directly |

**Examples**

```bash
# Batch mode (multiple H5 files)
python run_pipeline.py \\
    --folder /path/to/h5/folder \\
    --model-dir /path/to/kmeans/model/ \\
    --delta-t 10 \\
    --tau 0.9 \\
    --output results/

# Single H5 mode (debug)
python run_pipeline.py \\
    --h5 /path/to/file.h5 \\
    --model-dir /path/to/kmeans/model/ \\
    --delta-t 10 \\
    --tau 0.9 \\
    --output results/

# Verbose output
python run_pipeline.py --folder /path/to/h5/folder --model-dir /path/to/kmeans/model/ -v
```

#### `main() -> int`

Entry point for the CLI script. Returns 0 on success, non-zero on failure.

**Process**
1. Parse arguments with `parse_args()`
2. Validate input (exactly one of `--folder` or `--h5` required)
3. Load H5 data (`aligned_63d`, `x_vec`, `y_vec`, `z_vec`)
4. Run pipeline (`update()+finalize()` or `fit()`)
5. Save results with `pipeline.save()`
6. Print summary (k, tau, delta_t, model_dir, N Super Clusters, tokens in clusters)

**Returns**
- `int` — exit code (0 = success)

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
2. Zero diagonal (remove self-transitions): `W_ii = 0`
3. Row normalize: `M_prob_ij = W_ij / Σ_k(W_ik)` → probability matrix
4. Cosine similarity with i/j excluded: `S_ij = cos(M_prob[i] excluding i/j, M_prob[j] excluding i/j)` — prevents self-correlation inflation
5. Set diagonal `S_ii = 1.0`

---

## `BigClusterer`

Extracts Super Clusters based on similarity threshold.

### Constructor

```python
BigClusterer(tau=0.9)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.9 | Similarity threshold. If S_ij > tau, an edge is created between token i and j. Default: 0.9 |

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
    delta_t=10,
    min_transitions=0,
    symmetrize=True,
    model_dir=None,
    results_dir=None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 512 | K-Means cluster count |
| `tau` | float | 0.9 | Similarity threshold |
| `delta_t` | int | 10 | Transition interval |
| `min_transitions` | int | 0 | Minimum transition count |
| `symmetrize` | bool | True | Symmetrize matrix |
| `model_dir` | str or None | None | KMeans model directory (kmeans_model.joblib + kmeans_scaler.joblib) |
| `results_dir` | str or None | None | Pipeline output directory |

### Methods

#### `fit(X, x_vec, y_vec, z_vec, labels=None, tau=None, min_transitions=None, delta_t=None, symmetrize=None, model_dir=None, results_dir=None) -> self`

Run full pipeline.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `X` | (N, 63) float32 | aligned_63d hand pose data |
| `x_vec` | (N, 3) float32 | x-axis vector |
| `y_vec` | (N, 3) float32 | y-axis vector |
| `z_vec` | (N, 3) float32 | z-axis vector |
| `labels` | (N,) int or None | None | Token_ID per frame (input); if None, generated internally from X with loaded model |
| `tau` | float or None | Override constructor's tau |

| `min_transitions` | int or None | Override constructor's min_transitions |
| `delta_t` | int or None | Override constructor's delta_t |
| `symmetrize` | bool or None | Override constructor's symmetrize |
| `model_dir` | str or None | KMeans model directory (kmeans_model.joblib + kmeans_scaler.joblib) |
| `results_dir` | str or None | Override constructor's results_dir |

**Returns**
- `self`

#### `update(X, x_vec, y_vec, z_vec) -> self`

Accumulate one H5 file's data into the transition counter (batch mode).
KMeans model is loaded once on first call, then reused for subsequent calls.
Does **not** compute S or run BigClusterer — call `finalize()` after all updates.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | (N, 63) float32 | aligned_63d hand pose data |
| `x_vec` | (N, 3) float32 | x-axis vector |
| `y_vec` | (N, 3) float32 | y-axis vector |
| `z_vec` | (N, 3) float32 | z-axis vector |

**Returns**
- `self`

#### `finalize(tau=None) -> self`

Compute similarity matrix S and run BigClusterer.
Call after all `update()` calls.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | self.tau | Similarity threshold |

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

## CosineSimilarity (SLBHS.similarity.cosine_similarity)

Cosine similarity matrix computation from transition count matrix.

**Input:** `M: np.ndarray (k, k)` — transition count matrix
**Output:** `S: np.ndarray (k, k)` — cosine similarity matrix

### Constructor

```python
CosineSimilarity()
```

### Methods

#### compute

```python
CosineSimilarity.compute(self, M: np.ndarray, symmetrize: bool = True) -> np.ndarray
```

Compute cosine similarity matrix S.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| M | np.ndarray | required | (k, k) transition count matrix |
| symmetrize | bool | True | Symmetrize W = (M + M.T) / 2 |

| Return | Type | Description |
|--------|------|-------------|
| S | np.ndarray | (k, k) cosine similarity matrix |

#### get_matrix

```python
CosineSimilarity.get_matrix(self) -> np.ndarray
```

Returns the computed similarity matrix.

---

## HandLabeler (SLBHS.similarity.hand_labeler)

Classify L/R hand from orientation vectors.

**Input:** `x_vec, y_vec, z_vec: np.ndarray (N, 3)`
**Output:** `hand_labels: np.ndarray (N,)` — 'L' or 'R'

### Constructor

```python
HandLabeler()
```

### Methods

#### fit_predict

```python
HandLabeler.fit_predict(self, x_vec: np.ndarray, y_vec: np.ndarray, z_vec: np.ndarray) -> np.ndarray
```

Fit and predict L/R hand labels.

| Parameter | Type | Description |
|-----------|------|-------------|
| x_vec | np.ndarray | (N, 3) x-axis vectors |
| y_vec | np.ndarray | (N, 3) y-axis vectors |
| z_vec | np.ndarray | (N, 3) z-axis vectors |

| Return | Type | Description |
|--------|------|-------------|
| hand_labels | np.ndarray | (N,) '<U1' 'L' or 'R' |

#### fit

```python
HandLabeler.fit(self, x_vec: np.ndarray, y_vec: np.ndarray, z_vec: np.ndarray) -> None
```

Sklearn-style fit (stores results internally).

#### predict

```python
HandLabeler.predict(self, x_vec: np.ndarray, y_vec: np.ndarray, z_vec: np.ndarray) -> np.ndarray
```

Predict L/R hand labels (requires prior fit).

---

## TransitionCounter (SLBHS.similarity.transition_counter)

Build token transition matrix C[k x k].

**Input:** `labels: np.ndarray (N,)`, `hand_labels: np.ndarray (N,)`
**Output:** `C: np.ndarray (k, k)` — transition count matrix

### Constructor

```python
TransitionCounter(k: int = 1024, delta_t: int = 1, min_transitions: int = 0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| k | int | 1024 | Total number of tokens |
| delta_t | int | 1 | Steps to look back |
| min_transitions | int | 0 | Minimum transitions to keep |

### Methods

#### fit

```python
TransitionCounter.fit(self, labels: np.ndarray, hand_labels: np.ndarray, delta_t: int = None, min_transitions: int = None) -> self
```

Build transition matrix from labels and hand labels. Chainable.

| Parameter | Type | Description |
|-----------|------|-------------|
| labels | np.ndarray | (N,) token IDs (0 to k-1) |
| hand_labels | np.ndarray | (N,) 'L'/'R' hand labels |

#### update

```python
TransitionCounter.update(self, labels_batch: np.ndarray, hand_labels_batch: np.ndarray) -> self
```

Add a batch of labels to the transition matrix (cumulative). Chainable.

#### get_matrix

```python
TransitionCounter.get_matrix(self) -> np.ndarray
```

Returns the transition count matrix C (k, k).

### Usage Example

```python
from SLBHS.similarity import CosineSimilarity, HandLabeler, TransitionCounter

# Hand classification
labeler = HandLabeler()
hand_labels = labeler.fit_predict(x_vec, y_vec, z_vec)

# Build transition matrix
counter = TransitionCounter(k=1024, delta_t=1)
counter.fit(labels, hand_labels)
C = counter.get_matrix()

# Compute similarity
sim = CosineSimilarity()
S = sim.compute(C)
```

---

## SimilarityPipeline (SLBHS.similarity.similarity_pipeline)

End-to-end similarity computation pipeline.

**Input:** `clusterer` with `predict()` method + H5 folder
**Output:** `(k, k)` cosine similarity matrix

### Constructor

```python
SimilarityPipeline(clusterer, k: int = 1024, delta_t: int = 1, min_transitions: int = 0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| clusterer | object | required | Must have `predict(X) -> labels` method |
| k | int | 1024 | Number of token classes |
| delta_t | int | 1 | Steps to look back for transitions |
| min_transitions | int | 0 | Minimum transitions to keep |

**Raises:** `AttributeError` if clusterer lacks `predict()` method.

### Methods

#### run

```python
SimilarityPipeline.run(self, h5_folder: Union[str, Path], verbose: bool = True) -> 'SimilarityPipeline'
```

Process all H5 files in folder, build transition matrix, compute similarity. Chainable.

| Parameter | Type | Description |
|-----------|------|-------------|
| h5_folder | str or Path | Folder containing H5 files |
| verbose | bool | Print progress (default True) |

#### save

```python
SimilarityPipeline.save(self, results_dir: Union[str, Path]) -> dict
```

Save similarity matrix, transition matrix, and metadata to output directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| results_dir | str or Path | Output directory path |

| Return | Type | Description |
|--------|------|-------------|
| dict | Paths to saved files (similarity_matrix.npy, transition_matrix.npy, pipeline_meta.json) |

#### get_similarity_matrix

```python
SimilarityPipeline.get_similarity_matrix(self) -> np.ndarray
```

Returns the (k, k) cosine similarity matrix. Must call `run()` first.

#### get_transition_matrix

```python
SimilarityPipeline.get_transition_matrix(self) -> np.ndarray
```

Returns the (k, k) transition count matrix. Must call `run()` first.

### Usage Example

```python
from SLBHS.clustering import ThetaClusterer
from SLBHS.similarity import SimilarityPipeline

# Load pre-trained clusterer
clusterer = ThetaClusterer()
clusterer.load('/path/to/model/')

# Create pipeline
pipeline = SimilarityPipeline(clusterer=clusterer, k=1024)

# Run on all H5 files
pipeline.run(h5_folder='/path/to/h5/')

# Save results
pipeline.save('/path/to/results/')
```

---

## AttributeError: clusterer lacks predict()

Raised when clusterer passed to `SimilarityPipeline` does not have a `predict()` method.

**Example:**
```python
from SLBHS.similarity import SimilarityPipeline

class BadClusterer:
    pass

try:
    p = SimilarityPipeline(clusterer=BadClusterer())
except AttributeError as e:
    print(e)  # clusterer must have a 'predict' method. Got BadClusterer...
```

---

## `SLBHS.clustering.kmeans`

### `KMeansClusterer`

K-Means clusterer with inference support.

#### Constructor

```python
KMeansClusterer(X=None, results_dir=None, k=None, seed=42)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | np.ndarray | None | Training data (N, 63) |
| `results_dir` | str | None | Directory to save/load model files |
| `k` | int | None | Number of clusters |
| `seed` | int | 42 | Random seed |

#### Methods

##### `fit(k=None, seed=None, X=None, init='k-means++', n_init=10, max_iter=300, algorithm='lloyd', verbose_progress=True) -> Tuple[np.ndarray, np.ndarray]`

Run standard K-Means clustering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | None | Number of clusters |
| `seed` | int | None | Random seed |
| `X` | np.ndarray | None | Data (N, 63). If None, uses `self.X` |
| `init` | str | 'k-means++' | Initialization method |
| `n_init` | int | 10 | Number of initializations |
| `max_iter` | int | 300 | Max iterations per run |
| `algorithm` | str | 'lloyd' | Clustering algorithm ('lloyd', 'elkan', 'auto') |
| `verbose_progress` | bool | True | Show per-iteration progress |

**Returns**
- `labels`: `(N,)` int — cluster labels
- `centers`: `(k, 63)` float64 — cluster centers

##### `elbow(k_range, n_init=10, max_iter=300, seed=None) -> list`

Compute inertia for a range of k values.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_range` | list[int] | — | List of k values to try |
| `n_init` | int | 10 | Number of initializations |
| `max_iter` | int | 300 | Max iterations per run |
| `seed` | int | None | Random seed |

**Returns**
- `list[tuple]` — `[(k, inertia), ...]`

##### `silhouette(k_range, n_samples=5000, n_init=10, max_iter=300, seed=None) -> list`

Compute average silhouette score for a range of k values.
Subsamples `n_samples` frames for speed.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_range` | list[int] | — | List of k values to try |
| `n_samples` | int | 5000 | Max frames to subsample |
| `n_init` | int | 10 | Number of initializations |
| `max_iter` | int | 300 | Max iterations per run |
| `seed` | int | None | Random seed |

**Returns**
- `list[tuple]` — `[(k, score), ...]`

##### `save(results_dir=None, prefix='kmeans') -> dict`

Save labels, centers, and meta.json.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | str | None | Output directory |
| `prefix` | str | 'kmeans' | Filename prefix |

**Returns**
- `dict` with keys: `labels`, `centers`, `meta` (file paths)

##### `load(results_dir=None) -> Tuple[np.ndarray, np.ndarray]`

Load labels and centers from disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | str | None | Directory containing labels.npy, centers.npy, kmeans_meta.json |

**Returns**
- `labels`: `(N,)` int
- `centers`: `(k, 63)` float64

##### `save_model(results_dir=None, prefix='kmeans') -> dict`

Save sklearn model (KMeans + StandardScaler) to joblib files for inference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | str | None | Output directory |
| `prefix` | str | 'kmeans' | Filename prefix |

**Returns**
- `dict` with keys: `model`, `scaler` (paths to saved files)

##### `load_model(results_dir=None, prefix='kmeans') -> None`

Load sklearn model for inference. After loading, call `predict()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | str | None | Model directory |
| `prefix` | str | 'kmeans' | Filename prefix |

##### `fit_minibatch(k=None, seed=None, X=None, init='k-means++', n_init=3, max_iter=300, batch_size=5000, reassignment_ratio=0.01, max_no_improvement=10, verbose_progress=True) -> Tuple[np.ndarray, np.ndarray]`

Run MiniBatch K-Means (faster for large datasets).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | None | Number of clusters |
| `seed` | int | None | Random seed |
| `X` | np.ndarray | None | Data (N, 63). If None, uses `self.X` |
| `init` | str | 'k-means++' | Initialization method |
| `n_init` | int | 3 | Number of initializations |
| `max_iter` | int | 300 | Max iterations |
| `batch_size` | int | 5000 | Mini-batch size |
| `reassignment_ratio` | float | 0.01 | Reassignment ratio |
| `max_no_improvement` | int | 10 | Stop if inertia doesn't improve for this many iterations |
| `verbose_progress` | bool | True | Show per-iteration progress |

**Returns**
- `labels`: `(N,)` int — cluster labels
- `centers`: `(k, 63)` float64 — cluster centers

##### `fit_cosine_minibatch(k=None, seed=None, X_combined=None, scaler=None, init='k-means++', n_init=3, max_iter=300, batch_size=5000, reassignment_ratio=0.01, max_no_improvement=10, verbose_progress=True) -> Tuple[np.ndarray, np.ndarray]`

Run MiniBatch K-Means on combined features (63D scaled raw + 15D cosine = 78D).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | None | Number of clusters |
| `seed` | int | None | Random seed |
| `X_combined` | np.ndarray | None | `(N, 78)` scaled_raw + cosine features |
| `scaler` | StandardScaler | None | Fitted StandardScaler for raw features |
| `init` | str | 'k-means++' | Initialization method |
| `n_init` | int | 3 | Number of initializations |
| `max_iter` | int | 300 | Max iterations |
| `batch_size` | int | 5000 | Mini-batch size |
| `reassignment_ratio` | float | 0.01 | Reassignment ratio |
| `max_no_improvement` | int | 10 | Stop if no improvement for this many iterations |
| `verbose_progress` | bool | True | Show per-iteration progress |

**Returns**
- `labels`: `(N,)` int — cluster labels
- `centers`: `(k, 78)` float64 — cluster centers

**Examples**

```python
from SLBHS.clustering.kmeans import KMeansClusterer
from SLBHS.clustering.feature_transform import compute_cosine_features
from sklearn.preprocessing import StandardScaler

# Build 78D combined features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)          # (N, 63)
X_cosine = compute_cosine_features(X_raw)       # (N, 15)
X_combined = np.hstack([X_scaled, X_cosine * 3]) # (N, 78)

kc = KMeansClusterer(X=X_raw)
kc.fit_cosine_minibatch(k=512, X_combined=X_combined, scaler=scaler)
```

##### `predict(X_new) -> np.ndarray`

Classify new data. Requires `load_model()` or a `fit*()` method called first.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_new` | `(M, 63)` float32 | New hand pose vectors |

**Returns**
- `(M,)` int — cluster labels (0 to k-1)

**Examples**

```python
from SLBHS.clustering.kmeans import KMeansClusterer

kc = KMeansClusterer()
kc.load_model('results')
labels = kc.predict(new_data)
```

---

## `SLBHS.clustering.theta_clusterer`

### `ThetaClusterer`

Hand pose classifier based on circular angle features with fuzzy coarse coding.

**Three-stage pipeline:**
1. Theta Extraction — 9 bending angles + 4 spread angles from aligned_63d
2. Fuzzy Coarse Coding — angles quantized via overlapping state bands → bitstrings
3. Frequency Classification — each unique bitstring is a class; OOV fallback by Hamming similarity

**Input:** `X: np.ndarray (N, 63)` — aligned_63d from buildDatasetFromVideo.py

**Output:** `labels: np.ndarray (N,) int32` — class label IDs in [0, n_classes_-1]

#### Constructor

```python
ThetaClusterer(results_dir=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | str | None | Directory for model artifacts. Defaults to `<module_dir>/results` |

#### Methods

##### `fit`

```python
ThetaClusterer.fit(self, h5_folder=None, X=None, top_k=10000, verbose=True) -> self
```

Train the classifier.

**New mode (H5 folder):**
```python
clf.fit(h5_folder='/path/to/h5/folder/', top_k=10000)
```
Scans all `*.h5` files, extracts thetas, builds top-k frequency table. High frequency → small label ID.

**Legacy mode (array):**
```python
clf.fit(X=np.array)  # (N, 63) aligned poses
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h5_folder` | str | None | Path to folder containing H5 files (new mode) |
| `X` | np.ndarray | None | (N, 63) aligned poses (legacy mode) |
| `top_k` | int | 10000 | Maximum number of classes to keep (new mode) |
| `verbose` | bool | True | Print progress |

##### `predict`

```python
ThetaClusterer.predict(self, X_new) -> np.ndarray
```

Predict class labels for new aligned_63d data.

```python
labels = clf.predict(X_new)  # X_new: (M, 63) → labels: (M,)
```

**OOV handling:** If bitstring is unknown, find the stored class with most shared 1-bits (Hamming similarity). Ties broken by higher frequency.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_new` | np.ndarray | (M, 63) aligned hand poses |

| Return | Type | Description |
|--------|------|-------------|
| labels | np.ndarray | (M,) int32 class IDs in [0, n_classes_-1] |

##### `save`

```python
ThetaClusterer.save(self, path=None) -> dict
```

Persist model to directory.

```python
clf.save('/path/to/model/')
```

Supports both legacy (byte-keyed) and new (string-keyed) architectures. Directory layout:
- `theta_labels_str.json` — bitstring (str) → label_id
- `theta_ltb_str.json` — label_id → bitstring
- `theta_counts_str.json` — label_id → count
- `theta_meta.json` — metadata including n_classes_, top_k_

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | None | Output directory. Defaults to self.results_dir |

| Return | Type | Description |
|--------|------|-------------|
| dict | {'meta': meta_path} |

##### `load`

```python
ThetaClusterer.load(self, path=None) -> self
```

Load model artifacts from directory.

```python
clf.load('/path/to/model/')
```

Auto-detects architecture (legacy byte-keyed or new string-keyed).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | None | Input directory. Defaults to self.results_dir |

##### `histogram`

```python
ThetaClusterer.histogram(self, n=20) -> list[dict]
```

Return top N most frequent encoding statistics.

```python
hist = clf.histogram(n=20)
# [{'rank': 1, 'label': 0, 'bitstring': '000010...', 'count': 5717, 'bending_states': [...], 'spread_states': [...]}]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | 20 | Number of top entries to return |

| Return | Type | Description |
|--------|------|-------------|
| list[dict] | List of dicts with keys: rank, label, bitstring, count, bending_states, spread_states |

##### `generate_report`

```python
ThetaClusterer.generate_report(self, h5_folder, output_path, top_n=10, samples_per_class=3) -> str
```

Generate analysis report with top N encoding sample coordinates.

```python
clf.generate_report(
    h5_folder='/path/to/h5/',
    output_path='/path/to/report.txt',
    top_n=10,
    samples_per_class=2
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h5_folder` | str | required | Path to H5 folder (for raw coordinates) |
| `output_path` | str | required | Output report file path |
| `top_n` | int | 10 | Report only top_n rankings |
| `samples_per_class` | int | 3 | Number of samples per class |

| Return | Type | Description |
|--------|------|-------------|
| output_path | str | Path to the generated report file (e.g., `/path/to/report.txt`) |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_classes_` | int | Number of classes after pruning |
| `top_k_` | int | top_k value used during fit |
| `results_dir` | str | Directory for model artifacts |

#### Angle Extraction

##### Bending Angles (9 angles)

Points are 0-based MediaPipe landmark indices. Formula: `θ = arccos((A-B)·(C-B) / (|A-B|·|C-B|))`

| Index | Finger | Joint | Points | Description |
|-------|--------|-------|--------|-------------|
| 0 | thumb | MCP→IP→TIP | 2→3→4 | Thumb curling |
| 1 | index | PIP | 5→6→7 | Index PIP |
| 2 | index | DIP | 6→7→8 | Index DIP |
| 3 | middle | PIP | 9→10→11 | Middle PIP |
| 4 | middle | DIP | 10→11→12 | Middle DIP |
| 5 | ring | PIP | 13→14→15 | Ring PIP |
| 6 | ring | DIP | 14→15→16 | Ring DIP |
| 7 | pinky | PIP | 17→18→19 | Pinky PIP |
| 8 | pinky | DIP | 18→19→20 | Pinky DIP |

##### Spread Angles (4 angles)

Angle between adjacent finger PIP joints (computed via wrist-relative vectors). Uses PIP (not MCP) for better discrimination.

| Index | Combination | Description |
|-------|-------------|-------------|
| 0 | Thumb PIP → Index PIP | Thumb-Index spread |
| 1 | Index PIP → Middle PIP | Index-Middle spread |
| 2 | Middle PIP → Ring PIP | Middle-Ring spread |
| 3 | Ring PIP → Pinky PIP | Ring-Pinky spread |

#### Fuzzy Coarse Coding

**Bending angles (0°–180°):** 5 states with 30° overlap

| State | Range | Meaning |
|-------|-------|---------|
| 0 | 0°–60° | Tight fist |
| 1 | 30°–90° | Large bend |
| 2 | 60°–120° | Half bent |
| 3 | 90°–150° | Slight bend |
| 4 | 120°–180° | Straight |

**Spread angles (0°–60°):** 5 states with overlap

| State | Range | Meaning |
|-------|-------|---------|
| 0 | 0°–15° | Closed |
| 1 | 7.5°–22.5° | Converging |
| 2 | 15°–30° | Middle |
| 3 | 22.5°–37.5° | Spread |
| 4 | 30°–60° | Fully open |

**Bitstring length:** 65 bits (13 angles × 5 states). A single angle can fall into 1–2 adjacent states (multi-hot encoding). Maximum 2 consecutive 1s (impossible to have 3 due to 30° overlap design).

#### Usage Example

```python
from SLBHS.clustering.theta_clusterer import ThetaClusterer

# Train
clf = ThetaClusterer()
clf.fit('/path/to/h5/folder/', top_k=10000)
clf.save('/path/to/model/')

# Predict
clf.load('/path/to/model/')
labels = clf.predict(X_new)  # (M, 63) → (M,)

# Statistics
hist = clf.histogram(n=20)

# Report
clf.generate_report(
    h5_folder='/path/to/h5/folder/',
    output_path='/path/to/report.txt',
    top_n=20,
    samples_per_class=2
)
```

---

## `SLBHS.viz.plot_config`

### `get_cluster_colors(labels, cmap_name='gist_rainbow', n_clusters=512) -> list`

Return list of RGBA colors for a list/array of cluster labels.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | list or np.ndarray | — | Cluster labels |
| `cmap_name` | str | 'gist_rainbow' | Matplotlib colormap name |
| `n_clusters` | int | 512 | Number of distinct colors to generate |

**Returns**
- `list` — RGBA color tuples

**Examples**

```python
from SLBHS.viz.plot_config import get_cluster_colors

colors = get_cluster_colors(labels, cmap_name='gist_rainbow', n_clusters=512)
```

---

## `SLBHS.viz.gen_samples`

### `generate_samples(k, results_dir, samples_per_cluster=10, seed=42, create_zip=False) -> None`

Generate sample hand-pose images for each cluster.
Each image shows `samples_per_cluster` randomly selected frames from the cluster.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | — | Number of clusters (K-Means k) |
| `results_dir` | str | — | Directory containing `labels.npy`, `centers.npy`, and `aligned_63d_multi_*.npz` |
| `samples_per_cluster` | int | 10 | Number of sample frames per cluster |
| `seed` | int | 42 | Random seed for reproducibility |
| `create_zip` | bool | False | Also create a ZIP archive of all images |

**Output**
- `results_dir/clusters_{N}samples/cluster_{cluster_id:04d}.png` for each cluster
- Optionally: `results_dir/clusters_{k}_{N}samples.zip`

**Examples**

```python
from SLBHS.viz.gen_samples import generate_samples

generate_samples(
    k=1024,
    results_dir='TWSLT/results',
    samples_per_cluster=10,
    seed=42,
    create_zip=True
)
```

```bash
# CLI usage
python -m SLBHS.viz.gen_samples --k 1024 --results-dir TWSLT/results --samples-per-cluster 10
python -m SLBHS.viz.gen_samples --k 1024 --results-dir TWSLT/results --samples-per-cluster 10 --output-zip
```

### `parse_args() -> argparse.Namespace`

Parse command-line arguments for `gen_samples`.

**Returns**
- `argparse.Namespace` with the following attributes:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 1024 | Number of clusters (K-Means k) |
| `results_dir` | str | None | Results directory containing `labels.npy`, `centers.npy`, and `aligned_63d_multi_*.npz` |
| `samples_per_cluster` | int | 10 | Number of sample frames per cluster |
| `seed` | int | 42 | Random seed for reproducibility |
| `output_zip` | bool | False | Whether to create a ZIP archive of all images |

**Examples**

```bash
python -m SLBHS.viz.gen_samples --k 1024 --results-dir TWSLT/results --samples-per-cluster 10
python -m SLBHS.viz.gen_samples --k 1024 --results-dir TWSLT/results --samples-per-cluster 10 --output-zip
```

### `MEDIAPIPE_CONNECTIONS`

```python
MEDIAPIPE_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
```

List of 21 hand landmark index pairs defining the MediaPipe hand skeleton topology. Each tuple `(j, k_conn)` represents a bone connection from landmark `j` to landmark `k_conn`.

**Landmark layout**

| Landmark | Body part |
|----------|-----------|
| 0 | Wrist |
| 1–4 | Thumb (base → tip) |
| 5–8 | Index finger |
| 9–12 | Middle finger |
| 13–16 | Ring finger |
| 17–20 | Pinky finger |

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
| `data_dir` | str | `None` → auto-resolves to `~/.openclaw/media/inbound/` | H5 file path, directory or glob pattern; `None` defaults to `~/.openclaw/media/inbound/` |
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

---

## `SLBHS.viz.similarity_heatmap`

Standalone entry point for similarity heatmap visualization.

### Usage

```bash
cd /home/ubuntu/.openclaw/workspace-coding/SLBHS && source venv/bin/activate
python SLBHS/viz/similarity_heatmap.py --input /path/to/symmetrized_matrix.npy --output heatmap.png
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `~/.openclaw/media/inbound/symmetrized_matrix---dec21f33-b895-4377-a969-a4a6b7f7493c.npy` | Path to symmetrized matrix (.npy) |
| `--output` | `~/.openclaw/media/inbound/similarity_heatmap.png` | Output PNG path |

### What it does

1. Loads symmetrized transition matrix W
2. Row-normalizes to probability matrix M_prob
3. For each pair (i, j): computes cosine similarity of M_prob[i] and M_prob[j] **with dimensions i and j excluded** — this prevents self-correlation from inflating similarity scores
4. Renders heatmap using magma colormap

### Output

- PNG heatmap saved to `--output`
- `similarity_matrix_heatmap.npy` saved alongside (same directory as `--output`)

### Example

```bash
python SLBHS/viz/similarity_heatmap.py \
    --input results/symmetrized_matrix.npy \
    --output results/similarity_heatmap.png
```

