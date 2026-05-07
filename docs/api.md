# API 參考文件

## 主要 Module：`SLBHS.clustering.super_cluster_pipeline`

---

## `HandLabeler`

根據 `cross(vec_x, vec_y) dot vec_z` 判斷左手/右手。

### Methods

#### `fit_predict(x_vec, y_vec, z_vec) -> np.ndarray`

擬合並預測手性標籤。

**Parameters**

| 名稱 | 型別 | 說明 |
|------|------|------|
| `x_vec` | (N, 3) float32 | x 軸向量 |
| `y_vec` | (N, 3) float32 | y 軸向量 |
| `z_vec` | (N, 3) float32 | z 軸向量 |

**Returns**
- `(N,) '<U1'` — 'L' 或 'R'

**Example**

```python
from SLBHS.clustering.super_cluster_pipeline import HandLabeler

labeler = HandLabeler()
hand_labels = labeler.fit_predict(x_vec, y_vec, z_vec)
print(f"L={np.sum(hand_labels=='L')}, R={np.sum(hand_labels=='R')}")
```

---

## `TransitionCounter`

建立 Token 轉移矩陣 C[1024×1024]。

### Constructor

```python
TransitionCounter(k=1024, delta_t=1, min_transitions=0)
```

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `k` | int | 1024 | Token 總數 |
| `delta_t` | int | 1 | 往前看多少步 |
| `min_transitions` | int | 0 | 最低轉移次數 |

### Methods

#### `fit(token_ids, hand_labels, delta_t=None, min_transitions=None) -> self`

建立轉移矩陣。

**Parameters**

| 名稱 | 型別 | 說明 |
|------|------|------|
| `token_ids` | (N,) int | Token_ID，0-1023 |
| `hand_labels` | (N,) '<U1' | 'L' 或 'R' |
| `delta_t` | int or None | 覆蓋 constructor 的 delta_t |
| `min_transitions` | int or None | 覆蓋 constructor 的 min_transitions |

**Returns**
- `self`

#### `get_matrix() -> np.ndarray`

取得轉移矩陣。

**Returns**
- `(k, k)` float64 — 轉移計數矩陣（未歸一化）

---

## `SimilarityMatrix`

從轉移矩陣計算相似度矩陣 S。

### Methods

#### `compute(M, symmetrize=True) -> np.ndarray`

計算相似度矩陣。

**Parameters**

| 名稱 | 型別 | 說明 |
|------|------|------|
| `M` | (k, k) float64 | 原始轉移計數矩陣 C |
| `symmetrize` | bool | 是否對稱化（預設 True） |

**Returns**
- `(k, k)` float64 — cosine similarity 矩陣 S

**流程**
1. 對稱化：`W = (M + M.T) / 2`
2. 列歸一化：`M_ij = W_ij / Σ_k(W_ik)`
3. Cosine Similarity：`S_ij = cos(M_i, M_j)`

---

## `BigClusterer`

根據相似度閾值提取 Super Clusters。

### Constructor

```python
BigClusterer(tau=0.9)
```

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `tau` | float | 0.9 | 相似度閾值（0.0-1.0）|

### Methods

#### `fit(S, tau=None) -> self`

從相似度矩陣 S 提取 Super Clusters。

**Parameters**

| 名稱 | 型別 | 說明 |
|------|------|------|
| `S` | (k, k) float64 | cosine similarity 矩陣 |
| `tau` | float or None | 覆蓋 constructor 的 tau |

**Returns**
- `self`

#### `get_clusters() -> dict`

取得 Super Cluster mapping。

**Returns**
- `dict` — `{token_id: super_cluster_id}`

---

## `BigClusterPipeline`

串接全部流程，一鍵產出 Phase 2 結果。

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

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `k` | int | 512 | K-Means 群數 |
| `tau` | float | 0.9 | 相似度閾值 |
| `delta_t` | int | 1 | 轉移間隔 |
| `cosine_features` | bool | True | 是否使用 cosine feature（78D）|
| `min_transitions` | int | 0 | 最低轉移次數 |
| `symmetrize` | bool | True | 是否對稱化 |
| `results_dir` | str or None | None | KMeansClusterer 模型路徑 |

### Methods

#### `fit(X, x_vec, y_vec, z_vec, labels=None, k=None, tau=None, cosine_features=None, min_transitions=None, delta_t=None, symmetrize=None, results_dir=None) -> self`

執行完整 pipeline。

**Parameters**

| 名稱 | 型別 | 說明 |
|------|------|------|
| `X` | (N, 63) float32 | aligned_63d 手勢資料 |
| `x_vec` | (N, 3) float32 | x 軸向量 |
| `y_vec` | (N, 3) float32 | y 軸向量 |
| `z_vec` | (N, 3) float32 | z 軸向量 |
| `labels` | (N,) int or None | Token_ID；若 None 則依 cosine_features 模式產生 |
| `k` | int or None | 覆蓋 constructor 的 k |
| `tau` | float or None | 覆蓋 constructor 的 tau |
| `cosine_features` | bool or None | 覆蓋 constructor 的 cosine_features |
| `min_transitions` | int or None | 覆蓋 constructor 的 min_transitions |
| `delta_t` | int or None | 覆蓋 constructor 的 delta_t |
| `symmetrize` | bool or None | 覆蓋 constructor 的 symmetrize |
| `results_dir` | str or None | 覆蓋 constructor 的 results_dir |

**Returns**
- `self`

#### `save(results_dir) -> None`

儲存產出到 results_dir。

**產出檔案**

| 檔案 | 格式 | 說明 |
|------|------|------|
| `similarity_matrix.npy` | (k, k) float64 | Cosine similarity 矩陣 S |
| `transition_matrix.npy` | (k, k) float64 | 原始轉移計數矩陣 C |
| `symmetrized_matrix.npy` | (k, k) float64 | 對稱化後的矩陣 W |
| `super_cluster_map.json` | JSON | `{token_id: super_cluster_id}` |
| `pipeline_phase2.json` | JSON | Phase 2 完整摘要 |

---

## `SLBHS.clustering.kmeans`

### `KMeansClusterer`

K-Means 分群器，支援 inference。

#### Constructor

```python
KMeansClusterer(X=None, results_dir=None, k=None, seed=42)
```

#### Methods

##### `load_model(results_dir=None, prefix='kmeans') -> None`

載入預訓練模型（joblib 格式）。

##### `predict(X_new) -> np.ndarray`

對新資料分類。

**Parameters**
- `X_new`: `(M, 63)` float32 — 新手勢資料

**Returns**
- `(M,)` int — cluster labels（0 到 k-1）

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

H5 檔案讀取與快取。

#### Constructor

```python
DataLoader(data_dir=None, cache_dir=None)
```

| 參數 | 型別 | 說明 |
|------|------|------|
| `data_dir` | str | H5 檔案路徑、目錄或 glob pattern |
| `cache_dir` | str | 快取目錄 |

#### Methods

##### `load(force_reload=False) -> Tuple[np.ndarray, dict]`

載入 aligned_63d 資料。

**Returns**
- `X`: `(N, 63)` float32
- `meta`: dict（含 `n_frames`, `files`, `n_files`, `per_file_frames`）

**Example**

```python
from SLBHS.data.loader import DataLoader

loader = DataLoader(data_dir='/path/to/h5/')
X, meta = loader.load()
print(f"Loaded {meta['n_frames']} frames from {meta['n_files']} files")
```

---

_Last updated: 2026-05-07_