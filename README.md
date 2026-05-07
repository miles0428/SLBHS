# SLBHS — Sign Language Basic Handshapes

**超級手型分類（Super Cluster）Pipeline**：從手勢座標資料中，透過時序轉移相似度找出物理等價的手型群組。

## 安裝

```bash
# 直接從 GitHub 安裝（不需要 clone）
pip install git+https://github.com/miles0428/SLBHS.git

# 開發用：clone 後本地安裝
git clone https://github.com/miles0428/SLBHS.git
cd SLBHS
pip install .
```

## 快速開始

```bash
# 完整 pipeline：K-Means → SuperCluster → UMAP → 視覺化
python -m SLBHS.run_visualization --k 512 --n-super 20 --format both

# 使用 cosine features（63D scaled + 15D bone-angle = 78D combined）
python -m SLBHS.run_visualization --k 512 --n-super 20 --cosine-features

# 跳過 K-Means，使用快取結果
python -m SLBHS.run_visualization --skip-kmeans --skip-super

# 自訂 UMAP 採樣數量
python -m SLBHS.run_visualization --k 512 --n-super 20 --overview-umap-n 10000 --sc-umap-n 2000
```

## 資料格式

### 輸入 H5 檔案結構

```
file_crop---xxxxxxxxxx.h5
├── aligned_63d    (N, 63)   float32  — 21 個 landmark × 3 軸 = 63 維
├── x_vec         (N, 3)    float32  — x 軸向量
├── y_vec         (N, 3)    float32  — y 軸向量
├── z_vec         (N, 3)    float32  — z 軸向量
└── is_mirror     (N,)      bool     — 是否為鏡像（可選）
```

### 樣本 H5 路徑

```
/home/ubuntu/.openclaw/media/inbound/2022_12_12_14_00_中央流行疫情指揮中心嚴重特殊傳染性肺炎記者會_crop---87414a0f-15ac-4b38-8710-15c43fa52793.h5
```

---

## API 使用範例

### 完整 Super Cluster Pipeline

```python
import h5py
import numpy as np
from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

# 1. 讀取 H5 資料
h5_path = '/path/to/file_crop---xxxxxxxx.h5'
with h5py.File(h5_path, 'r') as f:
    X = f['aligned_63d'][:]         # (N, 63)
    x_vec = f['x_vec'][:]           # (N, 3)
    y_vec = f['y_vec'][:]           # (N, 3)
    z_vec = f['z_vec'][:]           # (N, 3)

# 2. 建立並執行 Pipeline（cosine feature 模式）
pipeline = BigClusterPipeline(
    k=512,               # K-Means 群數（需與 KMeansClusterer 模型一致）
    tau=0.9,             # 相似度閾值（S_ij > tau → 建邊）
    cosine_features=True,  # 使用 cosine feature（63D + 15D = 78D）
    results_dir='results'
)

pipeline.fit(
    X, x_vec, y_vec, z_vec,
    cosine_features=True,
    results_dir='results'
)
pipeline.save('results')

print(f"Super Clusters: {pipeline.big_clusterer.n_clusters}")
```

### 獨立使用各 Class

```python
from SLBHS.clustering.super_cluster_pipeline import (
    HandLabeler, TransitionCounter, SimilarityMatrix, BigClusterer
)

# Step 1：手性標記（L/R）
labeler = HandLabeler()
hand_labels = labeler.fit_predict(x_vec, y_vec, z_vec)

# Step 2：建立轉移矩陣
counter = TransitionCounter(k=1024, delta_t=1, min_transitions=0)
counter.fit(token_ids, hand_labels)
C = counter.get_matrix()

# Step 3：計算相似度矩陣
sim_matrix = SimilarityMatrix()
S = sim_matrix.compute(C, symmetrize=True)

# Step 4：提取 Super Clusters
clusterer = BigClusterer(tau=0.9)
clusterer.fit(S, tau=0.9)
super_cluster_map = clusterer.get_clusters()
```

### K-Means 分群（Inference）

```python
from SLBHS.clustering.kmeans import KMeansClusterer

# 載入預訓練模型
kc = KMeansClusterer()
kc.load_model('results')  # 載入 kmeans_model.joblib + kmeans_scaler.joblib

# 對新資料分類
new_data = np.load('new_handposes.npz')['X']  # (N, 63)
labels = kc.predict(new_data)  # (N,), 0-1023
```

### 使用 Cosine Features（78D）

```python
from SLBHS.clustering.kmeans import KMeansClusterer
from SLBHS.clustering.feature_transform import compute_cosine_features

# Cosine feature 模式需要：StandardScaler(63D) + 15D bone-angle cosine
# 使用 fit_cosine_minibatch() 訓練
kc = KMeansClusterer(results_dir='results', k=512)
X_cosine = compute_cosine_features(X)  # (N, 15)
X_combined = np.hstack([scaler.fit_transform(X), X_cosine * 3])  # (N, 78)
labels, centers = kc.fit_cosine_minibatch(X_combined=X_combined, scaler=scaler)

# 預測新資料
labels = kc.predict(new_data)  # 內部自動做 63D scale + cosine 拼接
```

---

## 核心 Class 說明

### `HandLabeler`

根據 `cross(vec_x, vec_y) dot vec_z` 判斷左手/右手：

- `dot < 0` → **'L'**（左手）
- `dot >= 0` → **'R'**（右手）

```python
labeler = HandLabeler()
hand_labels = labeler.fit_predict(x_vec, y_vec, z_vec)  # (N,) 'L'/'R'
```

### `TransitionCounter`

建立 Token 轉移矩陣 C[1024×1024]：

- 遍歷左右手軌道，統計 n→n+delta_t 的轉移
- 支援 `min_transitions` 過濾低頻轉移

```python
counter = TransitionCounter(k=1024, delta_t=1, min_transitions=0)
counter.fit(token_ids, hand_labels)
C = counter.get_matrix()  # (1024, 1024)
```

### `SimilarityMatrix`

從轉移矩陣計算相似度：

1. 對稱化：`W = (C + C.T) / 2`
2. 列歸一化：`M_ij = W_ij / Σ_k(W_ik)`
3. Cosine Similarity：`S_ij = cos(M_i, M_j)`

```python
sim = SimilarityMatrix()
S = sim.compute(C, symmetrize=True)  # (1024, 1024)
```

### `BigClusterer`

根據相似度閾值提取 Super Clusters：

- `S_ij > tau` → 建邊
- Connected Components → Super Cluster ID

```python
clusterer = BigClusterer(tau=0.9)
clusterer.fit(S, tau=0.9)
cluster_map = clusterer.get_clusters()  # {token_id: super_cluster_id}
```

### `BigClusterPipeline`

串接全部流程，一鍵產出：

```python
pipeline = BigClusterPipeline(k=512, tau=0.9, cosine_features=True)
pipeline.fit(X, x_vec, y_vec, z_vec, cosine_features=True, results_dir='results')
pipeline.save('results')
```

---

## 輸出檔案

| 檔案 | 格式 | 說明 |
|------|------|------|
| `similarity_matrix.npy` | (1024, 1024) float64 | Cosine similarity 矩陣 S |
| `transition_matrix.npy` | (1024, 1024) float64 | 原始轉移計數矩陣 C |
| `symmetrized_matrix.npy` | (1024, 1024) float64 | 對稱化後的矩陣 W |
| `super_cluster_map.json` | JSON | `{token_id: super_cluster_id}` |
| `pipeline_phase2.json` | JSON | Phase 2 完整摘要 |

---

## CLI 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--k` | 512 | K-Means 群數 |
| `--n-super` | 20 | Super Cluster 數量（Hierarchical，用於視覺化）|
| `--batch-size` | 5000 | MiniBatch K-Means batch size |
| `--seed` | 42 | 隨機種子 |
| `--n-neighbors` | 30 | UMAP n_neighbors |
| `--overview-umap-n` | 10000 | Overview UMAP 採樣數 |
| `--sc-umap-n` | 2000 | 每個 Super Cluster 的 UMAP 採樣數 |
| `--results-dir` | results | 結果輸出目錄 |
| `--dpi` | 200 | PNG 解析度 |
| `--format` | png | 輸出格式：png, svg, both |
| `--cosine-features` | — | 使用 78D cosine features |
| `--skip-kmeans` | — | 跳過 K-Means（使用快取）|
| `--skip-super` | — | 跳過 SuperCluster（使用快取）|
| `--skip-umap` | — | 跳過 UMAP（使用快取）|
| `--no-verbose` | — | 隱藏 K-Means 迭代進度 |

---

## 模組架構

```
SLBHS/
├── __init__.py              # 主要 public API exports
├── version.py               # 版本資訊
├── run_visualization.py     # CLI 入口：UMAP 視覺化 pipeline
├── data/
│   └── loader.py            # DataLoader：H5 讀取與快取
├── clustering/
│   ├── __init__.py
│   ├── kmeans.py            # KMeansClusterer：K-Means 分群、預測、模型管理
│   ├── super_cluster.py     # SuperClusterer：Agglomerative（視覺化用）
│   ├── super_cluster_pipeline.py  # Phase 2 pipeline（HandLabeler, TransitionCounter, SimilarityMatrix, BigClusterer, BigClusterPipeline）
│   └── feature_transform.py  # compute_cosine_features：15D bone-angle cosine
├── viz/
│   ├── visualizer.py         # SLBHSViz：UMAP 視覺化
│   └── layout.py             # GridLayout：網格佈局
└── docs/
    ├── pipeline.md          # Pipeline 流程說明
    └── api.md               # API 詳細文件
```

---

## 數學原理

### 手性判斷（Hand Labeling）

$$\text{dot} = \sum(\text{cross}(\vec{x}, \vec{y}) \odot \vec{z})$$

- `dot < 0` → 左手（L）
- `dot >= 0` → 右手（R）

物理意義：cross(x, y) 產生垂直於手掌平面的向量，與 z 點積為正表示右手，為負表示左手。

### 轉移矩陣（Transition Matrix）

$$C_{ij} = \sum_{\text{track}} \sum_{\delta=1}^{\Delta} \mathbb{1}[Token_{t}=i \wedge Token_{t+\delta}=j]$$

分左右手各算一次後相加。

### 相似度矩陣（Similarity Matrix）

1. 對稱化：$W = \frac{C + C^T}{2}$
2. 列歸一化：$M_{ij} = \frac{W_{ij}}{\sum_k W_{ik}}$（機率矩陣）
3. Cosine Similarity：$S_{ij} = \frac{M_i \cdot M_j}{\|M_i\| \|M_j\|}$

### Super Cluster Extraction

$$S_{ij} > \tau \Rightarrow \text{建邊}$$
Connected Components → Super Clusters

---

## 限制與已知議題

- 轉移矩陣使用 `np.add.at`，大量相同 token 可能有競爭條件（但對機率結果影響小）
- `BigClusterPipeline` 的 `cosine_features=False` 模式使用 MiniBatchKMeans，非正式 training

---

## 作者

Yu-Cheng Chung

## 授權

本專案僅供研究使用。詳見 `LICENSE` 檔案。

允許：
- 使用程式碼訓練自己的模型

禁止：
- 使用預訓練模型或輸出訓練其他模型
- 用於商業用途

商業授權請聯絡作者。

_Last updated: 2026-05-07_