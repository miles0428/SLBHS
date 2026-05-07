# BLUEPRINT.md — SLBHS Super Cluster Pipeline v3

**目標：** 透過時序轉移相似度找出物理等價的手型（Super Clusters）

**Pipeline v3：完全吃 K-Means model（.joblib），不碰 training。**

---

## 輸入

| 來源 | 內容 |
|------|------|
| H5 原始檔 | `aligned_63d`(N,63) + `x_vec`/`y_vec`/`z_vec`(N,3) |
| `kmeans_model.joblib` | KMeans 模型（預訓練），用於預測 |
| `kmeans_scaler.joblib` | StandardScaler（預訓練）|

---

## Pipeline 流程（v3）

```
H5 → KMeansClusterer.load_model(model_dir) → predict() → labels → C → S → SuperCluster
```

### Step 1：K-Means 預測（只 load，不 training）
- `KMeansClusterer.load_model(model_dir)` — 載入預訓練模型
- `KMeansClusterer.predict(X)` → Token_ID（N,）0~k-1

### Step 2：Hand Labeling（手性區分）
- `cross(vec_x, vec_y) dot vec_z`
- dot < 0 → "L"（左手）
- dot >= 0 → "R"（右手）

### Step 3：建立轉移矩陣 C[k×k]
- 左右手分開統計：`n → n+delta_t` 轉移
- `C = C_left + C_right`

### Step 4：對稱化 + 列歸一化
- `W = (C + C.T) / 2`
- `M_ij = W_ij / Σ_k(W_ik)` → 機率矩陣

### Step 5：相似度矩陣 S[k×k]
- `S_ij = cos(M_i, M_j)`

### Step 6：Super Cluster Extraction
- `S_ij > tau` → 建邊
- Connected Components → N 個 Super_Clusters

---

## 兩種模式

### `--h5` 模式（單一 H5，debug）
```bash
python run_pipeline.py \
  --h5 /path/to/single.h5 \
  --model-dir /path/to/kmeans/model/ \
  --k 1024 \
  --delta-t 10 \
  --tau 0.9 \
  --output results/
```
直接走 `fit()` 完成。

### `--folder` 模式（批次多 H5）
```bash
python run_pipeline.py \
  --folder /path/to/h5/folder \
  --model-dir /path/to/kmeans/model/ \
  --k 1024 \
  --delta-t 10 \
  --tau 0.9 \
  --output results/
```
每個 H5 走 `update()`（累加 C），最後 `finalize()` 計算 S + BigClusterer。

---

## CLI 參數

| 參數 | 說明 | 預設 |
|------|------|------|
| `--folder` | H5 folder（批次） | - |
| `--h5` | 單一 H5（debug） | - |
| `--model-dir` | KMeans 模型目錄（k 從 meta.json 讀取）| - |
| `--output` | 產出資料夾 | results |
| `--k` | ~~K-Means 群數~~（已移除，k 由 model meta.json 決定）| ~~1024~~ |
| `--tau` | 相似度閾值 | 0.9 |
| `--delta-t` | 轉移間隔 | 10 |
| `--min-transitions` | 最低轉移次數 | 0 |
| `--symmetrize` | 是否對稱化 | True |

---

## OOP 架構

### Class 一覽表

| Class | 模組 | 職責 |
|-------|------|------|
| `HandLabeler` | `clustering.super_cluster_pipeline` | vec_x×vec_y dot vec_z 區分 L/R |
| `TransitionCounter` | `clustering.super_cluster_pipeline` | 建立 C[k×k] 轉移矩陣（分左右手加總）|
| `SimilarityMatrix` | `clustering.super_cluster_pipeline` | 計算 S = cosine(M_prob) |
| `BigClusterer` | `clustering.super_cluster_pipeline` | Super Cluster Extraction（Connected Components）|
| `BigClusterPipeline` | `clustering.super_cluster_pipeline` | 串接全部，一鍵產出 |
| `KMeansClusterer` | `clustering.kmeans` | K-Means 預測 + 模型 load/save |

---

## BigClusterPipeline API（v3）

```python
class BigClusterPipeline:
    def __init__(self, k=1024, tau=0.9, delta_t=10,
                 min_transitions=0, symmetrize=True,
                 model_dir=None, results_dir=None):
        """
        model_dir: KMeans 模型目錄（只吃預訓練，不 training）
        results_dir: pipeline 產出路徑
        """

    def fit(self, X, x_vec, y_vec, z_vec,
            labels=None, k=None, tau=None,
            min_transitions=None, delta_t=None,
            symmetrize=None, model_dir=None,
            results_dir=None) -> self:
        """單一 H5 直接完成"""

    def update(self, X, x_vec, y_vec, z_vec) -> self:
        """吃一個 H5，累加 C。Model 只 load 一次。"""

    def finalize(self, tau=None) -> self:
        """所有 H5 跑完後呼叫，compute S + BigClusterer"""

    def save(self, results_dir) -> None:
        """儲存產出"""
```

---

## 輸出（Validation Criteria）

| 產出 | 路徑 | 成功特徵 |
|------|------|----------|
| Super Cluster mapping | `results/super_cluster_map.json` | JSON，含 N 個 Super_Cluster |
| S 矩陣 | `results/similarity_matrix.npy` | (k, k)，無 NaN |
| C 矩陣 | `results/transition_matrix.npy` | (k, k) |
| W 矩陣 | `results/symmetrized_matrix.npy` | (k, k)，對稱 |
| pipeline summary | `results/pipeline_phase2.json` | JSON 摘要 |

**異常指標：** 出現 "Error" / "NaN" → 失敗

---

_Last updated: 2026-05-07_
