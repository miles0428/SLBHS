# BLUEPRINT.md — SLBHS Super Cluster Pipeline（最終版）

**目標：** 透過時序轉移相似度找出物理等價的手型（Super Clusters）

**YC 確認範圍：** K-Means 已完成（1024群，含 cosine features）。專注做相似矩陣任務。

---

## 輸入

| 來源 | 內容 |
|------|------|
| H5 原始檔 | `aligned_63d`(N,63) + `x_vec`/`y_vec`/`z_vec`(N,3) |
| `labels.npy` | K-Means Token_ID，(N,)，0-1023（預訓練模型） |
| `kmeans_model.joblib` | K-Means 模型，用於預測 |

---

## Pipeline（7 Steps）

### Step 1：K-Means 預測
- 讀取 H5 `aligned_63d`
- 使用 KMeansClusterer 預測 → Token_ID（N,）0-1023
- 或讀取預先計算好的 `labels.npy`（若 H5 對應）

### Step 2：Hand Labeling（手性區分）
- 使用 `cross(vec_x, vec_y) dot vec_z`
- dot < 0 → "L"（左手）
- dot >= 0 → "R"（右手）
- **完全 bimodal，100% 區分度**

### Step 3：左右手軌道分離
- Left_Track：hand_label == "L" 的 Token_ID 序列（保持時間序）
- Right_Track：hand_label == "R" 的 Token_ID 序列（保持時間序）

### Step 4：建立轉移矩陣 C[1024×1024]
- 左軌：統計 n→n+1 轉移（左手中相鄰兩幀）
- 右軌：統計 n→n+1 轉移（右手）
- C = C_left + C_right（分開算後加在一起）
- 使用 `scipy.sparse` 儲存

### Step 5：對稱化 + 列歸一化
- W = (C + C.T) / 2
- M_ij = W_ij / Σ_k(W_ik) → 機率矩陣

### Step 6：二次相似度矩陣 S[1024×1024]
- S_ij = cos(M_i, M_j)
- 向量化：`sklearn.metrics.pairwise.cosine_similarity(M)`

### Step 7：Super Cluster Extraction
- τ 設定（建議 0.9，視情況調低）
- S_ij > τ → 建邊
- Connected Components → N 個 Super_Clusters
- 輸出：`Map: Token_ID → Super_Cluster_ID`

---

## Pipeline（批次多 H5 模式）

支援一次處理多個 H5 檔案，累加轉移矩陣後再 compute S + BigClusterer。

### 流程：
```
for each H5:
    pipeline.update(X, x_vec, y_vec, z_vec)  # 累加進 TransitionCounter.C
pipeline.finalize(tau)                        # compute S + BigClusterer
pipeline.save(results_dir)                    # 寫出產出
```

### update() — 累加 C（不改變其他狀態）
- HandLabeler.fit_predict(x_vec, y_vec, z_vec) → hand_labels
- KMeansClusterer.predict(X) → labels（cosine_features 模式）
- TransitionCounter.update(labels, hand_labels) → 累加進 self.C

### finalize() — 計算 S + BigClusterer
- SimilarityMatrix.compute(C) → S
- BigClusterer.fit(S, tau) → super_cluster_map

### CLI 使用方式：
```bash
python run_pipeline.py --folder /path/to/h5_folder --k 1024 --tau 0.9 --delta-t 10 --cosine-features --results-dir results
```

---

## OOP 架構

### Class 一覽表

| Class | 模組 | 職責 |
|-------|------|------|
| `HandLabeler` | `clustering.super_cluster_pipeline` | 用 vec_x×vec_y dot vec_z 區分 L/R |
| `TransitionCounter` | `clustering.super_cluster_pipeline` | 建立 C[1024×1024] 轉移矩陣（分左右手加總）|
| `SimilarityMatrix` | `clustering.super_cluster_pipeline` | 計算 S = cosine(M_prob) |
| `BigClusterer` | `clustering.super_cluster_pipeline` | Super Cluster Extraction（Connected Components）|
| `BigClusterPipeline` | `clustering.super_cluster_pipeline` | 串接全部，一鍵產出 |
| `KMeansClusterer` | `clustering.kmeans` | K-Means 分群、預測、模型儲存 |
| `SuperClusterer` | `clustering.super_cluster` | Hierarchical Agglomerative（純視覺化用）|
| `DataLoader` | `data.loader` | H5 讀取與快取 |

### 詳細 Class 規格

```python
class HandLabeler:
    """用 vec_x×vec_y dot vec_z 區分 L/R"""
    def fit_predict(self, x_vec, y_vec, z_vec) -> np.ndarray:
        # 輸出：(N,) 'L'/'R'

class TransitionCounter:
    """建立 C[1024×1024] 轉移矩陣（分左右手加總）"""
    def __init__(self, k=1024, delta_t=1, min_transitions=0)
    def fit(self, token_ids, hand_labels, delta_t=None, min_transitions=None) -> self:
        # 回傳：self.C (1024, 1024)

class SimilarityMatrix:
    """計算 S = cosine(M_prob)"""
    def compute(self, M, symmetrize=True) -> np.ndarray:
        # 輸出：S (1024, 1024)

class BigClusterer:
    """Super Cluster Extraction（根據相似度閾值）"""
    def __init__(self, tau=0.9)
    def fit(self, S, tau=None) -> self:
        # 輸出：self.cluster_map {int: int}

class BigClusterPipeline:
    """串接全部，一鍵產出"""
    def fit(self, X, x_vec, y_vec, z_vec,
            labels=None, k=None, tau=None,
            cosine_features=True, results_dir=None,
            min_transitions=0, delta_t=1, symmetrize=True) -> self
    def save(self, results_dir) -> None:
        # 產出檔案
```

---

## 輸出（Validation Criteria）

| 產出 | 路徑 | 成功特徵 |
|------|------|----------|
| Super Cluster mapping | `results/super_cluster_map.json` | JSON，含 N 個 Super_Cluster |
| S 矩陣 | `results/similarity_matrix.npy` | (1024, 1024)，無 NaN |
| M 矩陣 | `results/transition_matrix.npy` | (1024, 1024)，row sum ≈ 1 |
| W 矩陣 | `results/symmetrized_matrix.npy` | (1024, 1024)，對稱 |
| pipeline summary | `results/pipeline_phase2.json` | JSON，Phase 2 完整摘要 |

**異常指標：** 出現 "Error" / "NaN" → 失敗

---

## 執行方式

### Python API

```python
from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

pipeline = BigClusterPipeline(k=512, tau=0.9, cosine_features=True, results_dir='results')
pipeline.fit(X, x_vec, y_vec, z_vec, cosine_features=True, results_dir='results')
pipeline.save('results')
```

### CLI 入口

```bash
python run_pipeline.py --k 512 --tau 0.9 --cosine-features --results-dir results
python run_pipeline.py --h5 /path/to/file.h5 --k 512 --tau 0.9
```

---

_Last updated: 2026-05-07_