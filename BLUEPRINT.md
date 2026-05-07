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

## OOP 架構

```python
class HandLabeler:
    """用 vec_x×vec_y dot vec_z 區分 L/R"""
    def fit_predict(self, x_vec, y_vec, z_vec) -> np.ndarray:
        # 輸出：(N,) 'L'/'R'

class TransitionCounter:
    """建立 C[1024×1024] 轉移矩陣（分左右手加總）"""
    def fit(self, token_ids, hand_labels) -> np.ndarray:
        # C = C_left + C_right
        # 回傳：self.C (1024, 1024)

class SimilarityMatrix:
    """計算 S = cosine(M)"""
    def compute(self, M) -> np.ndarray:
        # 輸出：S (1024, 1024)

class ClusterExtractor:
    """Super Cluster Extraction"""
    def fit(self, S, tau=0.9) -> dict:
        # 輸出：super_cluster_map {int: [int]}

class SuperClusterPipeline:
    """串接全部，一鍵產出"""
    def fit(self, h5_path_or_X, x_vec, y_vec, z_vec, token_ids=None):
        # Step 1-7 全部在這裡執行
        pass
    def save(self, results_dir) -> None:
        # 產出檔案
        pass
```

---

## 輸出（Validation Criteria）

| 產出 | 路徑 | 成功特徵 |
|------|------|----------|
| Super Cluster mapping | `results/super_cluster_map.json` | JSON，含 N 個 Super_Cluster |
| S 矩陣 | `results/similarity_matrix.npy` | (1024, 1024)，無 NaN |
| M 矩陣 | `results/transition_matrix.npy` | (1024, 1024)，row sum ≈ 1 |
| W 矩陣 | `results/symmetrized_matrix.npy` | (1024, 1024)，對稱 |
| log | `results/pipeline.log` | "Step 1-7 completed" |

**異常指標：** 出現 "Error" / "NaN" → 失敗

---

_Last updated: 2026-05-07_
