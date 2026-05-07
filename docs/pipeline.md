# Super Cluster Pipeline — 詳細流程

## 概述

Super Cluster Pipeline 是 SLBHS Phase 2 的核心模組，透過**時序轉移相似度**找出物理等價的手型群組。

**核心假設**：如果兩個 Token 在時間序列中傾向於出現在相似的上下文（周圍的 Token），則它們可能代表相同的手型。

---

## 數學原理

### Step 1：手性標記（Hand Labeling）

根據 `cross(vec_x, vec_y) dot vec_z` 判斷左手/右手：

```
dot = sum(cross(vec_x, vec_y) * vec_z)
```

- `dot < 0` → **左手 (L)**
- `dot >= 0` → **右手 (R)**

物理意義：cross(x, y) 產生垂直於手掌平面的向量，與 z 點積為正表示右手，為負表示左手。

### Step 2：左右手軌道分離

將 Token 序列分離成兩個時間序列：
- `Left_Track`：hand_label == "L" 的 Token_ID 序列
- `Right_Track`：hand_label == "R" 的 Token_ID 序列

### Step 3：建立轉移矩陣 C[1024×1024]

$$C_{ij} = \sum_{\text{track} \in \{L,R\}} \sum_{\delta=1}^{\Delta} \mathbb{1}[Token_{t}=i \wedge Token_{t+\delta}=j]$$

分左右手各算一次後相加。

### Step 4：對稱化 + 列歸一化

$$W = \frac{C + C^T}{2}$$
$$M_{ij} = \frac{W_{ij}}{\sum_k W_{ik}}$$

M 是機率矩陣，每列 row sum = 1。

### Step 5：Cosine Similarity

$$S_{ij} = \frac{M_i \cdot M_j}{\|M_i\| \|M_j\|}$$

S 是相似度矩陣，S_ij 越大表示 Token i 和 Token j 的轉移行為越相似。

### Step 6：Super Cluster Extraction

- 設定閾值 τ
- 若 S_ij > τ，則在 Token i 和 Token j 之間建立一條邊
- 使用 Connected Components 演算法分群

---

## Pipeline 流程圖

```
输入 H5
  │
  ▼
[HandLabeler] ──→ hand_labels (L/R)
  │
  ▼
[BigClusterPipeline]
  │
  ├── 若 cosine_features=True:
  │   └── [KMeansClusterer.predict] ──→ labels
  │
  ├── 若 cosine_features=False:
  │   └── [MiniBatchKMeans] ──→ labels
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
输出：S.npy, C.npy, W.npy, super_cluster_map.json, pipeline_phase2.json
```

---

## 參數調優建議

### tau（相似度閾值）

| tau | 預期 clusters 數量 | 說明 |
|-----|-------------------|------|
| 0.95 | 多（小 clusters）| 高門檻，只有非常相似的 Token 才會被歸在一起 |
| 0.90 | 中等 | 預設值，平衡精確度和召回率 |
| 0.85 | 少（大 clusters）| 低門檻，更多 Token 被歸在一起 |

### delta_t（轉移間隔）

| delta_t | 適用情境 |
|---------|---------|
| 1 | 標準設定，只算連續兩幀 |
| 2-3 | 適用於取樣率較低或動作較慢的影片 |
| >3 | 通常不建議，可能引入過多雜訊 |

### min_transitions（最小轉移次數）

- 預設 0 表示不過濾
- 設定 >0 可以消除低頻噪聲
- 建議值：1-5

---

## 驗證方法

### 檢查 S 矩陣

```python
import numpy as np
S = np.load('results/similarity_matrix.npy')
assert not np.isnan(S).any(), "S contains NaN"
assert np.allclose(S, S.T), "S is not symmetric"
assert np.all(S <= 1.0), "S values > 1"
```

### 檢查 C 矩陣

```python
C = np.load('results/transition_matrix.npy')
assert C.shape == (1024, 1024), "C shape mismatch"
assert not np.isnan(C).any(), "C contains NaN"
```

### 檢查 Super Cluster Map

```python
import json
with open('results/super_cluster_map.json') as f:
    sc_map = json.load(f)
print(f"Tokens in clusters: {len(sc_map)}")
print(f"Unique super clusters: {len(set(sc_map.values()))}")
```

---

## 與 Hierarchical Clustering 的區別

| 項目 | BigClusterPipeline (Phase 2) | SuperClusterer (視覺化) |
|------|------------------------------|------------------------|
| 方法 | Connected Components | Agglomerative Hierarchical |
| 輸入 | 時序轉移相似度矩陣 S | K-Means centers |
| 目的 | 找出物理等價的手型 | 視覺化分群 |
| 用途 | 實際分類 | 圖表呈現 |

---

_Last updated: 2026-05-07_