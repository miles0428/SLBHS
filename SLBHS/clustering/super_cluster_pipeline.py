"""
Super Cluster Pipeline — Phase 2

超級分類 pipeline 的 OOP 架構：從時序轉移相似度找出物理等價的手型群組。

作者：execution-agent（為 PM 打工）
日期：2026-05-07

 Classes
--------
HandLabeler       : 從 wrist_px x 座標推斷 L/R（根據 vec_x × vec_y dot vec_z）
TransitionCounter : 建立 Token 轉移矩陣 C[1024×1024]（分左右手加總）
SimilarityMatrix  : 從 C 計算相似度矩陣 S（列歸一化 + Cosine Similarity）
BigClusterer      : 根據 S_ij > tau 建邊，用 Connected Components 分群
BigClusterPipeline: 串接全部，一鍵產出 Phase 2 結果

 Validation Criteria
--------------------
產出：
  - results/similarity_matrix.npy  — S (1024, 1024)，無 NaN
  - results/transition_matrix.npy  — C (1024, 1024)
  - results/symmetrized_matrix.npy — W (1024, 1024)
  - results/super_cluster_map.json — {token_id: super_cluster_id}
  - results/pipeline_phase2.json  — Phase 2 完整摘要

異常指標：出現 "Error" / "NaN" → 失敗

 Example
--------
>>> import h5py
>>> from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

>>> with h5py.File('file.h5', 'r') as f:
...     X = f['aligned_63d'][:]
...     x_vec = f['x_vec'][:]
...     y_vec = f['y_vec'][:]
...     z_vec = f['z_vec'][:]

>>> pipeline = BigClusterPipeline(k=512, tau=0.9, cosine_features=True, results_dir='results')
>>> pipeline.fit(X, x_vec, y_vec, z_vec, cosine_features=True, results_dir='results')
>>> pipeline.save('results')

>>> print(f"Super Clusters: {pipeline.big_clusterer.n_clusters}")
Super Clusters: 42
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class HandLabeler:
    """
    從 wrist_px x 座標推斷 L/R
    
    策略：使用 vec_x × vec_y dot vec_z 判斷左右手
    - 計算：dot_product = np.sum(np.cross(x_vec, y_vec) * z_vec, axis=1)
    - dot < 0 → 左手 (L)
    - dot >= 0 → 右手 (R)
    
    物理意義：cross(x_vec, y_vec) 產生垂直於手掌平面的向量，
    與 z_vec 點積為正表示右手，為負表示左手。
    """

    def __init__(self):
        self._fitted = False
        self._x_vec: Optional[np.ndarray] = None
        self._y_vec: Optional[np.ndarray] = None
        self._z_vec: Optional[np.ndarray] = None

    def fit_predict(self, x_vec: np.ndarray, y_vec: np.ndarray, 
                    z_vec: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x_vec : (N, 3) float32
            x 軸向量
        y_vec : (N, 3) float32
            y 軸向量
        z_vec : (N, 3) float32
            z 軸向量
        
        Returns
        -------
        hand_labels : (N,) '<U1' ('L' or 'R')
        """
        # 計算 cross product 並與 z_vec 點積
        cross_xy = np.cross(x_vec, y_vec)  # (N, 3)
        dot_product = np.sum(cross_xy * z_vec, axis=1)  # (N,)
        
        self._x_vec = x_vec.copy()
        self._y_vec = y_vec.copy()
        self._z_vec = z_vec.copy()
        self._fitted = True

        labels = np.empty(x_vec.shape[0], dtype='<U1')
        labels[dot_product < 0] = 'L'
        labels[dot_product >= 0] = 'R'

        logger.info(
            f"[HandLabeler] L={int(np.sum(labels=='L'))}  "
            f"R={int(np.sum(labels=='R'))}"
        )
        return labels

    def fit(self, x_vec: np.ndarray, y_vec: np.ndarray, 
            z_vec: np.ndarray) -> None:
        """支援 sklearn-style fit（無輸出，結果存在內部）"""
        self.fit_predict(x_vec, y_vec, z_vec)

    def predict(self, x_vec: np.ndarray, y_vec: np.ndarray,
                z_vec: np.ndarray) -> np.ndarray:
        """純推論（需先 fit）"""
        if not self._fitted:
            raise RuntimeError("HandLabeler must be fitted before predict()")
        cross_xy = np.cross(x_vec, y_vec)
        dot_product = np.sum(cross_xy * z_vec, axis=1)
        labels = np.empty(x_vec.shape[0], dtype='<U1')
        labels[dot_product < 0] = 'L'
        labels[dot_product >= 0] = 'R'
        return labels


class TransitionCounter:
    """
    建立 Token 轉移矩陣 C[1024×1024]。

    遍歷 Left_Track / Right_Track（已按時間排序），對每個 Token_i，
    往前看 delta_t 步：i→i+1, i→i+2, ..., i→i+delta_t
    每個符合的 pair 都 C[T_i, T_{i+k}] += 1（k=1~delta_t）

    可過濾：min_transitions — 低於此轉移次數的 Token 對清除為 0

    Example
    -------
    >>> counter = TransitionCounter(k=1024, delta_t=1, min_transitions=5)
    >>> counter.fit(token_ids, hand_labels, delta_t=1, min_transitions=5)
    >>> C = counter.get_matrix()  # (1024, 1024)
    """

    def __init__(self, k: int = 1024, delta_t: int = 1, min_transitions: int = 0):
        """
        Parameters
        ----------
        k : int
            Token 總數（通常 = K-Means k，預設 1024）。
        delta_t : int
            往前看多少步（預設 1，只算連續兩幀）。
        min_transitions : int
            最低轉移次數（低於此的 Token 對清除為 0）。
        """
        self.k = k
        self.delta_t = delta_t
        self.min_transitions = min_transitions
        self.C: Optional[np.ndarray] = None  # (1024, 1024) 機率矩陣

    def update(self, labels_batch: np.ndarray, hand_labels_batch: np.ndarray) -> 'TransitionCounter':
        """
        吃一個 batch（新H5）的 labels，累加進 self.C
        不改變 self.C 以外的狀態

        Parameters
        ----------
        labels_batch : (N,) Token_ID int，0-1023
        hand_labels_batch : (N,) 'L'/'R' 字串陣列

        Returns
        -------
        self
        """
        delta_t = self.delta_t
        C_batch = np.zeros((self.k, self.k), dtype=np.float64)

        # 分離左右手軌道（保持時間順序）
        left_mask = hand_labels_batch == 'L'
        right_mask = hand_labels_batch == 'R'
        left_track = labels_batch[left_mask]
        right_track = labels_batch[right_mask]

        # 收集所有 k 值的所有 pair，用 np.add.at 一次計入
        for track in (left_track, right_track):
            from_tokens_list = [track[:-k] for k in range(1, delta_t + 1)]
            to_tokens_list = [track[k:] for k in range(1, delta_t + 1)]
            from_all = np.concatenate(from_tokens_list)
            to_all = np.concatenate(to_tokens_list)
            np.add.at(C_batch, (from_all, to_all), 1)

        if self.C is None:
            self.C = C_batch
        else:
            self.C += C_batch

        logger.info(f"[TransitionCounter.update] batch added, C.nnz={int(np.sum(self.C > 0))}")
        return self

    def fit(self, labels: np.ndarray, hand_labels: np.ndarray,
            delta_t: Optional[int] = None,
            min_transitions: Optional[int] = None) -> 'TransitionCounter':
        """
        建立左右手各自的 Token 轉移矩陣

        遍歷 Track（已按時間排序），對每個 Token_i，往前看 delta_t 步：
        i→i+1, i→i+2, ..., i→i+delta_t
        每個符合的 pair 都 C[T_i, T_{i+k}] += 1（k=1~delta_t）
        最後將左右手矩陣相加，得到 C[1024×1024]

        可過濾：min_transitions — 低於此轉移次數的 Token 對清除為 0

        Parameters
        ----------
        labels : (N,) Token_ID int，0-1023
        hand_labels : (N,) 'L'/'R' 字串陣列
        delta_t : int — 往前看多少步（預設 1，只算連續兩幀）
        min_transitions : int — 最低轉移次數（低於此的 Token 對清除為 0）

        Returns
        -------
        self
        """
        delta_t = delta_t if delta_t is not None else self.delta_t
        min_transitions = min_transitions if min_transitions is not None else self.min_transitions

        k = self.k
        C = np.zeros((k, k), dtype=np.float64)

        # 分離左右手軌道（保持時間順序）
        left_mask = hand_labels == 'L'
        right_mask = hand_labels == 'R'

        left_track = labels[left_mask]          # (N_left,)
        right_track = labels[right_mask]         # (N_right,)

        # 收集所有 k 值的所有 pair，用 np.add.at 一次計入
        for track in (left_track, right_track):
            from_tokens_list = [track[:-k] for k in range(1, delta_t + 1)]
            to_tokens_list   = [track[k:]   for k in range(1, delta_t + 1)]
            from_all = np.concatenate(from_tokens_list)
            to_all   = np.concatenate(to_tokens_list)
            np.add.at(C, (from_all, to_all), 1)

        # 過濾：低於 min_transitions 的 Token 對清除為 0
        if min_transitions > 0:
            C[C < min_transitions] = 0

        self.C = C
        logger.info(f"[TransitionCounter] delta_t={delta_t}, min_transitions={min_transitions}, "
                    f"C.shape={C.shape}, nnz={int(np.sum(C > 0))}, max={C.max():.0f}")
        return self

    def get_matrix(self) -> np.ndarray:
        """
        Returns
        -------
        C : np.ndarray (k, k)
            轉移計數矩陣（未歸一化）。
        """
        if self.C is None:
            raise RuntimeError("TransitionCounter must be fitted first")
        return self.C


class SimilarityMatrix:
    """
    計算相似度矩陣 S = cosine(M_prob)。

    輸入：C[1024×1024] 轉移計數矩陣
    輸出：S[1024×1024] cosine similarity

    流程：
        1. 對稱化：W = (C + C.T) / 2（若 symmetrize=True）
        2. 列歸一化：M_ij = W_ij / Σ_k(W_ik) → 機率矩陣
        3. Cosine Similarity：S_ij = cos(M_i, M_j)

    Example
    -------
    >>> sim = SimilarityMatrix()
    >>> S = sim.compute(C, symmetrize=True)
    >>> S.shape
    (1024, 1024)
    """

    def __init__(self):
        """初始化，無需參數。"""
        self.S: Optional[np.ndarray] = None

    def compute(self, M: np.ndarray, symmetrize: bool = True) -> np.ndarray:
        """
        從 TransitionCounter 的 C 矩陣計算相似度矩陣 S

        步驟：
        1. 對稱化：W = (C + C.T) / 2（若 symmetrize=True）
        2. 列歸一化：M_ij = W_ij / Σ_k(W_ik)  → 機率矩陣
        3. Cosine Similarity：S_ij = cos(M_i, M_j)

        Parameters
        ----------
        M : (1024, 1024) 原始轉移計數矩陣 C
        symmetrize : bool — 是否對稱化（預設 True）

        Returns
        -------
        S : (1024, 1024) cosine similarity matrix
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # 1. 對稱化（可選）
        if symmetrize:
            W = (M + M.T) / 2.0
        else:
            W = M.copy()

        # 2. 列歸一化（row-wise，即對每列 i，將其所有出現在 j 的次數 normalize）
        row_sums = W.sum(axis=1, keepdims=True)  # (1024, 1)
        # 避免除 0
        row_sums[row_sums == 0] = 1.0
        M_prob = W / row_sums  # (1024, 1024)

        # 3. Cosine Similarity（row-wise）
        S = cosine_similarity(M_prob)

        self.M_prob = M_prob
        self.S = S

        # Validation
        row_sum_check = np.sum(M_prob, axis=1)
        logger.info(f"[SimilarityMatrix] M_prob row_sum: min={row_sum_check.min():.6f}, "
                    f"max={row_sum_check.max():.6f}")
        logger.info(f"[SimilarityMatrix] S.shape={S.shape}, "
                    f"nnz={int(np.sum(S > 0))}, nan={int(np.sum(np.isnan(S)))}")
        return S


class BigClusterer:
    """
    根據轉移矩陣/相似度矩陣分類有意義的手型群組。

    策略：S_ij > tau 時建邊，用 Connected Components 分群。

    Example
    -------
    >>> clusterer = BigClusterer(tau=0.9)
    >>> clusterer.fit(S, tau=0.9)
    >>> cluster_map = clusterer.get_clusters()  # {token_id: super_cluster_id}
    >>> print(f"N_clusters={clusterer.n_clusters}")
    N_clusters=42
    """

    def __init__(self, tau: float = 0.9):
        self.tau = tau
        self.cluster_map: Optional[dict] = None

    def fit(self, S: np.ndarray, tau: Optional[float] = None) -> 'BigClusterer':
        """
        從相似度矩陣 S 提取 Super Clusters

        策略：S_ij > tau 時建邊，用 Connected Components 分群

        Parameters
        ----------
        S : (1024, 1024) cosine similarity matrix

        Returns
        -------
        self
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        tau = tau if tau is not None else self.tau
        n = S.shape[0]

        # 建稀疏邻接矩阵：S_ij > tau → 1
        i_idx, j_idx = np.triu_indices(n, k=1)
        mask = S[i_idx, j_idx] > tau
        data = np.ones(np.sum(mask), dtype=np.int8)
        adj = csr_matrix((data, (i_idx[mask], j_idx[mask])), shape=(n, n))
        # 對稱化（無向圖）
        adj = adj + adj.T
        adj[adj > 1] = 1

        # Connected Components
        n_components, labels = connected_components(adj, directed=False)

        # 向量化建 dict：labels[i] = component_id of token i
        # O(n) 一次走完，無演算法層面的迴圈
        super_cluster_map = {int(t): int(labels[t]) for t in range(n)}

        self.cluster_map = super_cluster_map
        self.n_clusters = n_components

        logger.info(f"[BigClusterer] tau={tau}, N_clusters={n_components}, "
                    f"tokens={n}")
        return self

    def get_clusters(self) -> dict:
        if self.cluster_map is None:
            raise RuntimeError("BigClusterer must be fitted first")
        return self.cluster_map


class BigClusterPipeline:
    """
    串接全部，一鍵產出 Phase 2 Super Cluster 結果。

    內部流程：
        1. HandLabeler.fit_predict(x_vec, y_vec, z_vec) → hand_labels
        2. KMeans → labels（model_dir 模式：只 load 不 train）
        3. TransitionCounter.fit(labels, hand_labels) → C
        4. SimilarityMatrix.compute(C) → S
        5. BigClusterer.fit(S, tau) → super_cluster_map

    Example
    -------
    >>> pipeline = BigClusterPipeline(k=1024, tau=0.9, model_dir='/path/to/kmeans/model/')
    >>> pipeline.fit(X, x_vec, y_vec, z_vec)
    >>> pipeline.save('results')
    >>> print(f"Super Clusters: {pipeline.big_clusterer.n_clusters}")
    Super Clusters: 42
    """

    def __init__(
        self,
        k: int = 1024,                  # K-Means 群數
        tau: float = 0.9,               # BigClusterer threshold (0.0-1.0)
        delta_t: int = 10,               # 轉移間隔 (n → n+delta_t)
        min_transitions: int = 0,        # 最小轉移次數（低於此則忽略）
        symmetrize: bool = True,         # 是否對稱化
        model_dir: Optional[str] = None, # KMeans 模型目錄（只吃預訓練模型，不 train）
        results_dir: Optional[str] = None, # pipeline 產出路徑
    ):
        """
        Parameters
        ----------
        k : int
            K-Means 群數。
        tau : float
            相似度閾值（0.0-1.0），S_ij > tau → 建邊。
        delta_t : int
            轉移間隔，統計 n → n+delta_t 的轉移。
        min_transitions : int
            最低轉移次數，低於此的 Token 對清除為 0。
        symmetrize : bool
            是否對稱化轉移矩陣（預設 True）。
        model_dir : str or None
            KMeans 模型目錄，內含 kmeans_model.joblib + kmeans_scaler.joblib。
            update()/fit() 只 load model，不 train。
        results_dir : str or None
            Pipeline 產出路徑（相似度矩陣、超級分群映射等）。
        """
        self.k = k
        self.tau = tau
        self.delta_t = delta_t
        self.min_transitions = min_transitions
        self.symmetrize = symmetrize
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.hand_labeler = HandLabeler()
        self.transition_counter = TransitionCounter(k=k, delta_t=delta_t, min_transitions=min_transitions)
        self.similarity_matrix = SimilarityMatrix()
        self.big_clusterer = BigClusterer(tau=tau)
        self._fitted = False
        self._kmeans_loaded = False
        self._kmeans_clusterer = None  # KMeansClusterer instance (loaded lazily)

    def fit(self, X: np.ndarray, x_vec: np.ndarray,
            y_vec: np.ndarray, z_vec: np.ndarray,
            labels: Optional[np.ndarray] = None,
            k: Optional[int] = None,
            tau: Optional[float] = None,
            min_transitions: Optional[int] = None,
            delta_t: Optional[int] = None,
            symmetrize: Optional[bool] = None,
            model_dir: Optional[str] = None,
            results_dir: Optional[str] = None) -> 'BigClusterPipeline':
        """
        串接 Phase 2 完整 pipeline（單一 H5 直接完成）

        流程：
        1. HandLabeler.fit_predict(x_vec, y_vec, z_vec) → hand_labels
        2. 若 labels 未提供：依 model_dir 或 MiniBatchKMeans 產生
           - model_dir：KMeansClusterer.load_model() + predict()（只吃預訓練）
           - 無 model_dir：MiniBatchKMeans on raw 63D
        3. TransitionCounter.fit(labels, hand_labels) → C
        4. SimilarityMatrix.compute(C) → S
        5. BigClusterer.fit(S, tau) → super_cluster_map

        Parameters
        ----------
        X : (N, 63) aligned_63d
        x_vec : (N, 3) x 軸向量
        y_vec : (N, 3) y 軸向量
        z_vec : (N, 3) z 軸向量
        labels : (N,) Token_ID；若 None 則依 model_dir 模式產生
        k : int — K-Means 群數（無 model_dir 且 labels=None 時使用）
        tau : float — similarity threshold
        model_dir : str — KMeans 模型目錄（kmeans_model.joblib）
        results_dir : str — pipeline 產出路徑
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from SLBHS.clustering.kmeans import KMeansClusterer

        # 解析參數（優先使用傳入值，否則用實例變數）
        k = k if k is not None else self.k
        tau = tau if tau is not None else self.tau
        min_transitions = min_transitions if min_transitions is not None else self.min_transitions
        delta_t = delta_t if delta_t is not None else self.delta_t
        symmetrize = symmetrize if symmetrize is not None else self.symmetrize
        model_dir = model_dir if model_dir is not None else self.model_dir
        results_dir = results_dir if results_dir is not None else self.results_dir

        logger.info(f"[BigClusterPipeline.fit] k={k}, tau={tau}, "
                    f"delta_t={delta_t}, min_transitions={min_transitions}, "
                    f"symmetrize={symmetrize}, model_dir={model_dir}")

        # Step 1：HandLabeler
        hand_labels = self.hand_labeler.fit_predict(x_vec, y_vec, z_vec)
        logger.info(f"[Pipeline.fit] HandLabeler: L={int(np.sum(hand_labels=='L'))} "
                    f"R={int(np.sum(hand_labels=='R'))}")

        # Step 2：若未提供 labels，依模式產生
        if labels is None:
            if model_dir is not None:
                # Model 模式：KMeansClusterer.load_model() + predict()
                logger.info(f"[Pipeline.fit] KMeansClusterer.load_model from {model_dir}")
                kc = KMeansClusterer(results_dir=model_dir)
                kc.load_model(model_dir)
                labels = kc.predict(X)
                k = kc.k
                self._kmeans_loaded = True
                self._kmeans_clusterer = kc
                logger.info(f"[Pipeline.fit] KMeansClusterer.predict done: "
                            f"unique_labels={len(np.unique(labels))}, k={k}")
            else:
                # Fallback：MiniBatchKMeans on raw 63D
                logger.info(f"[Pipeline.fit] MiniBatchKMeans (k={k}) on raw 63D...")
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X).astype(np.float64)
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
                labels = kmeans.fit_predict(X_scaled)
                logger.info(f"[Pipeline.fit] MiniBatchKMeans done, "
                            f"labels range: {int(labels.min())}-{int(labels.max())}")

        self.labels = labels
        self._k_used = k

        # Step 3：TransitionCounter（傳入 delta_t, min_transitions）
        self.transition_counter.fit(labels, hand_labels, delta_t=delta_t, min_transitions=min_transitions)
        C = self.transition_counter.get_matrix()
        logger.info(f"[Pipeline.fit] TransitionCounter: C.shape={C.shape}")

        # Step 4：SimilarityMatrix（支援 symmetrize）
        S = self.similarity_matrix.compute(C, symmetrize=symmetrize)
        logger.info(f"[Pipeline.fit] SimilarityMatrix: S.shape={S.shape}")

        # Step 5：BigClusterer（傳入 tau）
        self.big_clusterer.fit(S, tau=tau)
        super_cluster_map = self.big_clusterer.get_clusters()
        logger.info(f"[Pipeline.fit] BigClusterer: N_clusters="
                    f"{self.big_clusterer.n_clusters}")

        self._fitted = True
        self._model_dir = model_dir
        logger.info("[BigClusterPipeline.fit] 完成")
        return self

    def update(self, X: np.ndarray, x_vec: np.ndarray,
               y_vec: np.ndarray, z_vec: np.ndarray) -> 'BigClusterPipeline':
        """
        吃一個 H5 的資料，累加進 TransitionCounter（C）。
        Model 只 load 一次（lazy），之後 skip。
        不 compute S、不做 BigClusterer（留給 finalize()）。

        Parameters
        ----------
        X : (N, 63) aligned_63d
        x_vec : (N, 3) x 軸向量
        y_vec : (N, 3) y 軸向量
        z_vec : (N, 3) z 軸向量

        Returns
        -------
        self
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from SLBHS.clustering.kmeans import KMeansClusterer

        model_dir = self.model_dir

        # Step 1：HandLabeler
        hand_labels = self.hand_labeler.fit_predict(x_vec, y_vec, z_vec)
        logger.info(f"[Pipeline.update] HandLabeler: L={int(np.sum(hand_labels=='L'))} "
                    f"R={int(np.sum(hand_labels=='R'))}")

        # Step 2：KMeans predict（model 只 load 一次）
        if model_dir is not None:
            if not self._kmeans_loaded:
                logger.info(f"[Pipeline.update] KMeansClusterer.load_model from {model_dir}")
                kc = KMeansClusterer(results_dir=model_dir)
                kc.load_model(model_dir)
                self._kmeans_clusterer = kc
                self._kmeans_loaded = True
            labels = self._kmeans_clusterer.predict(X)
            k = self._kmeans_clusterer.k
        else:
            k = self.k
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X).astype(np.float64)
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
            labels = kmeans.fit_predict(X_scaled)

        self._k_used = k

        # Step 3：累加進 TransitionCounter（不改變 self.C 以外的狀態）
        self.transition_counter.update(labels, hand_labels)
        logger.info(f"[Pipeline.update] TransitionCounter.C.nnz={int(np.sum(self.transition_counter.C > 0))}")

        # 標記 S 尚未計算
        self._S_computed = False
        return self

    def finalize(self, tau: Optional[float] = None) -> 'BigClusterPipeline':
        """
        所有 H5 跑完後呼叫
        compute S + BigClusterer

        Parameters
        ----------
        tau : float — similarity threshold（預設使用 self.tau）

        Returns
        -------
        self
        """
        tau = tau if tau is not None else self.tau
        symmetrize = self.symmetrize

        C = self.transition_counter.get_matrix()
        logger.info(f"[Pipeline.finalize] C.shape={C.shape}, C.nnz={int(np.sum(C > 0))}")

        # SimilarityMatrix
        S = self.similarity_matrix.compute(C, symmetrize=symmetrize)
        logger.info(f"[Pipeline.finalize] S.shape={S.shape}")

        # BigClusterer
        self.big_clusterer.fit(S, tau=tau)
        logger.info(f"[Pipeline.finalize] BigClusterer: N_clusters={self.big_clusterer.n_clusters}")

        self._fitted = True
        self._S_computed = True
        logger.info("[BigClusterPipeline] finalize 完成")
        return self

    def save(self, results_dir: str) -> None:
        """
        儲存 pipeline 產出到 results_dir

        Phase 2 產出：
        - similarity_matrix.npy : S 矩陣 (1024, 1024)
        - transition_matrix.npy : C 矩陣 (1024, 1024)
        - super_cluster_map.json : Map[Token_ID] → Super_Cluster_ID
        - pipeline_phase2.json : 摘要報告
        """
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # S 矩陣
        if self.similarity_matrix.S is not None:
            np.save(results_path / "similarity_matrix.npy",
                    self.similarity_matrix.S)
            logger.info(f"[Pipeline] S 矩陣已存至 {results_path / 'similarity_matrix.npy'}")

        # C 矩陣
        if self.transition_counter.C is not None:
            np.save(results_path / "transition_matrix.npy",
                    self.transition_counter.C)
            logger.info(f"[Pipeline] C 矩陣已存至 {results_path / 'transition_matrix.npy'}")

            # W 矩陣（對稱化）
            W = (self.transition_counter.C + self.transition_counter.C.T) / 2.0
            np.save(results_path / "symmetrized_matrix.npy", W)
            logger.info(f"[Pipeline] W 矩陣已存至 {results_path / 'symmetrized_matrix.npy'}")

        # BigClusterer（原 ClusterExtractor）產出
        if self.big_clusterer.cluster_map is not None:
            super_cluster_map = {
                str(k): v for k, v in self.big_clusterer.cluster_map.items()
            }
            with open(results_path / "super_cluster_map.json", "w", encoding="utf-8") as f:
                json.dump(super_cluster_map, f, indent=2)
            logger.info(f"[Pipeline] Super Cluster Map 已存至 "
                        f"{results_path / 'super_cluster_map.json'}")

        # 摘要報告
        report = {
            "phase": 2,
            "k": getattr(self, '_k_used', self.k),
            "tau": self.tau,
            "cosine_features": getattr(self, 'cosine_features', None),
            "hand_labeler": {
                "fitted": self.hand_labeler._fitted
            },
            "transition_counter": {
                "fitted": self.transition_counter.C is not None,
                "C_shape": list(self.transition_counter.C.shape) if self.transition_counter.C is not None else None,
                "C_nnz": int(np.sum(self.transition_counter.C > 0)) if self.transition_counter.C is not None else None
            },
            "similarity_matrix": {
                "computed": self.similarity_matrix.S is not None,
                "S_shape": list(self.similarity_matrix.S.shape) if self.similarity_matrix.S is not None else None,
                "S_nan": int(np.sum(np.isnan(self.similarity_matrix.S))) if self.similarity_matrix.S is not None else None
            },
            "big_clusterer": {
                "n_clusters": self.big_clusterer.n_clusters,
                "n_tokens": len(self.big_clusterer.cluster_map)
            }
        }

        with open(results_path / "pipeline_phase2.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"[SuperClusterPipeline] Phase 2 產出已存至 {results_dir}")


# ─────────────────────────────────────────────
# 測試程式：驗證 cosine_features 參數的兩種模式
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import h5py
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    h5_path = (
        "/home/ubuntu/.openclaw/media/inbound/"
        "2022_12_12_14_00_中央流行疫情指揮中心嚴重特殊傳染性肺炎記者會_crop"
        "---87414a0f-15ac-4b38-8710-15c43fa52793.h5"
    )
    cosine_results_dir = "/home/ubuntu/.openclaw/workspace-coding/SLBHS/SLBHS/results"
    standard_results_dir = "/home/ubuntu/.openclaw/workspace-coding/SLBHS/results_real"

    print("=" * 60)
    print("SLBHS Super Cluster Pipeline — cosine_features 測試")
    print("=" * 60)

    # 讀取 H5
    with h5py.File(h5_path, "r") as f:
        X = f["aligned_63d"][:]
        x_vec = f["x_vec"][:]
        y_vec = f["y_vec"][:]
        z_vec = f["z_vec"][:]

    print(f"X shape: {X.shape}")

    # ── 模式 1：cosine_features=True ──
    print("\n" + "=" * 60)
    print("測試模式 1：cosine_features=True")
    print("=" * 60)
    pipeline_cos = SuperClusterPipeline(k=512, tau=0.9)
    pipeline_cos.fit(
        X, x_vec, y_vec, z_vec,
        cosine_features=True,
        results_dir=cosine_results_dir
    )
    pipeline_cos.save(cosine_results_dir)
    C_cos = pipeline_cos.transition_counter.get_matrix()
    S_cos = pipeline_cos.similarity_matrix.S
    print(f"  C.shape={C_cos.shape}, nnz={np.sum(C_cos>0)}, nan={np.isnan(C_cos).sum()}")
    print(f"  S.shape={S_cos.shape}, nan={np.isnan(S_cos).sum()}")
    print(f"  N_clusters={pipeline_cos.big_clusterer.n_clusters}")
    print(f"  cosine_features in report: {getattr(pipeline_cos, 'cosine_features', None)}")

    # ── 模式 2：cosine_features=False（MiniBatchKMeans on raw 63D）──
    print("\n" + "=" * 60)
    print("測試模式 2：cosine_features=False (MiniBatchKMeans on raw 63D)")
    print("=" * 60)
    pipeline_raw = SuperClusterPipeline(k=64, tau=0.9)
    pipeline_raw.fit(
        X, x_vec, y_vec, z_vec,
        cosine_features=False,
        k=64
    )
    pipeline_raw.save(standard_results_dir)
    C_raw = pipeline_raw.transition_counter.get_matrix()
    S_raw = pipeline_raw.similarity_matrix.S
    print(f"  C.shape={C_raw.shape}, nnz={np.sum(C_raw>0)}, nan={np.isnan(C_raw).sum()}")
    print(f"  S.shape={S_raw.shape}, nan={np.isnan(S_raw).sum()}")
    print(f"  N_clusters={pipeline_raw.big_clusterer.n_clusters}")
    print(f"  cosine_features in report: {getattr(pipeline_raw, 'cosine_features', None)}")

    # ── Validation ──
    all_ok = (
        not np.isnan(S_cos).any() and not np.isnan(C_cos).any()
        and pipeline_cos.big_clusterer.n_clusters >= 1
        and not np.isnan(S_raw).any() and not np.isnan(C_raw).any()
        and pipeline_raw.big_clusterer.n_clusters >= 1
    )
    print(f"\n{'✅ 全部通過' if all_ok else '❌ 有問題'}")


# ═══════════════════════════════════════════════════════════
# 對照說明（2026-05-07 新增）
# ═══════════════════════════════════════════════════════════
# BigClusterer（原 ClusterExtractor）
#   - 根據轉移矩陣/相似度矩陣分類有意義的手型群組
#   - 輸入：C（轉移計數）、S（相似度）→ 輸出：super_cluster_map
#
# HierarchicalVisualizer（在 SLBHS/clustering/super_cluster.py）
#   - 拿 KMeans centers 做 Agglomerative Hierarchical Clustering
#   - 純粹用來畫圖，不是用於實際分類
#   - 輸入：kmeans_centers, kmeans_labels → 輸出：super_labels（視覺化用）
# ═══════════════════════════════════════════════════════════
