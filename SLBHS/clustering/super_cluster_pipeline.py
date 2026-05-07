"""
Super Cluster Pipeline — Phase 1
超級分類 pipeline 的 OOP 架構

建立以下 class：
- HandLabeler：從 wrist_px x 座標推斷 L/R（已實作）
- TransitionCounter：建立轉移矩陣（框架，Phase 2 實作）
- SimilarityMatrix：計算相似度矩陣（框架，Phase 2 實作）
- ClusterExtractor：Super Cluster Extraction（框架，Phase 2 實作）
- SuperClusterPipeline：串接全部，一鍵產出（框架，Phase 2 串接）

作者：execution-agent（為 PM 打工）
日期：2026-05-07
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
    建立轉移矩陣（框架，Phase 2 實作）
    
    遍歷 Left_Track / Right_Track，
    僅選取同手、Δt ≤ delta_t 的相鄰 Token 對（T_n, T_{n+k}），
    C[T_n, T_{n+k}] += 1
    
    可過濾：min_transitions — 低於此轉移次數的 Token 會被忽略
    """

    def __init__(self, k: int = 1024, delta_t: int = 1, min_transitions: int = 0):
        self.k = k
        self.delta_t = delta_t
        self.min_transitions = min_transitions
        self.M: Optional[np.ndarray] = None  # (1024, 1024) 機率矩陣

    def fit(self, labels: np.ndarray, hand_labels: np.ndarray,
            delta_t: Optional[int] = None,
            min_transitions: Optional[int] = None) -> 'TransitionCounter':
        """
        建立左右手各自的 Token 轉移矩陣

        遍歷 Track（已按時間排序），選取同手、相鄰（Δt ≤ 2）的 Token 對：
        C[T_n, T_{n+1}] += 1

        最後將左右手矩陣相加，得到 C[1024×1024]

        Parameters
        ----------
        labels : (N,) Token_ID int，0-1023
        hand_labels : (N,) 'L'/'R' 字串陣列

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

        def _count_transitions(track: np.ndarray, C: np.ndarray) -> np.ndarray:
            """
            統計同一軌道上、相鄰（Δt = delta_t）的 Token 轉移
            
            過濾：低於 min_transitions 次的 Token 會被忽略
            """
            # 只取 Δt = delta_t（即跳過 delta_t-1 幀）
            for i in range(len(track) - delta_t):
                t_from = int(track[i])
                t_to = int(track[i + delta_t])
                C[t_from, t_to] += 1
            return C

        _count_transitions(left_track, C)
        _count_transitions(right_track, C)

        # 過濾：低於 min_transitions 的 Token 對清除為 0
        if min_transitions > 0:
            C[C < min_transitions] = 0

        self.C = C
        logger.info(f"[TransitionCounter] delta_t={delta_t}, min_transitions={min_transitions}, "
                    f"C.shape={C.shape}, nnz={int(np.sum(C > 0))}, max={C.max():.0f}")
        return self

    def get_matrix(self) -> np.ndarray:
        if self.C is None:
            raise RuntimeError("TransitionCounter must be fitted first")
        return self.C


class SimilarityMatrix:
    """
    計算相似度矩陣（框架，Phase 2 實作）
    
    M: (1024, 1024) 機率矩陣
    輸出：S (1024, 1024) cosine similarity
    """

    def __init__(self):
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

    （原名 ClusterExtractor，2026-05-07 更名以區分於 HierarchicalVisualizer）
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

        # 建立 super_cluster_map
        super_cluster_map = {}
        for comp_id in range(n_components):
            members = np.where(labels == comp_id)[0]
            for token_id in members:
                super_cluster_map[int(token_id)] = int(comp_id)

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
    串接全部，一鍵產出

    內部流程：
    1. HandLabeler.fit_predict(x_vec, y_vec, z_vec) → hand_labels
    2. KMeans → labels（cosine_features 模式切換）
    3. TransitionCounter.fit(labels, hand_labels) → C
    4. SimilarityMatrix.compute(C) → S
    5. ClusterExtractor.fit(S, tau) → super_cluster_map
    """

    def __init__(
        self,
        k: int = 512,                    # K-Means 群數
        tau: float = 0.9,                # BigClusterer threshold (0.0-1.0)
        delta_t: int = 1,                # 轉移間隔 (n → n+delta_t)
        cosine_features: bool = True,    # 是否用 cosine feature
        min_transitions: int = 0,        # 最小轉移次數（低於此則忽略）
        symmetrize: bool = True,         # 是否對稱化
        results_dir: Optional[str] = None,
    ):
        self.k = k
        self.tau = tau
        self.delta_t = delta_t
        self.cosine_features = cosine_features
        self.min_transitions = min_transitions
        self.symmetrize = symmetrize
        self.results_dir = results_dir
        self.hand_labeler = HandLabeler()
        self.transition_counter = TransitionCounter(k=k, delta_t=delta_t, min_transitions=min_transitions)
        self.similarity_matrix = SimilarityMatrix()
        self.big_clusterer = BigClusterer(tau=tau)  # 原 ClusterExtractor，更名以區分於 HierarchicalVisualizer
        self._fitted = False

    def fit(self, X: np.ndarray, x_vec: np.ndarray,
            y_vec: np.ndarray, z_vec: np.ndarray,
            labels: Optional[np.ndarray] = None,
            k: Optional[int] = None,
            tau: Optional[float] = None,
            cosine_features: Optional[bool] = None,
            min_transitions: Optional[int] = None,
            delta_t: Optional[int] = None,
            symmetrize: Optional[bool] = None,
            results_dir: Optional[str] = None) -> 'BigClusterPipeline':
        """
        串接 Phase 2 完整 pipeline

        流程：
        1. HandLabeler.fit_predict(x_vec, y_vec, z_vec) → hand_labels
        2. 若 labels 未提供：
           - cosine_features=True：KMeansClusterer.predict(X) [cosine mode]
           - cosine_features=False：MiniBatchKMeans on raw 63D
        3. TransitionCounter.fit(labels, hand_labels) → C
        4. SimilarityMatrix.compute(C) → S
        5. ClusterExtractor.fit(S, tau) → super_cluster_map

        Parameters
        ----------
        X : (N, 63) aligned_63d
        x_vec : (N, 3) x 軸向量
        y_vec : (N, 3) y 軸向量
        z_vec : (N, 3) z 軸向量
        labels : (N,) Token_ID；若 None 則依 cosine_features 模式產生
        k : int — K-Means 群數（cosine_features=False 且 labels=None 時使用）
        tau : float — similarity threshold
        cosine_features : bool — 是否使用 cosine feature 模式
           True：StandardScaler + cosine(15D) → concat(78D) → KMeansClusterer.predict
           False：直接用 aligned_63d (N,63) → MiniBatchKMeans
        results_dir : str — KMeansClusterer 模型路徑（cosine_features=True 時必要）
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from SLBHS.clustering.kmeans import KMeansClusterer
        from SLBHS.clustering.feature_transform import compute_cosine_features

        # 解析參數（優先使用傳入值，否則用實例變數）
        k = k if k is not None else self.k
        tau = tau if tau is not None else self.tau
        cosine_features = cosine_features if cosine_features is not None else self.cosine_features
        min_transitions = min_transitions if min_transitions is not None else self.min_transitions
        delta_t = delta_t if delta_t is not None else self.delta_t
        symmetrize = symmetrize if symmetrize is not None else self.symmetrize
        results_dir = results_dir if results_dir is not None else self.results_dir

        logger.info(f"[BigClusterPipeline] cosine_features={cosine_features}, k={k}, "
                    f"tau={tau}, delta_t={delta_t}, min_transitions={min_transitions}, "
                    f"symmetrize={symmetrize}")

        # Step 1：HandLabeler
        hand_labels = self.hand_labeler.fit_predict(x_vec, y_vec, z_vec)
        logger.info(f"[Pipeline] HandLabeler: L={int(np.sum(hand_labels=='L'))} "
                    f"R={int(np.sum(hand_labels=='R'))}")

        # Step 2：若未提供 labels，依模式產生
        if labels is None:
            if cosine_features:
                # Cosine feature 模式：KMeansClusterer.predict()
                rd = results_dir or self.results_dir
                if rd is None:
                    raise ValueError(
                        'cosine_features=True requires results_dir '
                        '(KMeansClusterer model path)'
                    )
                logger.info(f"[Pipeline] KMeansClusterer.predict (cosine mode) from {rd}")
                kc = KMeansClusterer(results_dir=rd)
                kc.load_model(rd)
                labels = kc.predict(X)
                k = kc.k
                logger.info(f"[Pipeline] KMeansClusterer done: "
                            f"unique_labels={len(np.unique(labels))}, k={k}")
            else:
                # Standard 模式：MiniBatchKMeans on raw 63D
                logger.info(f"[Pipeline] MiniBatchKMeans (k={k}) on raw 63D...")
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X).astype(np.float64)
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
                labels = kmeans.fit_predict(X_scaled)
                logger.info(f"[Pipeline] MiniBatchKMeans done, "
                            f"labels range: {int(labels.min())}-{int(labels.max())}")

        self.labels = labels
        self._k_used = k  # actual k used (may differ if loaded from model)
        self.cosine_features = cosine_features

        # Step 3：TransitionCounter（傳入 delta_t, min_transitions）
        self.transition_counter.fit(labels, hand_labels, delta_t=delta_t, min_transitions=min_transitions)
        C = self.transition_counter.get_matrix()
        logger.info(f"[Pipeline] TransitionCounter: C.shape={C.shape}")

        # Step 4：SimilarityMatrix（支援 symmetrize）
        S = self.similarity_matrix.compute(C, symmetrize=symmetrize)
        logger.info(f"[Pipeline] SimilarityMatrix: S.shape={S.shape}")

        # Step 5：BigClusterer（傳入 tau）
        self.big_clusterer.fit(S, tau=tau)
        super_cluster_map = self.big_clusterer.get_clusters()
        logger.info(f"[Pipeline] BigClusterer: N_clusters="
                    f"{self.big_clusterer.n_clusters}")

        self._fitted = True
        logger.info("[SuperClusterPipeline] 完整 pipeline 完成")
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
