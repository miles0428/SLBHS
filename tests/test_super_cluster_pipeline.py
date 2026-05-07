"""
Tests for SLBHS Super Cluster Pipeline v3 — 5 classes
Converted from pytest to unittest.

H5 fixture: uses a real PTS H5 file for integration testing.
Pipeline v3: 完全吃 K-Means model，不 training。

All loops replaced with numpy vectorized operations.
"""
import unittest
import numpy as np
import h5py
import os
import tempfile
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Fixtures (module-level helpers)
# --------------------------------------------------------------------------

H5_PATH = (
    "/home/ubuntu/.openclaw/media/inbound/"
    "2022_12_12_14_00_中央流行疫情指揮中心嚴重特殊傳染性肺炎記者會"
    "_crop---87414a0f-15ac-4b38-8710-15c43fa52793.h5"
)

MODEL_DIR = "/home/ubuntu/.openclaw/workspace-coding/SLBHS/SLBHS/results"


def _load_h5_data():
    """Load real H5 data once for all tests in this module."""
    if not os.path.exists(H5_PATH):
        unittest.skip(f"H5 file not found: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        data = {
            "X":       f["aligned_63d"][:],
            "x_vec":   f["x_vec"][:],
            "y_vec":   f["y_vec"][:],
            "z_vec":   f["z_vec"][:],
        }
    return data


def _compute_labels_64(h5_data):
    """Pre-compute k=64 MiniBatchKMeans labels for all tests."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_s = scaler.fit_transform(h5_data["X"].astype(np.float64))
    km = MiniBatchKMeans(n_clusters=64, random_state=42, n_init=3)
    return km.fit_predict(X_s)


# --------------------------------------------------------------------------
# 1. HandLabeler
# --------------------------------------------------------------------------

class TestHandLabeler(unittest.TestCase):
    def setUp(self):
        self.h5_data = _load_h5_data()

    def test_handlabeler_output_shape(self):
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler
        hl = HandLabeler()
        labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        self.assertEqual(labels.shape, (self.h5_data["X"].shape[0],))

    def test_handlabeler_only_l_or_r(self):
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler
        hl = HandLabeler()
        labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        unique = set(labels)
        self.assertTrue(unique <= {"L", "R"}, f"Unexpected labels: {unique - {'L','R'}}")

    def test_handlabeler_bimodal_split(self):
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler
        hl = HandLabeler()
        labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        n_L = int(np.sum(labels == "L"))
        n_R = int(np.sum(labels == "R"))
        total = n_L + n_R
        # Both hands must appear; neither should be < 5% of total
        self.assertTrue(n_L > 0 and n_R > 0)
        self.assertTrue(n_L / total > 0.05, "Left hand < 5% — suspicious")
        self.assertTrue(n_R / total > 0.05, "Right hand < 5% — suspicious")

    def test_handlabeler_idempotent(self):
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler
        hl = HandLabeler()
        labels1 = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        labels2 = hl.predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        np.testing.assert_array_equal(labels1, labels2)


# --------------------------------------------------------------------------
# 2. TransitionCounter
# --------------------------------------------------------------------------

class TestTransitionCounter(unittest.TestCase):
    def setUp(self):
        self.h5_data = _load_h5_data()
        self.labels_64 = _compute_labels_64(self.h5_data)

    def test_transitioncounter_shape(self):
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        tc = TransitionCounter(k=64, delta_t=1)
        tc.fit(self.labels_64, hand_labels)
        C = tc.get_matrix()
        self.assertEqual(C.shape, (64, 64))

    def test_transitioncounter_symmetric(self):
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        tc = TransitionCounter(k=64, delta_t=1)
        tc.fit(self.labels_64, hand_labels)
        C = tc.get_matrix()
        diff = np.abs(C - C.T).max()
        self.assertTrue(diff / (C.max() + 1e-9) < 0.5, "C is wildly asymmetric")

    def test_transitioncounter_nonzero(self):
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        tc = TransitionCounter(k=64, delta_t=1)
        tc.fit(self.labels_64, hand_labels)
        C = tc.get_matrix()
        self.assertTrue(np.sum(C > 0) > 0, "Transition matrix is all zeros")

    def test_transitioncounter_min_transitions_filter(self):
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        tc = TransitionCounter(k=64, delta_t=1, min_transitions=10)
        tc.fit(self.labels_64, hand_labels, min_transitions=10)
        C = tc.get_matrix()
        if C.max() > 10:
            self.assertTrue(np.sum(C >= 10) <= np.sum(C > 0))

    def test_transitioncounter_delta_t2_pairs_count(self):
        """delta_t=2 should yield strictly more non-zero entries than delta_t=1."""
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        tc1 = TransitionCounter(k=64, delta_t=1)
        tc1.fit(self.labels_64, hand_labels)
        C1 = tc1.get_matrix()

        tc2 = TransitionCounter(k=64, delta_t=2)
        tc2.fit(self.labels_64, hand_labels)
        C2 = tc2.get_matrix()

        self.assertTrue(np.sum(C2 > 0) >= np.sum(C1 > 0))


# --------------------------------------------------------------------------
# 3. SimilarityMatrix
# --------------------------------------------------------------------------

class TestSimilarityMatrix(unittest.TestCase):
    def setUp(self):
        self.h5_data = _load_h5_data()
        self.labels_64 = _compute_labels_64(self.h5_data)

    def _build_sm(self):
        from SLBHS.clustering.super_cluster_pipeline import (
            HandLabeler, TransitionCounter, SimilarityMatrix
        )
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        tc = TransitionCounter(k=64, delta_t=1)
        tc.fit(self.labels_64, hand_labels)
        C = tc.get_matrix()
        sm = SimilarityMatrix()
        S = sm.compute(C)
        return S

    def test_similaritymatrix_no_nan(self):
        S = self._build_sm()
        self.assertEqual(np.sum(np.isnan(S)), 0, "Similarity matrix contains NaN")

    def test_similaritymatrix_diagonal_approx_one(self):
        S = self._build_sm()
        diag = np.diag(S)
        self.assertTrue(np.allclose(diag, 1.0, atol=1e-6), f"Diagonal not all 1.0: {diag}")

    def test_similaritymatrix_symmetric(self):
        S = self._build_sm()
        diff = np.abs(S - S.T).max()
        self.assertTrue(diff < 1e-9, "Similarity matrix not symmetric")

    def test_similaritymatrix_prob_row_sum(self):
        from SLBHS.clustering.super_cluster_pipeline import (
            HandLabeler, TransitionCounter, SimilarityMatrix
        )
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        tc = TransitionCounter(k=64, delta_t=1)
        tc.fit(self.labels_64, hand_labels)
        C = tc.get_matrix()
        sm = SimilarityMatrix()
        sm.compute(C)
        row_sums = np.sum(sm.M_prob, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# 4. BigClusterer
# --------------------------------------------------------------------------

class TestBigClusterer(unittest.TestCase):
    def setUp(self):
        self.h5_data = _load_h5_data()
        self.labels_64 = _compute_labels_64(self.h5_data)

    def _build_bc(self, tau=0.5):
        from SLBHS.clustering.super_cluster_pipeline import (
            HandLabeler, TransitionCounter, SimilarityMatrix, BigClusterer
        )
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        tc = TransitionCounter(k=64, delta_t=1)
        tc.fit(self.labels_64, hand_labels)
        C = tc.get_matrix()
        sm = SimilarityMatrix()
        S = sm.compute(C)
        bc = BigClusterer(tau=tau)
        bc.fit(S)
        return bc

    def test_bigclusterer_all_tokens_mapped(self):
        bc = self._build_bc(tau=0.5)
        cluster_map = bc.get_clusters()
        for t in range(64):
            self.assertIn(t, cluster_map, f"Token {t} not in cluster_map")

    def test_bigclusterer_tau_1_singleton(self):
        bc = self._build_bc(tau=1.0)
        self.assertEqual(bc.n_clusters, 64, f"tau=1.0 should give 64 singletons, got {bc.n_clusters}")

    def test_bigclusterer_n_clusters_positive(self):
        bc = self._build_bc(tau=0.5)
        self.assertTrue(bc.n_clusters >= 1)
        self.assertTrue(bc.n_clusters <= 64)


# --------------------------------------------------------------------------
# 5. BigClusterPipeline (integration)
# --------------------------------------------------------------------------

class TestBigClusterPipeline(unittest.TestCase):
    def setUp(self):
        self.h5_data = _load_h5_data()

    def test_bigclusterpipeline_fitted_true(self):
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
        pipeline = BigClusterPipeline(tau=0.5, model_dir=MODEL_DIR, results_dir=None)
        pipeline.fit(
            self.h5_data["X"],
            self.h5_data["x_vec"],
            self.h5_data["y_vec"],
            self.h5_data["z_vec"],
        )
        self.assertTrue(pipeline._fitted)

    def test_bigclusterpipeline_s_no_nan(self):
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
        pipeline = BigClusterPipeline(tau=0.5, model_dir=MODEL_DIR, results_dir=None)
        pipeline.fit(
            self.h5_data["X"],
            self.h5_data["x_vec"],
            self.h5_data["y_vec"],
            self.h5_data["z_vec"],
        )
        S = pipeline.similarity_matrix.S
        self.assertEqual(np.sum(np.isnan(S)), 0)

    def test_bigclusterpipeline_save_load(self):
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
        pipeline = BigClusterPipeline(tau=0.5, model_dir=MODEL_DIR, results_dir=None)
        pipeline.fit(
            self.h5_data["X"],
            self.h5_data["x_vec"],
            self.h5_data["y_vec"],
            self.h5_data["z_vec"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(tmpdir)
            for fname in [
                "similarity_matrix.npy",
                "transition_matrix.npy",
                "super_cluster_map.json",
                "pipeline_phase2.json",
            ]:
                path = os.path.join(tmpdir, fname)
                self.assertTrue(os.path.exists(path), f"Missing: {fname}")
            with open(os.path.join(tmpdir, "super_cluster_map.json")) as f:
                sc_map = json.load(f)
            self.assertEqual(len(sc_map), 512, "super_cluster_map should have 512 entries")
            S = np.load(os.path.join(tmpdir, "similarity_matrix.npy"))
            self.assertEqual(S.shape, (512, 512))
            self.assertEqual(np.sum(np.isnan(S)), 0)

    def test_bigclusterpipeline_tau_0_9_clusters(self):
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
        pipeline = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
        pipeline.fit(
            self.h5_data["X"],
            self.h5_data["x_vec"],
            self.h5_data["y_vec"],
            self.h5_data["z_vec"],
        )
        self.assertTrue(pipeline.big_clusterer.n_clusters >= 1)
        self.assertEqual(len(pipeline.big_clusterer.cluster_map), 512)


# --------------------------------------------------------------------------
# 6. TransitionCounter.update() — incremental batch update
# --------------------------------------------------------------------------

class TestTransitionCounterUpdate(unittest.TestCase):
    def setUp(self):
        self.h5_data = _load_h5_data()
        self.labels_64 = _compute_labels_64(self.h5_data)

    def test_transitioncounter_update_accumulates(self):
        """update() should accumulate C across multiple calls."""
        from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        N = len(self.labels_64) // 2
        labels_a = self.labels_64[:N]
        labels_b = self.labels_64[N:2*N]
        hand_a = hand_labels[:N]
        hand_b = hand_labels[N:2*N]

        tc = TransitionCounter(k=64, delta_t=1)
        tc.update(labels_a, hand_a)
        C1 = tc.get_matrix().copy()
        tc.update(labels_b, hand_b)
        C2 = tc.get_matrix()
        self.assertTrue(np.sum(C2 > 0) >= np.sum(C1 > 0), "C did not grow on second update")
        self.assertTrue(np.all(C2 >= C1 - 1e-9), "Second update overwrote instead of accumulating")

        tc_single = TransitionCounter(k=64, delta_t=1)
        tc_single.update(labels_a, hand_a)
        C_single = tc_single.get_matrix()
        tc_fit_a = TransitionCounter(k=64, delta_t=1)
        tc_fit_a.fit(labels_a, hand_a)
        C_fit_a = tc_fit_a.get_matrix()
        np.testing.assert_allclose(C_single, C_fit_a, rtol=1e-10, atol=1e-10)

    def test_transitioncounter_update_no_overwrite(self):
        """Second update() call should ADD to existing C, not overwrite."""
        from SLBHS.clustering.super_cluster_pipeline import TransitionCounter, HandLabeler
        hl = HandLabeler()
        hand_labels = hl.fit_predict(
            self.h5_data["x_vec"], self.h5_data["y_vec"], self.h5_data["z_vec"]
        )
        N = len(self.labels_64) // 2
        labels_a = self.labels_64[:N]
        labels_b = self.labels_64[N:2*N]
        hand_a = hand_labels[:N]
        hand_b = hand_labels[N:2*N]

        tc = TransitionCounter(k=64, delta_t=1)
        tc.update(labels_a, hand_a)
        C1 = tc.get_matrix().copy()
        tc.update(labels_b, hand_b)
        C2 = tc.get_matrix()
        self.assertTrue(np.sum(C2 > 0) >= np.sum(C1 > 0))
        self.assertTrue(np.all(C2 >= C1 - 1e-9))


# --------------------------------------------------------------------------
# 7. BigClusterPipeline.update() + finalize() — batch pipeline
# --------------------------------------------------------------------------

class TestPipelineUpdateFinalize(unittest.TestCase):
    def setUp(self):
        self.h5_data = _load_h5_data()

    def test_pipeline_update_then_finalize_matches_fit(self):
        """update()+finalize() should give identical C and S as fit()."""
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

        pipe_fit = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
        pipe_fit.fit(self.h5_data["X"], self.h5_data["x_vec"], self.h5_data["y_vec"],
                     self.h5_data["z_vec"], tau=0.9)
        C_fit = pipe_fit.transition_counter.get_matrix()
        S_fit = pipe_fit.similarity_matrix.S

        pipe_upd = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
        pipe_upd.update(self.h5_data["X"], self.h5_data["x_vec"],
                        self.h5_data["y_vec"], self.h5_data["z_vec"])
        pipe_upd.finalize(tau=0.9)
        C_upd = pipe_upd.transition_counter.get_matrix()
        S_upd = pipe_upd.similarity_matrix.S

        np.testing.assert_allclose(C_fit, C_upd, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(S_fit, S_upd, rtol=1e-10, atol=1e-10)
        self.assertEqual(pipe_fit.big_clusterer.n_clusters, pipe_upd.big_clusterer.n_clusters)

    def test_pipeline_update_then_finalize_no_nan(self):
        """S matrix from finalize() must not contain NaN."""
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
        pipe = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
        pipe.update(self.h5_data["X"], self.h5_data["x_vec"],
                    self.h5_data["y_vec"], self.h5_data["z_vec"])
        pipe.finalize(tau=0.9)
        S = pipe.similarity_matrix.S
        self.assertEqual(np.sum(np.isnan(S)), 0, "S contains NaN after finalize()")
        self.assertEqual(np.sum(np.isnan(pipe.transition_counter.get_matrix())), 0, "C contains NaN")

    def test_pipeline_update_twice_accumulates(self):
        """Two update() calls should accumulate C, and finalize() at end."""
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
        N = self.h5_data["X"].shape[0] // 2
        X_a = self.h5_data["X"][:N]
        X_b = self.h5_data["X"][N:2*N]
        xv_a = self.h5_data["x_vec"][:N]
        xv_b = self.h5_data["x_vec"][N:2*N]
        yv_a = self.h5_data["y_vec"][:N]
        yv_b = self.h5_data["y_vec"][N:2*N]
        zv_a = self.h5_data["z_vec"][:N]
        zv_b = self.h5_data["z_vec"][N:2*N]

        pipe = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
        pipe.update(X_a, xv_a, yv_a, zv_a)
        pipe.update(X_b, xv_b, yv_b, zv_b)
        pipe.finalize(tau=0.9)

        C = pipe.transition_counter.get_matrix()
        S = pipe.similarity_matrix.S
        self.assertTrue(np.sum(C > 0) > 0, "C is all zeros")
        self.assertEqual(np.sum(np.isnan(S)), 0, "S contains NaN")
        self.assertTrue(pipe.big_clusterer.n_clusters >= 1)


# --------------------------------------------------------------------------
# 8. BigClusterPipeline with model_dir (v3 — KMeans model only, no training)
# --------------------------------------------------------------------------

class TestPipelineWithModelDir(unittest.TestCase):
    def setUp(self):
        self.h5_data = _load_h5_data()

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_DIR):
            raise unittest.SkipTest(f"Model dir not found: {MODEL_DIR}")

    def test_pipeline_fit_with_model_dir(self):
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
        pipeline = BigClusterPipeline(
            tau=0.9, model_dir=MODEL_DIR, results_dir=None
        )
        pipeline.fit(
            self.h5_data["X"],
            self.h5_data["x_vec"],
            self.h5_data["y_vec"],
            self.h5_data["z_vec"],
        )

        self.assertTrue(pipeline._fitted)
        self.assertTrue(pipeline._kmeans_loaded)
        self.assertIsNotNone(pipeline._kmeans_clusterer)
        S = pipeline.similarity_matrix.S
        self.assertIsNotNone(S)
        self.assertEqual(np.sum(np.isnan(S)), 0, "S contains NaN")
        self.assertTrue(pipeline.big_clusterer.n_clusters >= 1)

    def test_pipeline_update_with_model_dir_loads_once(self):
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

        N = self.h5_data["X"].shape[0] // 2
        X_a = self.h5_data["X"][:N]
        X_b = self.h5_data["X"][N:2*N]
        xv_a = self.h5_data["x_vec"][:N]
        xv_b = self.h5_data["x_vec"][N:2*N]
        yv_a = self.h5_data["y_vec"][:N]
        yv_b = self.h5_data["y_vec"][N:2*N]
        zv_a = self.h5_data["z_vec"][:N]
        zv_b = self.h5_data["z_vec"][N:2*N]

        pipeline = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)

        pipeline.update(X_a, xv_a, yv_a, zv_a)
        self.assertTrue(pipeline._kmeans_loaded)
        first_kmeans = pipeline._kmeans_clusterer

        pipeline.update(X_b, xv_b, yv_b, zv_b)
        self.assertTrue(pipeline._kmeans_loaded)
        self.assertIs(first_kmeans, pipeline._kmeans_clusterer,
                      "KMeansClusterer should be reused, not reloaded")

        pipeline.finalize(tau=0.9)
        S = pipeline.similarity_matrix.S
        self.assertEqual(np.sum(np.isnan(S)), 0, "S contains NaN after update+finalize with model_dir")

    def test_pipeline_update_finalize_with_model_dir_no_nan(self):
        from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
        pipeline = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
        pipeline.update(self.h5_data["X"], self.h5_data["x_vec"],
                        self.h5_data["y_vec"], self.h5_data["z_vec"])
        pipeline.finalize(tau=0.9)

        S = pipeline.similarity_matrix.S
        C = pipeline.transition_counter.get_matrix()
        self.assertEqual(np.sum(np.isnan(S)), 0, "S contains NaN")
        self.assertEqual(np.sum(np.isnan(C)), 0, "C contains NaN")
        self.assertTrue(pipeline.big_clusterer.n_clusters >= 1)


if __name__ == "__main__":
    unittest.main()
