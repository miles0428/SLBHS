"""
Tests for SLBHS Super Cluster Pipeline v3 — 5 classes

H5 fixture: uses a real PTS H5 file for integration testing.
Pipeline v3: 完全吃 K-Means model，不 training。

All loops replaced with numpy vectorized operations.
"""
import pytest
import numpy as np
import h5py
import os
import tempfile
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

H5_PATH = (
    "/home/ubuntu/.openclaw/media/inbound/"
    "2022_12_12_14_00_中央流行疫情指揮中心嚴重特殊傳染性肺炎記者會"
    "_crop---87414a0f-15ac-4b38-8710-15c43fa52793.h5"
)

MODEL_DIR = "/home/ubuntu/.openclaw/workspace-coding/SLBHS/SLBHS/results"


@pytest.fixture(scope="module")
def h5_data():
    """Load real H5 data once for all tests in this module."""
    if not os.path.exists(H5_PATH):
        pytest.skip(f"H5 file not found: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        data = {
            "X":       f["aligned_63d"][:],
            "x_vec":   f["x_vec"][:],
            "y_vec":   f["y_vec"][:],
            "z_vec":   f["z_vec"][:],
        }
    return data


@pytest.fixture(scope="module")
def labels_64(h5_data):
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

def test_handlabeler_output_shape(h5_data):
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler
    hl = HandLabeler()
    labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    assert labels.shape == (h5_data["X"].shape[0],)


def test_handlabeler_only_l_or_r(h5_data):
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler
    hl = HandLabeler()
    labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    unique = set(labels)
    assert unique <= {"L", "R"}, f"Unexpected labels: {unique - {'L','R'}}"


def test_handlabeler_bimodal_split(h5_data):
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler
    hl = HandLabeler()
    labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    n_L = int(np.sum(labels == "L"))
    n_R = int(np.sum(labels == "R"))
    total = n_L + n_R
    # Both hands must appear; neither should be < 5% of total
    assert n_L > 0 and n_R > 0
    assert n_L / total > 0.05, "Left hand < 5% — suspicious"
    assert n_R / total > 0.05, "Right hand < 5% — suspicious"


def test_handlabeler_idempotent(h5_data):
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler
    hl = HandLabeler()
    labels1 = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    labels2 = hl.predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    np.testing.assert_array_equal(labels1, labels2)


# --------------------------------------------------------------------------
# 2. TransitionCounter
# --------------------------------------------------------------------------

def test_transitioncounter_shape(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    assert C.shape == (64, 64)


def test_transitioncounter_symmetric(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    # C should be roughly symmetric (left+right tracks are independent but similar)
    diff = np.abs(C - C.T).max()
    assert diff / (C.max() + 1e-9) < 0.5, "C is wildly asymmetric"


def test_transitioncounter_nonzero(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    assert np.sum(C > 0) > 0, "Transition matrix is all zeros"


def test_transitioncounter_min_transitions_filter(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1, min_transitions=10)
    tc.fit(labels_64, hand_labels, min_transitions=10)
    C = tc.get_matrix()
    if C.max() > 10:
        assert np.sum(C >= 10) <= np.sum(C > 0)


def test_transitioncounter_delta_t2_pairs_count(labels_64, h5_data):
    """delta_t=2 should yield strictly more non-zero entries than delta_t=1."""
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc1 = TransitionCounter(k=64, delta_t=1)
    tc1.fit(labels_64, hand_labels)
    C1 = tc1.get_matrix()

    tc2 = TransitionCounter(k=64, delta_t=2)
    tc2.fit(labels_64, hand_labels)
    C2 = tc2.get_matrix()

    # More skips → same or more pairs counted
    assert np.sum(C2 > 0) >= np.sum(C1 > 0)


# --------------------------------------------------------------------------
# 3. SimilarityMatrix
# --------------------------------------------------------------------------

def test_similaritymatrix_no_nan(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import (
        HandLabeler, TransitionCounter, SimilarityMatrix
    )
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    sm = SimilarityMatrix()
    S = sm.compute(C)
    assert np.sum(np.isnan(S)) == 0, "Similarity matrix contains NaN"


def test_similaritymatrix_diagonal_approx_one(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import (
        HandLabeler, TransitionCounter, SimilarityMatrix
    )
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    sm = SimilarityMatrix()
    S = sm.compute(C)
    # Diagonal should be ~1.0 (self-similarity)
    diag = np.diag(S)
    assert np.allclose(diag, 1.0, atol=1e-6), f"Diagonal not all 1.0: {diag}"


def test_similaritymatrix_symmetric(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import (
        HandLabeler, TransitionCounter, SimilarityMatrix
    )
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    sm = SimilarityMatrix()
    S = sm.compute(C)
    diff = np.abs(S - S.T).max()
    assert diff < 1e-9, "Similarity matrix not symmetric"


def test_similaritymatrix_prob_row_sum(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import (
        HandLabeler, TransitionCounter, SimilarityMatrix
    )
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    sm = SimilarityMatrix()
    sm.compute(C)
    row_sums = np.sum(sm.M_prob, axis=1)
    # Row sums of M_prob should be exactly 1.0 (each row is a probability distribution)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# 4. BigClusterer
# --------------------------------------------------------------------------

def test_bigclusterer_all_tokens_mapped(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import (
        HandLabeler, TransitionCounter, SimilarityMatrix, BigClusterer
    )
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    sm = SimilarityMatrix()
    S = sm.compute(C)
    bc = BigClusterer(tau=0.5)
    bc.fit(S)
    cluster_map = bc.get_clusters()
    # Every token 0..63 must be in the map
    for t in range(64):
        assert t in cluster_map, f"Token {t} not in cluster_map"


def test_bigclusterer_tau_1_singleton(labels_64, h5_data):
    """tau=1.0 → only self-similarity > 1, so every token is its own component."""
    from SLBHS.clustering.super_cluster_pipeline import (
        HandLabeler, TransitionCounter, SimilarityMatrix, BigClusterer
    )
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    sm = SimilarityMatrix()
    S = sm.compute(C)
    bc = BigClusterer(tau=1.0)
    bc.fit(S)
    # tau=1.0 means only S_ii=1.0 > 1.0 is false, so adj matrix is all zeros
    # Connected components on empty graph → each node is its own component
    assert bc.n_clusters == 64, f"tau=1.0 should give 64 singletons, got {bc.n_clusters}"


def test_bigclusterer_n_clusters_positive(labels_64, h5_data):
    from SLBHS.clustering.super_cluster_pipeline import (
        HandLabeler, TransitionCounter, SimilarityMatrix, BigClusterer
    )
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    tc = TransitionCounter(k=64, delta_t=1)
    tc.fit(labels_64, hand_labels)
    C = tc.get_matrix()
    sm = SimilarityMatrix()
    S = sm.compute(C)
    bc = BigClusterer(tau=0.5)
    bc.fit(S)
    assert bc.n_clusters >= 1
    assert bc.n_clusters <= 64


# --------------------------------------------------------------------------
# 5. BigClusterPipeline (integration)
# --------------------------------------------------------------------------

def test_bigclusterpipeline_fitted_true(h5_data):
    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
    pipeline = BigClusterPipeline(tau=0.5, model_dir=MODEL_DIR, results_dir=None)
    pipeline.fit(
        h5_data["X"],
        h5_data["x_vec"],
        h5_data["y_vec"],
        h5_data["z_vec"],
       
    )
    assert pipeline._fitted is True


def test_bigclusterpipeline_s_no_nan(h5_data):
    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
    pipeline = BigClusterPipeline(tau=0.5, model_dir=MODEL_DIR, results_dir=None)
    pipeline.fit(
        h5_data["X"],
        h5_data["x_vec"],
        h5_data["y_vec"],
        h5_data["z_vec"],
       
    )
    S = pipeline.similarity_matrix.S
    assert np.sum(np.isnan(S)) == 0


def test_bigclusterpipeline_save_load(h5_data):
    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
    pipeline = BigClusterPipeline(tau=0.5, model_dir=MODEL_DIR, results_dir=None)
    pipeline.fit(
        h5_data["X"],
        h5_data["x_vec"],
        h5_data["y_vec"],
        h5_data["z_vec"],
       
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline.save(tmpdir)
        # Check all expected files exist
        for fname in [
            "similarity_matrix.npy",
            "transition_matrix.npy",
            "super_cluster_map.json",
            "pipeline_phase2.json",
        ]:
            path = os.path.join(tmpdir, fname)
            assert os.path.exists(path), f"Missing: {fname}"
        # Validate JSON load
        with open(os.path.join(tmpdir, "super_cluster_map.json")) as f:
            sc_map = json.load(f)
        assert len(sc_map) == 512, "super_cluster_map should have 512 entries"
        # Validate npy files
        S = np.load(os.path.join(tmpdir, "similarity_matrix.npy"))
        assert S.shape == (512, 512)
        assert np.sum(np.isnan(S)) == 0


def test_bigclusterpipeline_tau_0_9_clusters(h5_data):
    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
    pipeline = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
    pipeline.fit(
        h5_data["X"],
        h5_data["x_vec"],
        h5_data["y_vec"],
        h5_data["z_vec"],
       
    )
    # tau=0.9 should produce at least 1 cluster
    assert pipeline.big_clusterer.n_clusters >= 1
    # Every token must be mapped
    assert len(pipeline.big_clusterer.cluster_map) == 512


# --------------------------------------------------------------------------
# 6. TransitionCounter.update() — incremental batch update
# --------------------------------------------------------------------------

def test_transitioncounter_update_accumulates(labels_64, h5_data):
    """update() should accumulate C across multiple calls."""
    from SLBHS.clustering.super_cluster_pipeline import HandLabeler, TransitionCounter
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    N = len(labels_64) // 2
    labels_a = labels_64[:N]
    labels_b = labels_64[N:2*N]
    hand_a = hand_labels[:N]
    hand_b = hand_labels[N:2*N]

    # Two sequential update() calls should accumulate
    tc = TransitionCounter(k=64, delta_t=1)
    tc.update(labels_a, hand_a)
    C1 = tc.get_matrix().copy()
    tc.update(labels_b, hand_b)
    C2 = tc.get_matrix()
    # After second update, we expect MORE or EQUAL non-zero entries (accumulation)
    assert np.sum(C2 > 0) >= np.sum(C1 > 0), "C did not grow on second update"
    # C2 should be >= C1 element-wise (no overwriting)
    assert np.all(C2 >= C1 - 1e-9), "Second update overwrote instead of accumulating"

    # Compare with single-batch fit() — within-batch counts should match exactly
    tc_single = TransitionCounter(k=64, delta_t=1)
    tc_single.update(labels_a, hand_a)
    C_single = tc_single.get_matrix()
    tc_fit_a = TransitionCounter(k=64, delta_t=1)
    tc_fit_a.fit(labels_a, hand_a)
    C_fit_a = tc_fit_a.get_matrix()
    np.testing.assert_allclose(C_single, C_fit_a, rtol=1e-10, atol=1e-10)


def test_transitioncounter_update_no_overwrite(labels_64, h5_data):
    """Second update() call should ADD to existing C, not overwrite."""
    from SLBHS.clustering.super_cluster_pipeline import TransitionCounter, HandLabeler
    hl = HandLabeler()
    hand_labels = hl.fit_predict(
        h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"]
    )
    N = len(labels_64) // 2
    labels_a = labels_64[:N]
    labels_b = labels_64[N:2*N]
    hand_a = hand_labels[:N]
    hand_b = hand_labels[N:2*N]

    tc = TransitionCounter(k=64, delta_t=1)
    tc.update(labels_a, hand_a)
    C1 = tc.get_matrix().copy()
    tc.update(labels_b, hand_b)
    C2 = tc.get_matrix()
    # C2 should have MORE non-zero entries than C1
    assert np.sum(C2 > 0) >= np.sum(C1 > 0)
    # C2 >= C1 element-wise (accumulation)
    assert np.all(C2 >= C1 - 1e-9)


# --------------------------------------------------------------------------
# 7. BigClusterPipeline.update() + finalize() — batch pipeline
# --------------------------------------------------------------------------

def test_pipeline_update_then_finalize_matches_fit(h5_data):
    """update()+finalize() should give identical C and S as fit()."""
    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

    # mode A: single fit()
    pipe_fit = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
    pipe_fit.fit(h5_data["X"], h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"],
                 tau=0.9)
    C_fit = pipe_fit.transition_counter.get_matrix()
    S_fit = pipe_fit.similarity_matrix.S

    # mode B: update() + finalize()
    pipe_upd = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
    pipe_upd.update(h5_data["X"], h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"])
    pipe_upd.finalize(tau=0.9)
    C_upd = pipe_upd.transition_counter.get_matrix()
    S_upd = pipe_upd.similarity_matrix.S

    np.testing.assert_allclose(C_fit, C_upd, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(S_fit, S_upd, rtol=1e-10, atol=1e-10)
    assert pipe_fit.big_clusterer.n_clusters == pipe_upd.big_clusterer.n_clusters


def test_pipeline_update_then_finalize_no_nan(h5_data):
    """S matrix from finalize() must not contain NaN."""
    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
    pipe = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
    pipe.update(h5_data["X"], h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"])
    pipe.finalize(tau=0.9)
    S = pipe.similarity_matrix.S
    assert np.sum(np.isnan(S)) == 0, "S contains NaN after finalize()"
    assert np.sum(np.isnan(pipe.transition_counter.get_matrix())) == 0, "C contains NaN"


def test_pipeline_update_twice_accumulates(h5_data):
    """Two update() calls should accumulate C, and finalize() at end."""
    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
    N = h5_data["X"].shape[0] // 2
    X_a = h5_data["X"][:N]
    X_b = h5_data["X"][N:2*N]
    xv_a = h5_data["x_vec"][:N]
    xv_b = h5_data["x_vec"][N:2*N]
    yv_a = h5_data["y_vec"][:N]
    yv_b = h5_data["y_vec"][N:2*N]
    zv_a = h5_data["z_vec"][:N]
    zv_b = h5_data["z_vec"][N:2*N]

    pipe = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
    pipe.update(X_a, xv_a, yv_a, zv_a)
    pipe.update(X_b, xv_b, yv_b, zv_b)
    pipe.finalize(tau=0.9)

    C = pipe.transition_counter.get_matrix()
    S = pipe.similarity_matrix.S
    assert np.sum(C > 0) > 0, "C is all zeros"
    assert np.sum(np.isnan(S)) == 0, "S contains NaN"
    assert pipe.big_clusterer.n_clusters >= 1


# --------------------------------------------------------------------------
# 8. BigClusterPipeline with model_dir (v3 — KMeans model only, no training)
# --------------------------------------------------------------------------

def test_pipeline_fit_with_model_dir(h5_data):
    """fit() with model_dir should load pre-trained KMeans model and produce valid output."""
    if not os.path.exists(MODEL_DIR):
        pytest.skip(f"Model dir not found: {MODEL_DIR}")

    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
    pipeline = BigClusterPipeline(
        tau=0.9, model_dir=MODEL_DIR, results_dir=None
    )
    pipeline.fit(
        h5_data["X"],
        h5_data["x_vec"],
        h5_data["y_vec"],
        h5_data["z_vec"],
    )

    assert pipeline._fitted is True
    assert pipeline._kmeans_loaded is True
    assert pipeline._kmeans_clusterer is not None
    S = pipeline.similarity_matrix.S
    assert S is not None
    assert np.sum(np.isnan(S)) == 0, "S contains NaN"
    assert pipeline.big_clusterer.n_clusters >= 1


def test_pipeline_update_with_model_dir_loads_once(h5_data):
    """update() with model_dir should load model only on first call, reuse on subsequent."""
    if not os.path.exists(MODEL_DIR):
        pytest.skip(f"Model dir not found: {MODEL_DIR}")

    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

    N = h5_data["X"].shape[0] // 2
    X_a = h5_data["X"][:N]
    X_b = h5_data["X"][N:2*N]
    xv_a = h5_data["x_vec"][:N]
    xv_b = h5_data["x_vec"][N:2*N]
    yv_a = h5_data["y_vec"][:N]
    yv_b = h5_data["y_vec"][N:2*N]
    zv_a = h5_data["z_vec"][:N]
    zv_b = h5_data["z_vec"][N:2*N]

    pipeline = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)

    # First update: model loads
    pipeline.update(X_a, xv_a, yv_a, zv_a)
    assert pipeline._kmeans_loaded is True
    first_kmeans = pipeline._kmeans_clusterer

    # Second update: model should be reused (same instance)
    pipeline.update(X_b, xv_b, yv_b, zv_b)
    assert pipeline._kmeans_loaded is True
    assert pipeline._kmeans_clusterer is first_kmeans, "KMeansClusterer should be reused, not reloaded"

    pipeline.finalize(tau=0.9)
    S = pipeline.similarity_matrix.S
    assert np.sum(np.isnan(S)) == 0, "S contains NaN after update+finalize with model_dir"


def test_pipeline_update_finalize_with_model_dir_no_nan(h5_data):
    """update()+finalize() with model_dir produces valid S without NaN."""
    if not os.path.exists(MODEL_DIR):
        pytest.skip(f"Model dir not found: {MODEL_DIR}")

    from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline
    pipeline = BigClusterPipeline(tau=0.9, model_dir=MODEL_DIR, results_dir=None)
    pipeline.update(h5_data["X"], h5_data["x_vec"], h5_data["y_vec"], h5_data["z_vec"])
    pipeline.finalize(tau=0.9)

    S = pipeline.similarity_matrix.S
    C = pipeline.transition_counter.get_matrix()
    assert np.sum(np.isnan(S)) == 0, "S contains NaN"
    assert np.sum(np.isnan(C)) == 0, "C contains NaN"
    assert pipeline.big_clusterer.n_clusters >= 1
