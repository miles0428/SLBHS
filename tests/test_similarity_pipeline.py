import pytest
import numpy as np
import os
import sys
import warnings
import logging

# Suppress verbose logs
warnings.filterwarnings('ignore')
logging.disable(logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SLBHS.clustering import ThetaClusterer
from SLBHS.similarity import SimilarityPipeline, TransitionCounter, CosineSimilarity, HandLabeler

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class TestTransitionCounter:
    """Tests for TransitionCounter"""
    
    def test_fit_update(self):
        """TransitionCounter fit/update should build transition matrix"""
        labels = np.array([0, 1, 0, 2, 1, 0])
        hand_labels = np.array(['L', 'R', 'L', 'R', 'L', 'R'])
        counter = TransitionCounter(k=3, delta_t=1)
        counter.fit(labels, hand_labels)
        C = counter.get_matrix()
        assert C.shape == (3, 3)
        assert C.dtype == np.float64
    
    def test_matrix_nonzero(self):
        """transition matrix should be built and non-zero"""
        labels = np.array([0, 1, 0, 2, 1, 0])
        hand_labels = np.array(['L', 'R', 'L', 'R', 'L', 'R'])
        counter = TransitionCounter(k=3, delta_t=1)
        counter.fit(labels, hand_labels)
        C = counter.get_matrix()
        assert C.shape == (3, 3)
        assert C.dtype == np.float64
        assert np.sum(C) > 0  # matrix has counts


class TestCosineSimilarity:
    """Tests for CosineSimilarity"""
    
    def test_compute(self):
        """compute() should return (k, k) similarity matrix"""
        # CosineSimilarity.compute(M, symmetrize) — M is a (k, k) transition count matrix
        k = 16
        # Random positive transition matrix
        M = np.random.rand(k, k)
        M = (M + M.T) / 2.0  # symmetric
        np.fill_diagonal(M, 0)  # no self-transitions
        
        sim = CosineSimilarity()
        S = sim.compute(M, symmetrize=True)
        assert S.shape == (k, k)
        # Diagonal should be 1.0
        assert np.allclose(np.diag(S), 1.0)


class TestSimilarityPipeline:
    """Tests for SimilarityPipeline"""
    
    def test_run(self, tmp_path):
        """run() should produce similarity_matrix.npy and transition_matrix.npy"""
        # Fit clusterer first
        clusterer = ThetaClusterer()
        clusterer.fit(h5_folder=DATA_DIR, top_k=16, verbose=False)
        
        pipeline = SimilarityPipeline(clusterer=clusterer, k=16, delta_t=5)
        pipeline.run(h5_folder=DATA_DIR)
        pipeline.save(str(tmp_path))
        
        assert os.path.exists(tmp_path / 'similarity_matrix.npy')
        assert os.path.exists(tmp_path / 'transition_matrix.npy')
        
        W = np.load(tmp_path / 'similarity_matrix.npy')
        assert W.shape == (16, 16)