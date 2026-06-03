import pytest
import numpy as np
import os
import sys
import warnings
import logging

# Suppress verbose theta clusterer logs
warnings.filterwarnings('ignore')
logging.disable(logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SLBHS.clustering import ThetaClusterer

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class TestThetaClusterer:
    """Tests for ThetaClusterer"""
    
    def test_fit(self):
        """fit() should train on H5 data and produce a model"""
        clusterer = ThetaClusterer()
        clusterer.fit(h5_folder=DATA_DIR, top_k=32, verbose=False)
        assert clusterer.n_classes_ is not None
        assert clusterer.n_classes_ > 0
        assert clusterer.label_counts_ is not None
    
    def test_save_load(self, tmp_path):
        """save() and load() should preserve model state"""
        clusterer = ThetaClusterer()
        clusterer.fit(h5_folder=DATA_DIR, top_k=16, verbose=False)
        model_dir = str(tmp_path / 'model')
        clusterer.save(model_dir)
        
        loaded = ThetaClusterer()
        loaded.load(model_dir)  # instance method, not classmethod
        assert loaded.n_classes_ == clusterer.n_classes_
        assert loaded.label_counts_ == clusterer.label_counts_
    
    def test_generate_report(self, tmp_path):
        """generate_report() should write a file"""
        clusterer = ThetaClusterer()
        clusterer.fit(h5_folder=DATA_DIR, top_k=16, verbose=False)
        report_path = str(tmp_path / 'report.txt')
        clusterer.generate_report(DATA_DIR, report_path)
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 0


class TestDataLoader:
    """Tests for DataLoader"""
    
    def test_load(self):
        """DataLoader should load aligned_63d from H5"""
        from SLBHS.data.loader import DataLoader
        loader = DataLoader(data_dir=DATA_DIR)
        result = loader.load()
        # load() returns a tuple (X,)
        assert isinstance(result, tuple)
        X = result[0]
        assert X.shape[1] == 63  # 21 keypoints * 3
        assert X.shape[0] > 0