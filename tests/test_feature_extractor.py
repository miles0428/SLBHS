import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SLBHS.data.feature_extractor import HandFeatureExtractor


class TestHandFeatureExtractor:
    """Tests for HandFeatureExtractor (multi-hand API)"""

    @pytest.fixture
    def extractor(self):
        return HandFeatureExtractor(
            model_path=os.path.join(os.path.dirname(__file__), 'models', 'hand_landmarker.task'),
            rgb_input=False
        )

    @pytest.fixture
    def test_image(self):
        import cv2
        img_path = os.path.join(os.path.dirname(__file__), 'data', 'YC_test_image.jpg')
        image = cv2.imread(img_path)
        assert image is not None, "Failed to load test image"
        return image

    def test_extract_returns_list(self, extractor, test_image):
        """extract() should return a list"""
        result = extractor.extract(test_image)
        assert isinstance(result, list), "extract() should return a list"
        assert len(result) >= 1, "At least one hand should be detected in test image"

    def test_extract_returns_empty_list_when_no_hand(self, extractor):
        """extract() should return [] when no hand detected"""
        import cv2
        # Create a blank/-noise image unlikely to have a hand
        blank = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = extractor.extract(blank)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_extract_two_hands_detected(self, extractor, test_image):
        """Test image should return exactly 2 hands, one Left and one Right"""
        result = extractor.extract(test_image)
        assert len(result) == 2, f"Expected 2 hands detected, got {len(result)}"

        handednesses = [r['handedness'] for r in result]
        assert set(handednesses) == {'Left', 'Right'}, \
            f"Expected one Left and one Right, got {handednesses}"

    def test_extract_hand_features_per_hand(self, extractor, test_image):
        """Each hand should have aligned_63d of shape (63,) and unit vectors"""
        result = extractor.extract(test_image)
        assert len(result) == 2

        for r in result:
            assert r['aligned_63d'].shape == (63,), \
                f"aligned_63d should be (63,), got {r['aligned_63d'].shape}"
            assert r['aligned_63d'].dtype == np.float32
            for key in ['x_vec', 'y_vec', 'z_vec']:
                assert r[key].shape == (3,), \
                    f"{key} should be (3,), got {r[key].shape}"
                assert np.isclose(np.linalg.norm(r[key]), 1.0, atol=1e-6), \
                    f"{key} should be unit vector"
            assert r['wrist_px'].shape == (3,)

    def test_extract_normal_vectors_per_hand(self, extractor, test_image):
        """Each hand should have x_vec, y_vec, z_vec as unit vectors"""
        result = extractor.extract(test_image)
        for r in result:
            for key in ['x_vec', 'y_vec', 'z_vec']:
                vec = r[key]
                assert vec.shape == (3,), f"{key} should be (3,), got {vec.shape}"
                assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6), \
                    f"{key} should be unit vector"

    def test_extract_wrist_px_per_hand(self, extractor, test_image):
        """Each hand should have wrist_px of shape (3,)"""
        result = extractor.extract(test_image)
        for r in result:
            assert 'wrist_px' in r
            wrist = r['wrist_px']
            assert wrist.shape == (3,), f"wrist_px should be (3,), got {wrist.shape}"

    def test_extract_handledness_per_hand(self, extractor, test_image):
        """Each hand should have handedness string"""
        result = extractor.extract(test_image)
        for r in result:
            assert 'handedness' in r
            assert r['handedness'] in ['Left', 'Right']

    def test_extract_batch(self, extractor, test_image):
        """extract_batch() should return list of list[dict]"""
        results = extractor.extract_batch([test_image, test_image])
        assert len(results) == 2
        # Each element is a list of hand dicts
        assert all(isinstance(r, list) for r in results)
        # Each should have 2 hands
        assert all(len(r) == 2 for r in results)
        # First hand from first image
        assert results[0][0]['aligned_63d'].shape == (63,)
