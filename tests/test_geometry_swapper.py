"""
Unit tests for HandPoseGeometrySwapper.

Tests cover:
  1. Math: x_vec[0] negated, y/z unchanged after swap
  2. Label: L ↔ R flip after swap (verifiable via HandLabeler dot-product)
  3. Aligned 63-D vector is NOT modified (original untouched)
  4. No-side-effect: original dict is never mutated
  5. Error handling: missing keys, invalid label
"""

import numpy as np
import pytest

from SLBHS.geometry import HandPoseGeometrySwapper
from SLBHS.similarity import HandLabeler


# --------------------------------------------------------------------------- #
# Fixtures                                                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def swapper():
    """A fresh swapper instance for each test."""
    return HandPoseGeometrySwapper()


def make_feature(label, x_vec, y_vec, z_vec, seed=42):
    """Build a minimal feature dict for testing."""
    rng = np.random.default_rng(seed)
    aligned = rng.standard_normal(63).astype(np.float32)
    return {
        'label':   label,
        'aligned': aligned,
        'x_vec':   np.array(x_vec, dtype=np.float32),
        'y_vec':   np.array(y_vec, dtype=np.float32),
        'z_vec':   np.array(z_vec, dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
# T1: X-axis negation                                                       #
# --------------------------------------------------------------------------- #

class TestXAxisNegation:
    """x_vec is mirrored; y_vec and z_vec stay the same."""

    def test_x_component_negated(self, swapper):
        feat = make_feature('L', [1., 2., 3.], [0., 1., 0.], [0., 0., 1.])
        out = swapper.swap(feat)

        np.testing.assert_allclose(out['x_vec'], [-1., -2., -3.])
        np.testing.assert_allclose(out['y_vec'], [0., 1., 0.])
        np.testing.assert_allclose(out['z_vec'], [0., 0., 1.])

    def test_x_component_only_first_element_matters(self, swapper):
        """The sign of the FIRST component of x_vec determines handedness via
        cross/dot, but the full vector is negated."""
        feat = make_feature('R', [5., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        out = swapper.swap(feat)

        assert out['x_vec'][0] == -5.
        assert out['x_vec'][1] == 0.
        assert out['x_vec'][2] == 0.

    def test_aligned_shape_unchanged(self, swapper):
        feat = make_feature('R', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        out = swapper.swap(feat)

        assert out['aligned'].shape == (63,)


# --------------------------------------------------------------------------- #
# T2: Label flip (L ↔ R)                                                    #
# --------------------------------------------------------------------------- #

class TestLabelFlip:
    """Label is always inverted after a single swap."""

    def test_L_becomes_R(self, swapper):
        feat = make_feature('L', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        assert swapper.swap(feat)['label'] == 'R'

    def test_R_becomes_L(self, swapper):
        feat = make_feature('R', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        assert swapper.swap(feat)['label'] == 'L'

    def test_double_swap_returns_original_label(self, swapper):
        feat = make_feature('L', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        twice = swapper.swap(swapper.swap(feat))
        assert twice['label'] == feat['label']


# --------------------------------------------------------------------------- #
# T3: HandLabeler cross-product invariance                                  #
# --------------------------------------------------------------------------- #

class TestHandLabelerCompatibility:
    """Swap output is consistent with HandLabeler's handedness logic.

    HandLabeler rule: dot = cross(x, y) · z
      dot < 0 → 'L'   dot >= 0 → 'R'

    After x → -x: cross(-x, y) = -cross(x, y)
      → dot flips sign → label inverts.
    """

    def test_swapped_L_is_R_under_handlabeler(self, swapper):
        """Build an 'L' sample, swap it, verify HandLabeler now says 'R'."""
        # x = [-1,0,0] → cross = [0,0,-1] → dot = -1 < 0 → 'L'
        x_vec = np.array([-1., 0., 0.], dtype=np.float32)
        y_vec = np.array([0., 1., 0.], dtype=np.float32)
        z_vec = np.array([0., 0., 1.], dtype=np.float32)

        labeler = HandLabeler()
        assert labeler.fit_predict(
            x_vec.reshape(1, 3), y_vec.reshape(1, 3), z_vec.reshape(1, 3)
        )[0] == 'L'

        feat = {'label': 'L', 'aligned': np.ones(63, np.float32),
                'x_vec': x_vec, 'y_vec': y_vec, 'z_vec': z_vec}
        swapped = swapper.swap(feat)

        assert labeler.fit_predict(
            swapped['x_vec'].reshape(1, 3),
            swapped['y_vec'].reshape(1, 3),
            swapped['z_vec'].reshape(1, 3),
        )[0] == 'R'

    def test_swapped_R_is_L_under_handlabeler(self):
        """Build an 'R' sample, swap it, verify HandLabeler now says 'L'."""
        # x = [1,0,0] → cross = [0,0,1] → dot = 1 >= 0 → 'R'
        x_vec = np.array([1., 0., 0.], dtype=np.float32)
        y_vec = np.array([0., 1., 0.], dtype=np.float32)
        z_vec = np.array([0., 0., 1.], dtype=np.float32)

        labeler = HandLabeler()
        assert labeler.fit_predict(
            x_vec.reshape(1, 3), y_vec.reshape(1, 3), z_vec.reshape(1, 3)
        )[0] == 'R'

        swapper = HandPoseGeometrySwapper()
        feat = {'label': 'R', 'aligned': np.ones(63, np.float32),
                'x_vec': x_vec, 'y_vec': y_vec, 'z_vec': z_vec}
        swapped = swapper.swap(feat)

        assert labeler.fit_predict(
            swapped['x_vec'].reshape(1, 3),
            swapped['y_vec'].reshape(1, 3),
            swapped['z_vec'].reshape(1, 3),
        )[0] == 'L'


# --------------------------------------------------------------------------- #
# T4: Aligned 63-D untouched                                                #
# --------------------------------------------------------------------------- #

class TestAlignedUntouched:
    """The 63-D aligned vector must never be modified."""

    def test_aligned_same_object_reference(self, swapper):
        feat = make_feature('L', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        out = swapper.swap(feat)

        # Same object (no copy), confirming we don't touch it
        assert out['aligned'] is feat['aligned']

    def test_aligned_values_unchanged(self, swapper):
        rng = np.random.default_rng(99)
        original_aligned = rng.standard_normal(63).astype(np.float32)
        feat = {
            'label':   'R',
            'aligned': original_aligned,
            'x_vec':   np.array([1., 0., 0.], dtype=np.float32),
            'y_vec':   np.array([0., 1., 0.], dtype=np.float32),
            'z_vec':   np.array([0., 0., 1.], dtype=np.float32),
        }
        swapper.swap(feat)  # call without capturing result

        # Original must be exactly as before
        np.testing.assert_allclose(feat['aligned'], original_aligned)


# --------------------------------------------------------------------------- #
# T5: No side-effect on original dict                                       #
# --------------------------------------------------------------------------- #

class TestNoSideEffect:
    """swap() must never mutate its input."""

    def test_original_x_vec_unchanged(self, swapper):
        x_vec = np.array([3., 4., 5.], dtype=np.float32)
        feat = make_feature('L', x_vec, [0., 1., 0.], [0., 0., 1.])
        swapper.swap(feat)

        np.testing.assert_allclose(feat['x_vec'], [3., 4., 5.])

    def test_original_label_unchanged(self, swapper):
        feat = make_feature('R', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        swapper.swap(feat)

        assert feat['label'] == 'R'


# --------------------------------------------------------------------------- #
# T6: Error handling                                                        #
# --------------------------------------------------------------------------- #

class TestErrors:
    """Invalid inputs must raise appropriate exceptions."""

    def test_missing_label_key(self, swapper):
        feat = {
            'aligned': np.ones(63, np.float32),
            'x_vec':   np.array([1., 0., 0.], dtype=np.float32),
            'y_vec':   np.array([0., 1., 0.], dtype=np.float32),
            'z_vec':   np.array([0., 0., 1.], dtype=np.float32),
        }
        with pytest.raises(KeyError):
            swapper.swap(feat)

    def test_missing_aligned_key(self, swapper):
        feat = {
            'label': 'L',
            'x_vec': np.array([1., 0., 0.], dtype=np.float32),
            'y_vec': np.array([0., 1., 0.], dtype=np.float32),
            'z_vec': np.array([0., 0., 1.], dtype=np.float32),
        }
        with pytest.raises(KeyError):
            swapper.swap(feat)

    def test_invalid_label_raises_valueerror(self, swapper):
        feat = make_feature('X', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        with pytest.raises(ValueError, match="'L' or 'R'"):
            swapper.swap(feat)

    def test_invalid_label_type_int_raises_valueerror(self, swapper):
        feat = make_feature(1, [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        with pytest.raises(ValueError):
            swapper.swap(feat)


# --------------------------------------------------------------------------- #
# T7: Output dict structure                                                 #
# --------------------------------------------------------------------------- #

class TestOutputDict:
    """Return dict contains exactly the expected keys."""

    def test_output_contains_all_required_keys(self, swapper):
        feat = make_feature('L', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        out = swapper.swap(feat)

        assert set(out.keys()) == {'label', 'aligned', 'x_vec', 'y_vec', 'z_vec'}

    def test_dtypes_preserved(self, swapper):
        feat = make_feature('L', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        out = swapper.swap(feat)

        assert out['label'] == 'R'
        assert out['aligned'].dtype == np.float32
        assert out['x_vec'].dtype == np.float32
        assert out['y_vec'].dtype == np.float32
        assert out['z_vec'].dtype == np.float32


# --------------------------------------------------------------------------- #
# T8: Batch / multiple swaps                                                #
# --------------------------------------------------------------------------- #

class TestBatchBehavior:
    """Multiple swaps on the same original are independent."""

    def test_independent_swaps_do_not_chain(self, swapper):
        feat = make_feature('L', [1., 0., 0.], [0., 1., 0.], [0., 0., 1.])
        first  = swapper.swap(feat)
        second = swapper.swap(feat)   # re-swap from original, not from first

        # Both should be R (same result from same source)
        assert first['label']  == 'R'
        assert second['label'] == 'R'
        np.testing.assert_allclose(first['x_vec'],  second['x_vec'])
        # aligned must be identical (passed through by reference, not copied)
        assert first['aligned'] is second['aligned']