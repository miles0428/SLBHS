"""
ThetaClusterer: Finger-angle based hand pose classification.

Three-stage pipeline on aligned_63d data:
  1. Theta Extraction  — reshape (N,63)→(N,21,3), compute 9 bending angles
  2. Fuzzy Coarse Coding — quantise angles via overlapping state bands → bitstrings
  3. Frequency Classification — each unique bitstring IS a class; OOV fallback by
     Hamming similarity (most shared 1-bits wins)

Interface mirrors KMeansClusterer: fit / predict / save / load.
"""
import numpy as np
import os
import json
import glob
import h5py
import logging
from collections import Counter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Angle index map (1-based landmark indices used in MediaPipe convention)
# --------------------------------------------------------------------------
_THETA_INDEX = {
    0: (2,  3,  4),   # thumb MCP→IP→TIP
    1: (5,  6,  7),   # index PIP
    2: (6,  7,  8),   # index DIP
    3: (9, 10, 11),   # middle PIP
    4: (10, 11, 12),  # middle DIP
    5: (13, 14, 15),  # ring PIP
    6: (14, 15, 16),  # ring DIP
    7: (17, 18, 19),  # pinky PIP
    8: (18, 19, 20),  # pinky DIP
}

# Spread angle index: adjacent PIP pairs (1-based MediaPipe landmark indices)
_SPREAD_INDEX = {
    0: (2, 6),    # Thumb PIP → Index PIP
    1: (6, 10),   # Index PIP → Middle PIP
    2: (10, 14),  # Middle PIP → Ring PIP
    3: (14, 18),  # Ring PIP → Pinky PIP
}

_N_BENDING_ANGLES = 9
_N_SPREAD_ANGLES = 4
_N_ANGLES = _N_BENDING_ANGLES + _N_SPREAD_ANGLES  # 13 total
_N_STATES = 5          # 5 fuzzy states per angle
_N_BITS = _N_ANGLES * _N_STATES  # 65 bits total

# --------------------------------------------------------------------------
# Fuzzy state boundaries (in degrees, shared across all angles)
# Q=5: straight finger = high angle (120°~180°), curled = low angle (0°~60°)
# -------------------------------------------------------------------------
_STATE_BOUNDS = [
    (  0.0,  60.0),   # state 0: Tight fist — 0° ~ 60°
    ( 30.0,  90.0),   # state 1: Large bend — 30° ~ 90°
    ( 60.0, 120.0),   # state 2: Half bent — 60° ~ 120°
    ( 90.0, 150.0),   # state 3: Slight bend — 90° ~ 150°
    (120.0, 180.0),   # state 4: Straight — 120° ~ 180°
]

# Spread angle state boundaries (for 4 spread angles)
_STATE_BOUNDS_SPREAD = [
    (0.0, 15.0),    # state 0: Closed — 0°~15°
    (7.5, 22.5),    # state 1: Converging — 7.5°~22.5°
    (15.0, 30.0),   # state 2: Middle — 15°~30°
    (22.5, 37.5),   # state 3: Spread — 22.5°~37.5°
    (30.0, 60.0),   # state 4: Fully open — 30°~60°
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _compute_theta(A, B, C):
    """
    Central angle (in degrees) subtended by chord AC at point B.
    A, B, C : ndarray (3,) — 3D landmark coordinates
    B is the vertex (joint).
    """
    v1 = A - B          # B→A vector
    v2 = C - B          # B→C vector
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2) / (norm1 * norm2 + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg  # direct interior angle θ (0°~180°)


def _compute_spread(A, B, C):
    """
    Compute the spread angle between two finger base vectors at the wrist.
    A, B, C : ndarray (3,) — 3D landmark coordinates
    A = MCP_a vector (from wrist to MCP_a)
    B = wrist (origin)
    C = MCP_b vector (from wrist to MCP_b)

    Returns angle in degrees: 0° = fingers together, 180° = fingers spread wide.
    """
    v1 = A - B  # MCP_a → wrist
    v2 = C - B  # MCP_b → wrist
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_angle = np.dot(v1, v2) / (norm1 * norm2 + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def _extract_thetas(X, theta_index=_THETA_INDEX, spread_index=_SPREAD_INDEX):
    """
    Extract the 9 finger-bending angles and 4 finger-spread angles from aligned_63d data.

    Args:
        X : np.ndarray (N, 63)
    Returns:
        theta : np.ndarray (N, 13) — 9 bending + 4 spread angles in degrees
    """
    if X.ndim != 2 or X.shape[1] != 63:
        raise ValueError(f'X must be (N, 63); got shape {X.shape}')

    N = X.shape[0]
    n_bending = _N_BENDING_ANGLES  # 9
    n_spread = _N_SPREAD_ANGLES    # 4
    theta = np.full((N, n_bending + n_spread), np.nan, dtype=np.float64)

    wrist_coords = np.zeros(3)

    # Extract bending angles (indices 0-8)
    for idx, (a, b, c) in theta_index.items():
        A = X[:, a * 3 : a * 3 + 3]   # (N, 3)
        B = X[:, b * 3 : b * 3 + 3]
        C = X[:, c * 3 : c * 3 + 3]

        for row in range(N):
            try:
                Theta = _compute_theta(A[row], B[row], C[row])
                if np.isfinite(Theta):
                    theta[row, idx] = Theta
                else:
                    theta[row, idx] = np.nan
            except Exception:
                theta[row, idx] = np.nan

    # Extract spread angles (indices 9-12)
    for idx, (mcp_a, mcp_b) in spread_index.items():
        vec_a = X[:, mcp_a * 3 : mcp_a * 3 + 3] - wrist_coords
        vec_b = X[:, mcp_b * 3 : mcp_b * 3 + 3] - wrist_coords
        for row in range(N):
            try:
                spread = _compute_spread(vec_a[row], wrist_coords, vec_b[row])
                if np.isfinite(spread):
                    theta[row, n_bending + idx] = spread
                else:
                    theta[row, n_bending + idx] = np.nan
            except Exception:
                theta[row, n_bending + idx] = np.nan

    return theta


def _theta_to_bitstring(theta, q=5):
    """
    Fuzzy coarse coding: map 13 angles → 65-bit multi-hot bitstring.

    Args:
        theta : np.ndarray (N, 13)
        q     : int, number of state intervals (default 5)
    Returns:
        bits  : np.ndarray (N, 65) — uint8, values 0 or 1
    """
    if q != 5:
        raise ValueError(f'ThetaClusterer currently only supports q=5 (got q={q})')

    N = theta.shape[0]
    n_bending = _N_BENDING_ANGLES  # 9
    n_spread = _N_SPREAD_ANGLES    # 4
    bits = np.zeros((N, _N_BITS), dtype=np.uint8)

    for row in range(N):
        # First 9 bending angles
        for angle_idx in range(n_bending):
            Theta = theta[row, angle_idx]
            bit_offset = angle_idx * _N_STATES

            if not np.isfinite(Theta):
                continue

            if Theta < 0.0 or Theta > 180.0:
                logger.warning('[ThetaClusterer] Bending angle out of [0°,180°] range: %.2f° at row %d, angle %d',
                               Theta, row, angle_idx)

            for state_idx, (lo, hi) in enumerate(_STATE_BOUNDS):
                if lo <= Theta < hi:
                    bits[row, bit_offset + state_idx] = 1

        # Last 4 spread angles
        for angle_idx in range(n_spread):
            Theta = theta[row, n_bending + angle_idx]
            bit_offset = (n_bending + angle_idx) * _N_STATES

            if not np.isfinite(Theta):
                continue

            if Theta < 0.0 or Theta > 180.0:
                logger.warning('[ThetaClusterer] Spread angle out of [0°,180°] range: %.2f° at row %d, angle %d',
                               Theta, row, n_bending + angle_idx)

            for state_idx, (lo, hi) in enumerate(_STATE_BOUNDS_SPREAD):
                if lo <= Theta < hi:
                    bits[row, bit_offset + state_idx] = 1

    return bits


def _bitstring_to_bytes(bs_arr):
    """
    Convert (N, 65) uint8 bitstrings to a list of bytes objects (one per row).

    Returns:
        list of bytes objects, length N
    """
    N = bs_arr.shape[0]
    padded = int(np.ceil(_N_BITS / 8)) * 8
    n_bytes = padded // 8

    result = []
    for row in bs_arr:
        b = 0
        shift = 0
        for bit in row:
            if bit:
                b |= (1 << shift)
            shift += 1
            if shift == 8:
                result.append(b)
                b = 0
                shift = 0
        if shift > 0:
            result.append(b)

    out = np.array(result, dtype=np.uint8)
    rows = []
    for i in range(N):
        rows.append(bytes(out[i * n_bytes:(i + 1) * n_bytes]))
    return rows


def _shared_bits(b1, b2):
    """Count shared 1-bits between two byte-strings (OOV fallback)."""
    shared = int.from_bytes(b1, 'little') & int.from_bytes(b2, 'little')
    return bin(shared).count('1')


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------

class ThetaClusterer:
    """
    Fit / predict hand-pose classes based on finger-bending angles.

    New architecture (H5-based training):
        tc = ThetaClusterer(results_dir='/path/to/output')
        tc.fit('/path/to/h5_folder', top_k=10000)   # train from H5 files
        hist = tc.histogram(n=20)                    # inspect top N codes
        tc.save('/path/to/model')                   # save model
        tc.load('/path/to/model')                   # load model
        labels = tc.predict(X_new)                  # predict (M, 63)

    Legacy architecture (array-based training, backward compatible):
        tc = ThetaClusterer(results_dir='/path/to/output')
        tc.fit(X)               # X : (N, 63) aligned_63d
        labels = tc.predict(X_new)
        tc.save()
        tc.load()
    """

    # Class-level constants (preserve for external access)
    _THETA_INDEX = _THETA_INDEX
    _SPREAD_INDEX = _SPREAD_INDEX
    _STATE_BOUNDS = _STATE_BOUNDS
    _STATE_BOUNDS_SPREAD = _STATE_BOUNDS_SPREAD

    def __init__(self, results_dir=None):
        """
        Args:
            results_dir : str, directory for model artefacts.
        """
        self.results_dir = results_dir or self._default_results_dir()
        os.makedirs(self.results_dir, exist_ok=True)

        # Fitted artefacts (legacy byte-keyed)
        self.bitstring_to_label_ = {}   # {bytes : int label}
        self.bitstring_freq_     = {}   # {bytes : int count}
        self.n_classes_          = None
        self.q_                  = 5

        # New architecture (string-keyed, set by fit_h5 or convert_legacy)
        self.bitstring_to_label_str_ = None  # {str : label_id}
        self.label_to_bitstring_ = None       # {label_id : str}
        self.label_counts_ = None              # {label_id : count}
        self.top_k_ = None

    @staticmethod
    def _default_results_dir():
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'results')

    # --------------------------------------------------------------------------
    # Fit (H5-based, new architecture)
    # --------------------------------------------------------------------------

    def fit(self, h5_folder=None, X=None, top_k=1024, verbose=True):
        """
        Train the classifier.

        New mode (h5_folder):
            fit(h5_folder='/path/to/h5/folder', top_k=10000)
            → scan all *.h5 files, extract thetas, build top-k frequency table

        Legacy mode (X array, backward compatible):
            fit(X=np.array)
            → use existing bitstring statistics from X

        Args:
            h5_folder : str, path to folder containing *.h5 files (new mode)
            X         : np.ndarray (N, 63), aligned hand poses (legacy mode)
            top_k     : int, only keep top_k most frequent codes (new mode)
            verbose   : bool
        Returns:
            self (chainable)
        """
        if h5_folder is not None:
            return self.fit_h5(h5_folder, top_k=top_k, verbose=verbose)
        elif X is not None:
            return self.fit_legacy(X)
        else:
            raise ValueError("Must provide either h5_folder or X")

    def fit_h5(self, h5_folder, top_k=10000, verbose=True):
        """
        Train from H5 files.

        Args:
            h5_folder: Path to H5 file folder
            top_k: Only keep top_k most frequent codes
            verbose: Print progress
        Returns:
            self
        """
        h5_files = glob.glob(os.path.join(h5_folder, '*.h5'))
        if not h5_files:
            raise ValueError(f"No H5 files found: {h5_folder}")

        if verbose:
            print(f"[ThetaClusterer] Found {len(h5_files)} H5 files")

        # Collect all bitstrings
        all_bitstrings = []
        total_frames = 0

        for h5_path in h5_files:
            if verbose:
                print(f"[ThetaClusterer] Processing: {os.path.basename(h5_path)}")

            with h5py.File(h5_path, 'r') as f:
                if 'aligned_63d' not in f:
                    if verbose:
                        print(f"[ThetaClusterer] Skipping (no aligned_63d)")
                    continue

                X = f['aligned_63d'][:]
                thetas = _extract_thetas(X)
                bitstrings = _theta_to_bitstring(thetas)

                for i in range(len(bitstrings)):
                    bit_str = ''.join(bitstrings[i].astype(str))
                    all_bitstrings.append(bit_str)

                total_frames += len(bitstrings)

        if verbose:
            print(f"[ThetaClusterer] Total frames: {total_frames}")

        # Count frequency
        counter = Counter(all_bitstrings)
        if verbose:
            print(f"[ThetaClusterer] Unique encodings: {len(counter)}")

        # Take top_k
        self.top_k_ = top_k
        top_items = counter.most_common(top_k)

        # Build string-keyed mapping
        self.bitstring_to_label_str_ = {}
        self.label_to_bitstring_ = {}
        self.label_counts_ = {}

        for label_id, (bitstring, count) in enumerate(top_items):
            self.bitstring_to_label_str_[bitstring] = label_id
            self.label_to_bitstring_[label_id] = bitstring
            self.label_counts_[label_id] = count

        self.n_classes_ = len(self.bitstring_to_label_str_)

        if verbose:
            print(f"[ThetaClusterer] Classes after trimming: {self.n_classes_}")

        return self

    def fit_legacy(self, X, q=5):
        """
        Legacy fit from array (backward compatible).
        Args:
            X : np.ndarray (N, 63)
            q : int
        Returns:
            self
        """
        self.q_ = q

        logger.info('[ThetaClusterer] Extracting thetas from X=%s ...', X.shape)
        theta = _extract_thetas(X)
        bad = np.sum(~np.isfinite(theta), axis=1).astype(bool)
        logger.info('[ThetaClusterer] thetas shape=%s  NaN/Inf rows=%d', theta.shape, bad.sum())

        self.n_angles_ = theta.shape[1]
        self.n_bending_ = _N_BENDING_ANGLES
        self.n_spread_ = _N_SPREAD_ANGLES

        logger.info('[ThetaClusterer] Fuzzy coarse coding (q=%d) ...', q)
        bitstrings = _theta_to_bitstring(theta, q=q)
        logger.info('[ThetaClusterer] bitstrings shape=%s', bitstrings.shape)

        logger.info('[ThetaClusterer] Building frequency map ...')
        byte_keys = _bitstring_to_bytes(bitstrings)
        freq_map = {}
        for bk in byte_keys:
            freq_map[bk] = freq_map.get(bk, 0) + 1

        sorted_keys = sorted(freq_map.keys(), key=lambda k: -freq_map[k])
        self.bitstring_to_label_ = {k: i for i, k in enumerate(sorted_keys)}
        self.bitstring_freq_ = freq_map
        self.n_classes_ = len(self.bitstring_to_label_)

        logger.info('[ThetaClusterer] Fitted. n_classes=%d  q=%d', self.n_classes_, q)
        return self

    def get_labels(self, X=None):
        """
        Alias for fit().fit_predict-compatible shim.
        """
        if X is None:
            raise ValueError('X is required for get_labels()')
        return self.predict(X)

    # --------------------------------------------------------------------------
    # Histogram
    # --------------------------------------------------------------------------

    def histogram(self, n=20):
        """
        Return top N most frequent encoding statistics.

        Args:
            n: number of top entries to return
        Returns:
            list of dict: [{'rank', 'bitstring', 'count', 'bending_states', 'spread_states'}, ...]
        """
        if self.label_counts_ is None and self.bitstring_freq_ is None:
            raise RuntimeError("Need fit() or load() before histogram()")

        result = []
        if self.label_counts_ is not None:
            # New architecture
            items = sorted(self.label_counts_.items(), key=lambda x: -x[1])
            for rank, (label_id, count) in enumerate(items, 1):
                if rank > n:
                    break
                bitstring = self.label_to_bitstring_[label_id]
                bits = np.array([int(b) for b in bitstring])

                result.append({
                    'rank': rank,
                    'label': label_id,
                    'bitstring': bitstring,
                    'count': count,
                    'bending_states': [self._format_bending_states(bits, i) for i in range(_N_BENDING_ANGLES)],
                    'spread_states': [self._format_spread_states(bits, i) for i in range(_N_SPREAD_ANGLES)],
                })
        else:
            # Legacy architecture — build from byte keys
            items = sorted(self.bitstring_freq_.items(), key=lambda x: -x[1])
            for rank, (bk, count) in enumerate(items, 1):
                if rank > n:
                    break
                bits_arr = np.unpackbits(np.frombuffer(bk * (65 // 8 + 1), dtype=np.uint8))[:_N_BITS]
                bitstring = ''.join(bits_arr.astype(str))

                result.append({
                    'rank': rank,
                    'label': rank - 1,
                    'bitstring': bitstring,
                    'count': count,
                    'bending_states': [self._format_bending_states(bits_arr, i) for i in range(_N_BENDING_ANGLES)],
                    'spread_states': [self._format_spread_states(bits_arr, i) for i in range(_N_SPREAD_ANGLES)],
                })

        return result

    # --------------------------------------------------------------------------
    # OOV fallback helper
    # --------------------------------------------------------------------------

    def _find_oov_match(self, bitstring):
        """OOV fallback: find most similar label by shared 1-bits"""
        bits = bitstring.astype(bool)

        best_label = -1
        best_overlap = -1

        for label_id, stored_bitstring in self.label_to_bitstring_.items():
            stored_bits = np.array([int(b) for b in stored_bitstring]).astype(bool)
            overlap = np.sum(bits & stored_bits)

            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label_id
            elif overlap == best_overlap:
                if self.label_counts_[label_id] > self.label_counts_[best_label]:
                    best_label = label_id

        return best_label

    # --------------------------------------------------------------------------
    # Predict
    # --------------------------------------------------------------------------

    def predict(self, X_new):
        """
        Predict class labels for new aligned_63d data.

        OOV (unknown bitstring):
            Find the stored bitstring with the most shared 1-bits.
            Ties broken by higher frequency.

        Args:
            X_new : np.ndarray (M, 63)
        Returns:
            labels : np.ndarray (M,), int32 label IDs in [0, n_classes_-1]
        """
        if self.bitstring_to_label_ is None and self.bitstring_to_label_str_ is None:
            raise RuntimeError('Must fit() or load() before predict()')

        theta = _extract_thetas(X_new)
        bitstrings = _theta_to_bitstring(theta, q=self.q_)

        M = X_new.shape[0]
        labels = np.full(M, -1, dtype=np.int32)

        # Convert to string keys for lookup
        bitstrings_str = [''.join(bs.astype(str)) for bs in bitstrings]

        if self.bitstring_to_label_str_ is not None:
            # New architecture: string-keyed lookup + OOV fallback
            for i, bit_str in enumerate(bitstrings_str):
                if bit_str in self.bitstring_to_label_str_:
                    labels[i] = self.bitstring_to_label_str_[bit_str]
                else:
                    labels[i] = self._find_oov_match(bitstrings[i])
        else:
            # Legacy architecture: byte-keyed lookup + OOV fallback
            byte_keys = _bitstring_to_bytes(bitstrings)
            for i, bk in enumerate(byte_keys):
                if bk in self.bitstring_to_label_:
                    labels[i] = self.bitstring_to_label_[bk]
                else:
                    best_key = None
                    best_score = -1
                    for stored_key in self.bitstring_to_label_:
                        score = _shared_bits(bk, stored_key)
                        if score > best_score:
                            best_score = score
                            best_key = stored_key
                        elif score == best_score:
                            if self.bitstring_freq_.get(stored_key, 0) > self.bitstring_freq_.get(best_key, 0):
                                best_key = stored_key
                    if best_key is None:
                        best_key = min(self.bitstring_to_label_,
                                        key=lambda k: -self.bitstring_freq_.get(k, 0))
                    labels[i] = self.bitstring_to_label_[best_key]

        return labels

    # --------------------------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------------------------

    def save(self, path=None):
        """
        Persist model to directory.

        Supports both legacy byte-keyed and new string-keyed architectures.

        Directory layout:
            theta_labels.json   — legacy: bitstring key → label_id (base64)
            theta_labels_str.json — new: bitstring (str) → label_id
            theta_freq.json     — legacy: bitstring key → count (base64)
            theta_freq_str.json — new: label_id → count
            theta_meta.json     — metadata
        """
        out_dir = path if path else self.results_dir
        os.makedirs(out_dir, exist_ok=True)

        import base64
        import codecs

        # Legacy byte-keyed architecture
        if self.bitstring_to_label_:
            label_json = {}
            for bk, label_id in self.bitstring_to_label_.items():
                b64 = codecs.encode(bk, 'base64').decode()
                label_json[b64] = int(label_id)

            freq_json = {}
            for bk, count in self.bitstring_freq_.items():
                b64 = codecs.encode(bk, 'base64').decode()
                freq_json[b64] = int(count)

            labels_path = os.path.join(out_dir, 'theta_labels.json')
            freq_path   = os.path.join(out_dir, 'theta_freq.json')

            with open(labels_path, 'w') as f:
                json.dump(label_json, f)
            with open(freq_path, 'w') as f:
                json.dump(freq_json, f)

        # New string-keyed architecture
        if self.bitstring_to_label_str_ is not None:
            # bitstring_to_label_str_: {str: int} → save directly
            labels_str_path = os.path.join(out_dir, 'theta_labels_str.json')
            with open(labels_str_path, 'w') as f:
                json.dump(self.bitstring_to_label_str_, f)

            # label_to_bitstring_: {int: str} → save directly
            with open(os.path.join(out_dir, 'theta_ltb_str.json'), 'w') as f:
                json.dump(self.label_to_bitstring_, f)

            # label_counts_: {int: int} → save directly
            if self.label_counts_ is not None:
                with open(os.path.join(out_dir, 'theta_counts_str.json'), 'w') as f:
                    json.dump(self.label_counts_, f)

        meta = {
            'n_classes': self.n_classes_,
            'q': self.q_,
            'n_bits': _N_BITS,
            'n_angles': _N_ANGLES,
            'n_bending': _N_BENDING_ANGLES,
            'n_spread': _N_SPREAD_ANGLES,
            'top_k': getattr(self, 'top_k_', None),
            'has_new_architecture': self.bitstring_to_label_str_ is not None,
        }

        meta_path = os.path.join(out_dir, 'theta_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info('[ThetaClusterer] Saved to %s (%d classes)', out_dir, self.n_classes_)
        return {'meta': meta_path}

    def load(self, path=None):
        """
        Load model artefacts from directory.

        Supports both legacy byte-keyed and new string-keyed architectures.
        Automatically detects which architecture is present.
        """
        in_dir = path if path else self.results_dir

        meta_path   = os.path.join(in_dir, 'theta_meta.json')

        with open(meta_path) as f:
            meta = json.load(f)

        has_new = meta.get('has_new_architecture', False)

        if has_new:
            # Load new string-keyed architecture
            labels_str_path = os.path.join(in_dir, 'theta_labels_str.json')
            ltb_str_path    = os.path.join(in_dir, 'theta_ltb_str.json')
            counts_str_path = os.path.join(in_dir, 'theta_counts_str.json')

            with open(labels_str_path) as f:
                self.bitstring_to_label_str_ = json.load(f)

            with open(ltb_str_path) as f:
                self.label_to_bitstring_ = json.load(f)

            with open(counts_str_path) as f:
                self.label_counts_ = json.load(f)

            # Ensure label_id keys are int (JSON may have string keys)
            self.bitstring_to_label_str_ = {k: int(v) for k, v in self.bitstring_to_label_str_.items()}
            self.label_to_bitstring_ = {int(k): v for k, v in self.label_to_bitstring_.items()}
            self.label_counts_ = {int(k): int(v) for k, v in self.label_counts_.items()}

            # Clear legacy fields
            self.bitstring_to_label_ = None
            self.bitstring_freq_ = None
        else:
            # Load legacy byte-keyed architecture
            labels_path = os.path.join(in_dir, 'theta_labels.json')
            freq_path   = os.path.join(in_dir, 'theta_freq.json')

            import codecs

            with open(labels_path) as f:
                label_json = json.load(f)
            with open(freq_path) as f:
                freq_json = json.load(f)

            self.bitstring_to_label_ = {}
            self.bitstring_freq_ = {}
            for b64, label_id in label_json.items():
                bk = codecs.decode(b64.encode(), 'base64')
                self.bitstring_to_label_[bk] = int(label_id)

            for b64, count in freq_json.items():
                bk = codecs.decode(b64.encode(), 'base64')
                self.bitstring_freq_[bk] = int(count)

            # Clear new architecture fields
            self.bitstring_to_label_str_ = None
            self.label_to_bitstring_ = None
            self.label_counts_ = None

        self.n_classes_ = meta['n_classes']
        self.q_ = meta['q']
        self.top_k_ = meta.get('top_k')

        logger.info('[ThetaClusterer] Loaded from %s (%d classes)', in_dir, self.n_classes_)
        return self

    # --------------------------------------------------------------------------
    # Report helpers
    # --------------------------------------------------------------------------

    def _collect_bitstring_samples(self, h5_folder):
        """
        Scan H5 folder, collect bitstring → sample list.
        Returns: dict {bitstring: [{h5, frame_idx, aligned, angles}, ...]}
        """
        bitstring_to_samples = {}
        h5_files = glob.glob(os.path.join(h5_folder, '*.h5'))

        for h5_path in h5_files:
            with h5py.File(h5_path, 'r') as f:
                if 'aligned_63d' not in f:
                    continue
                X = f['aligned_63d'][:]
                thetas = _extract_thetas(X)
                bitstrings = _theta_to_bitstring(thetas)

                for i in range(len(bitstrings)):
                    bit_str = ''.join(bitstrings[i].astype(str))
                    if bit_str not in bitstring_to_samples:
                        bitstring_to_samples[bit_str] = []
                    bitstring_to_samples[bit_str].append({
                        'h5': os.path.basename(h5_path),
                        'frame_idx': i,
                        'aligned': X[i].copy(),
                        'angles': thetas[i].copy(),
                    })

        return bitstring_to_samples

    def _format_bending_states(self, bits, angle_idx):
        """Format state interval for a single bending angle"""
        offset = angle_idx * _N_STATES
        states = []
        for state_idx in range(5):
            if bits[offset + state_idx] == 1:
                lo, hi = _STATE_BOUNDS[state_idx]
                states.append(f"{lo:.0f}-{hi:.0f}")
        return states if states else ['N/A']

    def _format_spread_states(self, bits, angle_idx):
        """Format state interval for a single spread angle"""
        offset = (_N_BENDING_ANGLES + angle_idx) * _N_STATES
        states = []
        for state_idx in range(5):
            if bits[offset + state_idx] == 1:
                lo, hi = _STATE_BOUNDS_SPREAD[state_idx]
                states.append(f"{lo:.1f}-{hi:.1f}")
        return states if states else ['N/A']

    def _format_histogram_item(self, h):
        """Format single histogram item (for generate_report)"""
        bitstring = h['bitstring']
        bits = np.array([int(b) for b in bitstring])

        names_bending = ["thumb", "index_PIP", "index_DIP", "middle_PIP", "middle_DIP",
                        "ring_PIP", "ring_DIP", "pinky_PIP", "pinky_DIP"]
        names_spread = ["thumb-index", "index-middle", "middle-ring", "ring-pinky"]

        lines = []
        lines.append(f"Bitstring: {bitstring}")
        lines.append("")
        lines.append("  Bending angle ranges:")
        for i, name in enumerate(names_bending):
            states = self._format_bending_states(bits, i)
            lines.append(f"    {name:12}: {', '.join(states)}")
        lines.append("")
        lines.append("  Spread angle ranges:")
        for i, name in enumerate(names_spread):
            states = self._format_spread_states(bits, i)
            lines.append(f"    {name:12}: {', '.join(states)}")

        return '\n'.join(lines)

    # --------------------------------------------------------------------------
    # Generate Report
    # --------------------------------------------------------------------------

    def generate_report(self, h5_folder, output_path, top_n=10, samples_per_class=3):
        """
        Generate analysis report with top N encoding sample coordinates.

        Args:
            h5_folder: H5 file folder (for raw coordinates)
            output_path: Output report file path
            top_n: Report only top_n rankings
            samples_per_class: Number of samples per class

        Returns:
            report_path: Report file path
        """
        # Collect samples
        bitstring_to_samples = self._collect_bitstring_samples(h5_folder)

        # Generate report
        lines = []
        lines.append("=" * 80)
        lines.append("ThetaClusterer Analysis Report")
        lines.append("=" * 80)
        lines.append(f"H5 folder: {h5_folder}")
        lines.append(f"H5 file count: {len(glob.glob(os.path.join(h5_folder, '*.h5')))}")
        lines.append(f"Total samples: {sum(len(v) for v in bitstring_to_samples.values())}")
        lines.append(f"Unique encodings: {len(bitstring_to_samples)}")
        lines.append("")

        for h in self.histogram(n=top_n):
            bitstring = h['bitstring']
            samples = bitstring_to_samples.get(bitstring, [])[:samples_per_class]

            lines.append("")
            lines.append("=" * 80)
            lines.append(f"Rank {h['rank']} | label={h['label']} | count={h['count']}")
            lines.append("=" * 80)
            lines.append(self._format_histogram_item(h))

            lines.append("")
            lines.append(f"  Sample coordinates ({len(samples)} total):")
            for j, sample in enumerate(samples):
                lines.append("")
                lines.append(f"    ─── Sample {j+1} ───")
                lines.append(f"    Source: {sample['h5']} (frame={sample['frame_idx']})")
                lines.append("    Aligned 63d (21 points):")
                for pt in range(21):
                    x, y, z = sample['aligned'][pt*3:pt*3+3]
                    lines.append(f"      point {pt:2d}: [{x:8.4f}, {y:8.4f}, {z:8.4f}]")
                lines.append("    13 angle values:")
                angles = sample['angles']
                names_bending = ["thumb", "index_PIP", "index_DIP", "middle_PIP", "middle_DIP",
                               "ring_PIP", "ring_DIP", "pinky_PIP", "pinky_DIP"]
                names_spread = ["thumb-index", "index-middle", "middle-ring", "ring-pinky"]
                for i, name in enumerate(names_bending):
                    lines.append(f"      {name:12}: {angles[i]:.2f}°")
                for i, name in enumerate(names_spread):
                    lines.append(f"      {name:12}: {angles[9+i]:.2f}°")

        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Report saved to {output_path}")
        return output_path
