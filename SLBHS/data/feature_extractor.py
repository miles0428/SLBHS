"""
HandFeatureExtractor: Extract 63d hand pose features from image or array.

Math references:
- HandPoseTracker.process_landmarks_math() from buildDatasetFromVideo_copy
- compute_orthogonal_basis() for x/y/z_vec calculation
"""

import numpy as np
import cv2
import os

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandFeatureExtractor:
    """
    Extract hand pose features (aligned_63d + orthogonal basis) from image.

    Input:
        image: np.ndarray, shape=(H,W,3) in BGR (OpenCV format)
               or RGB depending on `rgb_input` flag

    Output: dict with keys:
        aligned_63d : np.ndarray (63,) — 21 keypoints × 3 axes, wrist-aligned
        x_vec : np.ndarray (3,) — orthogonal basis X
        y_vec : np.ndarray (3,) — orthogonal basis Y
        z_vec : np.ndarray (3,) — palm normal
        handedness : str — "Left" or "Right"
        wrist_px : np.ndarray (3,) — wrist pixel coordinates
    """

    def __init__(self, model_path="hand_landmarker.task", num_hands=2,
                 min_detection_confidence=0.3, min_presence_confidence=0.3,
                 min_tracking_confidence=0.3, rgb_input=False):
        """
        Args:
            model_path: path to MediaPipe hand_landmarker.task
            num_hands: 1 or 2
            min_detection_confidence: float
            min_presence_confidence: float
            min_tracking_confidence: float
            rgb_input: if True, treat input as RGB; else BGR (OpenCV default)
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.rgb_input = rgb_input

    def extract(self, image):
        """
        Extract features from a single image.

        Args:
            image: np.ndarray (H,W,3) in BGR or RGB

        Returns:
            list[dict]: list of hand feature dicts, each with keys:
                  {'aligned_63d', 'x_vec', 'y_vec', 'z_vec', 'handedness', 'wrist_px'}
                  Returns empty list [] if no hand detected
        """
        h, w, _ = image.shape

        if self.rgb_input:
            image_rgb = image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)

        if not results.hand_landmarks or not results.handedness:
            return []

        # Process ALL detected hands
        all_hands = []
        for lms, hand_obj in zip(results.hand_landmarks, results.handedness):
            # Mirror label (same logic as source file)
            raw_label = hand_obj[0].category_name
            label = "Left" if raw_label == "Right" else "Right"

            lms_array = np.array([[lm.x, lm.y, lm.z] for lm in lms])
            all_hands.append(self._process_landmarks_math(lms_array, label, w, h))

        return all_hands

    def _process_landmarks_math(self, lms_array, label, w, h):
        """Core math: wrist-origin alignment + orthogonal basis computation."""
        wrist_origin = lms_array[0].copy()
        lms_centered = lms_array - wrist_origin

        # Distance normalization
        d = np.linalg.norm(lms_centered[9])
        if d > 0:
            lms_centered /= d

        # Pixel coordinates
        wrist_pixel = np.array([
            wrist_origin[0] * w,
            wrist_origin[1] * h,
            wrist_origin[2] * w
        ])

        R, x_vec, y_vec, z_vec = self._compute_orthogonal_basis(lms_centered, label)
        aligned_lms = np.dot(lms_centered, R)

        return {
            'aligned_63d': aligned_lms.flatten().astype(np.float32),
            'x_vec': x_vec.astype(np.float32),
            'y_vec': y_vec.astype(np.float32),
            'z_vec': z_vec.astype(np.float32),
            'handedness': label,
            'wrist_px': wrist_pixel.astype(np.float32)
        }

    def _compute_orthogonal_basis(self, norm_lms, label):
        """Compute orthonormal basis (X, Y, Z) from normalized landmarks."""
        # Z axis = palm normal (cross of index base and pinky base vectors)
        p1 = norm_lms[5]   # Index MCP
        p2 = norm_lms[17]  # Pinky MCP
        z_axis = np.cross(p1 - norm_lms[0], p2 - norm_lms[0])
        z_axis /= (np.linalg.norm(z_axis) + 1e-8)
        if label == 'Right':
            z_axis *= -1

        # Y axis = finger direction (index tip - wrist)
        y_temp = norm_lms[9] - norm_lms[0]
        y_temp /= (np.linalg.norm(y_temp) + 1e-8)

        # X axis = cross(Y, Z)
        x_axis = np.cross(y_temp, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-8)

        # Reconstruct Y = cross(Z, X)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= (np.linalg.norm(y_axis) + 1e-8)

        if label == 'Right':
            x_axis *= -1

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return R, x_axis, y_axis, z_axis

    def extract_batch(self, images):
        """
        Extract from a list of images.

        Args:
            images: list of np.ndarray

        Returns:
            list of list[dict]: one list per image, each containing 0 or more hand dicts
        """
        return [self.extract(img) for img in images]