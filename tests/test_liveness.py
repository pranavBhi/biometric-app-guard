"""Unit tests for EAR calculation, BlinkDetector, and liveness verification."""

import unittest
import numpy as np
from utils.liveness import (
    calculate_ear,
    is_blinking,
    BlinkDetector,
    LivenessDetector,
    euclidean_distance,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)


class TestLiveness(unittest.TestCase):
    """Test suite for liveness detection and EAR calculations."""

    def test_euclidean_distance(self):
        self.assertAlmostEqual(euclidean_distance((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(euclidean_distance((1, 1), (1, 1)), 0.0)

    def test_calculate_ear_open_eye(self):
        """Synthetic open eye landmarks: wide vertical opening relative to width."""
        # Eye width = 40 (p1 at (10, 20), p4 at (50, 20))
        # Top points: p2 at (20, 15), p3 at (40, 15)
        # Bottom points: p6 at (20, 25), p5 at (40, 25)
        # Vertical 1 = 10, Vertical 2 = 10. EAR = (10 + 10) / (2 * 40) = 20 / 80 = 0.25
        landmarks = {
            LEFT_EYE_INDICES[0]: (10.0, 20.0),  # p1
            LEFT_EYE_INDICES[1]: (20.0, 15.0),  # p2
            LEFT_EYE_INDICES[2]: (40.0, 15.0),  # p3
            LEFT_EYE_INDICES[3]: (50.0, 20.0),  # p4
            LEFT_EYE_INDICES[4]: (40.0, 25.0),  # p5
            LEFT_EYE_INDICES[5]: (20.0, 25.0),  # p6
        }
        ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
        self.assertAlmostEqual(ear, 0.25, places=4)

    def test_calculate_ear_closed_eye(self):
        """Synthetic closed eye landmarks: nearly zero vertical distance."""
        # Top points at y=19.5, bottom points at y=20.5 (distance = 1.0)
        # Horizontal width = 40. EAR = (1 + 1) / (2 * 40) = 2 / 80 = 0.025
        landmarks = {
            LEFT_EYE_INDICES[0]: (10.0, 20.0),
            LEFT_EYE_INDICES[1]: (20.0, 19.5),
            LEFT_EYE_INDICES[2]: (40.0, 19.5),
            LEFT_EYE_INDICES[3]: (50.0, 20.0),
            LEFT_EYE_INDICES[4]: (40.0, 20.5),
            LEFT_EYE_INDICES[5]: (20.0, 20.5),
        }
        ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
        self.assertAlmostEqual(ear, 0.025, places=4)

    def test_is_blinking_open_and_closed_eyes(self):
        """Verify is_blinking returns (False, EAR) when open and (True, EAR) when closed."""
        w, h = 640, 480
        # Normalized coordinates so that lm.x * w and lm.y * h match exact test pixels
        open_landmarks = {
            LEFT_EYE_INDICES[0]: (10.0 / w, 20.0 / h),
            LEFT_EYE_INDICES[1]: (20.0 / w, 15.0 / h),
            LEFT_EYE_INDICES[2]: (40.0 / w, 15.0 / h),
            LEFT_EYE_INDICES[3]: (50.0 / w, 20.0 / h),
            LEFT_EYE_INDICES[4]: (40.0 / w, 25.0 / h),
            LEFT_EYE_INDICES[5]: (20.0 / w, 25.0 / h),
            RIGHT_EYE_INDICES[0]: (70.0 / w, 20.0 / h),
            RIGHT_EYE_INDICES[1]: (80.0 / w, 15.0 / h),
            RIGHT_EYE_INDICES[2]: (100.0 / w, 15.0 / h),
            RIGHT_EYE_INDICES[3]: (110.0 / w, 20.0 / h),
            RIGHT_EYE_INDICES[4]: (100.0 / w, 25.0 / h),
            RIGHT_EYE_INDICES[5]: (80.0 / w, 25.0 / h),
        }
        is_closed, ear = is_blinking(open_landmarks, w, h)
        self.assertFalse(is_closed)
        self.assertAlmostEqual(ear, 0.25, places=4)

        closed_landmarks = {
            LEFT_EYE_INDICES[0]: (10.0 / w, 20.0 / h),
            LEFT_EYE_INDICES[1]: (20.0 / w, 19.5 / h),
            LEFT_EYE_INDICES[2]: (40.0 / w, 19.5 / h),
            LEFT_EYE_INDICES[3]: (50.0 / w, 20.0 / h),
            LEFT_EYE_INDICES[4]: (40.0 / w, 20.5 / h),
            LEFT_EYE_INDICES[5]: (20.0 / w, 20.5 / h),
            RIGHT_EYE_INDICES[0]: (70.0 / w, 20.0 / h),
            RIGHT_EYE_INDICES[1]: (80.0 / w, 19.5 / h),
            RIGHT_EYE_INDICES[2]: (100.0 / w, 19.5 / h),
            RIGHT_EYE_INDICES[3]: (110.0 / w, 20.0 / h),
            RIGHT_EYE_INDICES[4]: (100.0 / w, 20.5 / h),
            RIGHT_EYE_INDICES[5]: (80.0 / w, 20.5 / h),
        }
        is_closed, ear = is_blinking(closed_landmarks, w, h)
        self.assertTrue(is_closed)
        self.assertAlmostEqual(ear, 0.025, places=4)

    def test_blink_detector_valid_blink(self):
        """Test a normal blink cycle: Open -> Closed (3 frames) -> Open."""
        detector = BlinkDetector(ear_threshold=0.20, consec_frames_min=2, consec_frames_max=10)

        # Open eyes
        blink_occurred, is_closed = detector.update(0.30)
        self.assertFalse(blink_occurred)
        self.assertFalse(is_closed)

        # Eye starts closing (frame 1)
        blink_occurred, is_closed = detector.update(0.15)
        self.assertFalse(blink_occurred)
        self.assertTrue(is_closed)

        # Eye closed (frame 2)
        blink_occurred, is_closed = detector.update(0.12)
        self.assertFalse(blink_occurred)
        self.assertTrue(is_closed)

        # Eye closed (frame 3)
        blink_occurred, is_closed = detector.update(0.14)
        self.assertFalse(blink_occurred)
        self.assertTrue(is_closed)

        # Eye reopens -> Blink completes!
        blink_occurred, is_closed = detector.update(0.28)
        self.assertTrue(blink_occurred)
        self.assertFalse(is_closed)
        self.assertEqual(detector.total_blinks, 1)

    def test_blink_detector_prolonged_closure(self):
        """Test prolonged closure (e.g. squint or sleep) is rejected as a blink."""
        detector = BlinkDetector(ear_threshold=0.20, consec_frames_min=2, consec_frames_max=4)

        detector.update(0.30)
        # Closed for 5 frames (> max of 4)
        for _ in range(5):
            detector.update(0.10)

        # Reopens
        blink_occurred, is_closed = detector.update(0.30)
        self.assertFalse(blink_occurred)
        self.assertEqual(detector.total_blinks, 0)

    def test_liveness_detector_empty_frame(self):
        """Verify LivenessDetector handles empty/blank frames gracefully without crash."""
        with LivenessDetector() as detector:
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.process_frame(blank_frame)
            self.assertFalse(result.face_detected)
            self.assertEqual(result.total_blinks, 0)


if __name__ == "__main__":
    unittest.main()
