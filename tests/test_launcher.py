"""Unit and mock tests for launcher.py."""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from launcher import parse_args, load_authorized_encoding, launch_application, run_launcher


class TestLauncher(unittest.TestCase):
    """Test suite for launcher.py functionality."""

    def test_parse_args_defaults(self):
        """Verify default CLI arguments match required architecture."""
        args = parse_args([])
        self.assertEqual(args.app, "Google Chrome")
        self.assertEqual(args.authorized_image, "data/authorized/me.jpg")
        self.assertEqual(args.tolerance, 0.55)
        self.assertEqual(args.ear_threshold, 0.21)
        self.assertFalse(args.register)
        self.assertFalse(args.headless)
        self.assertFalse(args.dry_run)

    def test_parse_args_custom(self):
        """Verify custom arguments can be passed via CLI."""
        args = parse_args(["-a", "Terminal", "-i", "data/custom.jpg", "-t", "0.45", "--headless"])
        self.assertEqual(args.app, "Terminal")
        self.assertEqual(args.authorized_image, "data/custom.jpg")
        self.assertEqual(args.tolerance, 0.45)
        self.assertTrue(args.headless)

    def test_load_authorized_encoding_missing_file(self):
        """Non-existent image returns None cleanly without throwing."""
        encoding = load_authorized_encoding("data/authorized/does_not_exist.jpg")
        self.assertIsNone(encoding)

    @patch("subprocess.run")
    def test_launch_application(self, mock_subprocess):
        """Verify subprocess.run invokes ['open', '-a', target_app]."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        success = launch_application("Google Chrome")
        self.assertTrue(success)
        mock_subprocess.assert_called_once_with(["open", "-a", "Google Chrome"], check=True)

    @patch("subprocess.run")
    def test_launch_custom_application(self, mock_subprocess):
        """Verify custom app launch."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        success = launch_application("Slack")
        self.assertTrue(success)
        mock_subprocess.assert_called_once_with(["open", "-a", "Slack"], check=True)

    def test_run_launcher_missing_image_returns_error_code(self):
        """Verify missing image exits with code 1."""
        code = run_launcher(authorized_image_path="non_existent_file.jpg", headless=True)
        self.assertEqual(code, 1)

    @patch("launcher.load_authorized_encoding")
    @patch("launcher.os.path.exists")
    @patch("launcher.cv2.VideoCapture")
    @patch("launcher.LivenessDetector")
    @patch("launcher.subprocess.run")
    @patch("launcher.face_recognition")
    def test_run_launcher_successful_auth(
        self,
        mock_fr,
        mock_subp,
        mock_liveness_cls,
        mock_cap_cls,
        mock_exists,
        mock_load_enc,
    ):
        """Verify complete flow: face matches, blink occurs -> app launched."""
        mock_exists.return_value = True
        dummy_enc = np.zeros(128)
        mock_load_enc.return_value = dummy_enc

        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        # Mock face recognition
        mock_fr.face_locations.return_value = [(10, 50, 50, 10)]
        mock_fr.face_encodings.return_value = [dummy_enc]
        mock_fr.face_distance.return_value = [0.2]  # < 0.55 tolerance

        # Mock liveness detector
        from utils.liveness import LivenessResult
        mock_detector = MagicMock()
        mock_detector.process_frame.return_value = LivenessResult(
            face_detected=True,
            ear_avg=0.25,
            blink_occurred=True,
            total_blinks=1,
        )
        mock_liveness_cls.return_value = mock_detector

        mock_subp.return_value = MagicMock(returncode=0)

        # Run launcher
        code = run_launcher(
            target_app="Google Chrome",
            authorized_image_path="data/authorized/me.jpg",
            headless=True,
            timeout=5.0,
        )

        self.assertEqual(code, 0)
        mock_subp.assert_called_once_with(["open", "-a", "Google Chrome"], check=True)


if __name__ == "__main__":
    unittest.main()
