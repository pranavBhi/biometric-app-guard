import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app_guard


class TestAppGuard(unittest.TestCase):
    """Test suite for app_guard.py control flow and session management."""

    @patch("subprocess.check_output")
    def test_ask_password_failsafe_correct_passcode(self, mock_check_output):
        """Failsafe succeeds when user enters correct passcode."""
        mock_check_output.return_value = app_guard.FAILSAFE_PASSCODE
        result = app_guard.ask_password_failsafe("Mail")
        self.assertTrue(result)

    @patch("subprocess.check_output")
    @patch("subprocess.run")
    def test_ask_password_failsafe_incorrect_passcode(self, mock_run, mock_check_output):
        """Failsafe fails when user enters incorrect passcode."""
        mock_check_output.return_value = "wrong_code"
        result = app_guard.ask_password_failsafe("Mail")
        self.assertFalse(result)

    @patch("app_guard.get_working_camera")
    @patch("app_guard.mp_mesh.process")
    @patch("app_guard.face_recognition.face_locations")
    @patch("app_guard.face_recognition.face_encodings")
    @patch("app_guard.face_recognition.face_distance")
    @patch("app_guard.is_blinking")
    @patch("app_guard.cv2.namedWindow")
    @patch("app_guard.cv2.imshow")
    @patch("app_guard.cv2.waitKey")
    def test_authenticate_face_success_returns_strict_true(
        self,
        mock_wait_key,
        mock_imshow,
        mock_named_window,
        mock_is_blinking,
        mock_face_dist,
        mock_face_enc,
        mock_face_loc,
        mock_mp_process,
        mock_get_cam,
    ):
        """Verify authenticate_face renders namedWindow, confirmation banner, and releases resources on match."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_get_cam.return_value = mock_cap

        # Mock mesh landmarks
        mock_mesh_result = MagicMock()
        mock_mesh_result.multi_face_landmarks = [MagicMock(landmark=[MagicMock()])]
        mock_mp_process.return_value = mock_mesh_result

        # Liveness blink: return closed for 2 frames, then open
        mock_is_blinking.side_effect = [(True, 0.15), (True, 0.15), (False, 0.28)]

        mock_face_loc.return_value = [(10, 50, 50, 10)]
        mock_face_enc.return_value = [np.zeros(128)]
        mock_face_dist.return_value = [0.2]  # Match <= 0.55
        mock_wait_key.return_value = -1

        result = app_guard.authenticate_face(interactive_display=True)

        self.assertIs(result, True)
        mock_named_window.assert_called_with("Biometric Authentication", app_guard.cv2.WINDOW_AUTOSIZE)
        mock_imshow.assert_called()
        mock_wait_key.assert_any_call(400)  # Green banner shown for 400ms
        mock_cap.release.assert_called()

    @patch("app_guard.time.sleep")
    @patch("subprocess.run")
    def test_elevate_to_foreground_sleeps_and_runs(self, mock_subp, mock_sleep):
        """elevate_to_foreground executes elevation and sleeps 0.3s without re-launching Python."""
        app_guard.elevate_to_foreground()
        mock_sleep.assert_called_with(0.3)
        # Ensure open -a BiometricGuard is NEVER called to prevent process cascading
        for call_args in mock_subp.call_args_list:
            args = call_args[0][0] if call_args[0] else []
            self.assertNotIn("BiometricGuard", args)

    @patch("app_guard.time.sleep")
    @patch("app_guard.cv2.VideoCapture")
    def test_get_working_camera_retries_on_initial_failure(self, mock_video_capture, mock_sleep):
        """get_working_camera does not abort on initial read failure and retries during warmup."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # First read fails (False, None), second read succeeds
        mock_cap.read.side_effect = [
            (False, None),
            (True, dummy_frame),
            (True, dummy_frame),
            (True, dummy_frame),
            (True, dummy_frame),
            (True, dummy_frame),
        ]
        mock_video_capture.return_value = mock_cap

        cap = app_guard.get_working_camera(timeout_per_index=1.0)
        self.assertIsNotNone(cap)
        self.assertEqual(cap, mock_cap)

    @patch("app_guard.time.sleep")
    @patch("app_guard.cv2.VideoCapture")
    def test_get_working_camera_polls_is_opened_until_true(self, mock_video_capture, mock_sleep):
        """get_working_camera polls isOpened() during hardware power-on before declaring failure."""
        mock_cap = MagicMock()
        # isOpened returns False first, then True
        mock_cap.isOpened.side_effect = [False, True, True, True, True, True, True]
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_video_capture.return_value = mock_cap

        cap = app_guard.get_working_camera(timeout_per_index=1.0)
        self.assertIsNotNone(cap)
        self.assertEqual(cap, mock_cap)

    @patch("app_guard.time.sleep")
    @patch("app_guard.cv2.VideoCapture")
    def test_get_working_camera_denied_sets_failure_reason(self, mock_video_capture, mock_sleep):
        """When all indices fail, get_working_camera sets LAST_AUTH_FAILURE_REASON to CAMERA_PERMISSION_DENIED."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        cap = app_guard.get_working_camera(timeout_per_index=0.1)
        self.assertIsNone(cap)
        self.assertEqual(app_guard.LAST_AUTH_FAILURE_REASON, app_guard.AUTH_REASON_CAMERA_PERMISSION_DENIED)

    def test_log_message_writes_to_file(self):
        """log_message successfully appends to /tmp/app_guard.log."""
        test_msg = "[Test] Automated test log message verification."
        app_guard.log_message(test_msg)
        with open(app_guard.LOG_FILE, "r") as f:
            content = f.read()
        self.assertIn(test_msg, content)

    @patch("subprocess.run")
    def test_dismiss_lingering_dialogs_invokes_killall(self, mock_subp):
        """dismiss_lingering_dialogs calls killall osascript."""
        app_guard.dismiss_lingering_dialogs()
        mock_subp.assert_called_with(
            ["killall", "osascript"],
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
            check=False,
        )

    @patch("app_guard.get_working_camera")
    @patch("app_guard.mp_mesh.process")
    @patch("app_guard.face_recognition.face_locations")
    @patch("app_guard.face_recognition.face_encodings")
    @patch("app_guard.face_recognition.face_distance")
    @patch("app_guard.is_blinking")
    @patch("app_guard.cv2.imshow")
    @patch("app_guard.cv2.waitKey")
    def test_authenticate_face_headless_path_does_not_call_imshow(
        self,
        mock_wait_key,
        mock_imshow,
        mock_is_blinking,
        mock_face_dist,
        mock_face_enc,
        mock_face_loc,
        mock_mp_process,
        mock_get_cam,
    ):
        """Headless verification path processes frames without calling cv2.imshow or cv2.waitKey."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_get_cam.return_value = mock_cap

        mock_mesh_result = MagicMock()
        mock_mesh_result.multi_face_landmarks = [MagicMock(landmark=[MagicMock()])]
        mock_mp_process.return_value = mock_mesh_result

        mock_is_blinking.side_effect = [(True, 0.15), (True, 0.15), (False, 0.28)]
        mock_face_loc.return_value = [(10, 50, 50, 10)]
        mock_face_enc.return_value = [np.zeros(128)]
        mock_face_dist.return_value = [0.2]

        result = app_guard.authenticate_face(interactive_display=False)

        self.assertIs(result, True)
        mock_imshow.assert_not_called()
        mock_wait_key.assert_not_called()

    @patch("app_guard.dismiss_lingering_dialogs")
    @patch("app_guard.unhide_app")
    @patch("app_guard.hide_app")
    @patch("app_guard.elevate_to_foreground")
    @patch("app_guard.authenticate_face")
    @patch("app_guard.ask_password_failsafe")
    @patch("app_guard.get_frontmost_app")
    @patch("app_guard.time.sleep", side_effect=[None, KeyboardInterrupt])
    def test_strict_sequential_flow_on_biometric_success_no_failsafe(
        self,
        mock_sleep,
        mock_get_app,
        mock_failsafe,
        mock_auth_face,
        mock_elevate,
        mock_hide,
        mock_unhide,
        mock_dismiss,
    ):
        """When authenticate_face succeeds, ask_password_failsafe is NEVER called."""
        mock_get_app.return_value = "Mail"
        mock_auth_face.return_value = True

        try:
            app_guard.main()
        except KeyboardInterrupt:
            pass

        mock_hide.assert_called_with("Mail")
        mock_elevate.assert_called()
        mock_auth_face.assert_called()
        mock_failsafe.assert_not_called()
        mock_unhide.assert_called_with("Mail")
        mock_dismiss.assert_called()

    @patch("app_guard.dismiss_lingering_dialogs")
    @patch("app_guard.unhide_app")
    @patch("app_guard.hide_app")
    @patch("app_guard.elevate_to_foreground")
    @patch("app_guard.authenticate_face")
    @patch("app_guard.ask_password_failsafe")
    @patch("app_guard.get_frontmost_app")
    @patch("app_guard.time.sleep", side_effect=[None, KeyboardInterrupt])
    def test_strict_sequential_flow_on_biometric_failure_calls_failsafe(
        self,
        mock_sleep,
        mock_get_app,
        mock_failsafe,
        mock_auth_face,
        mock_elevate,
        mock_hide,
        mock_unhide,
        mock_dismiss,
    ):
        """When authenticate_face fails, ask_password_failsafe is called only after auth completes."""
        mock_get_app.return_value = "Mail"
        mock_auth_face.return_value = False
        mock_failsafe.return_value = True

        try:
            app_guard.main()
        except KeyboardInterrupt:
            pass

        mock_hide.assert_called_with("Mail")
        mock_elevate.assert_called()
        mock_auth_face.assert_called()
        mock_failsafe.assert_called_once()
        mock_unhide.assert_called_with("Mail")

    @patch("app_guard.dismiss_lingering_dialogs")
    @patch("app_guard.unhide_app")
    @patch("app_guard.hide_app")
    @patch("app_guard.elevate_to_foreground")
    @patch("app_guard.authenticate_face")
    @patch("app_guard.ask_password_failsafe")
    @patch("app_guard.get_frontmost_app")
    @patch("app_guard.time.sleep")
    def test_unlocked_session_does_not_retrigger_while_frontmost(
        self,
        mock_sleep,
        mock_get_app,
        mock_failsafe,
        mock_auth_face,
        mock_elevate,
        mock_hide,
        mock_unhide,
        mock_dismiss,
    ):
        """As long as Mail remains the frontmost application, do NOT re-trigger authentication or passcode."""
        mock_get_app.return_value = "Mail"
        mock_auth_face.return_value = True
        # First sleep (1.0s on unlock), second sleep (0.5s on active session continue), then exit loop
        mock_sleep.side_effect = [None, KeyboardInterrupt]

        try:
            app_guard.main()
        except KeyboardInterrupt:
            pass

        # authenticate_face was called exactly once on initial focus
        mock_auth_face.assert_called_once()
        mock_failsafe.assert_not_called()
        mock_unhide.assert_called_once_with("Mail")

    def test_is_ignored_switch_app(self):
        """is_ignored_switch_app correctly identifies internal/transient apps and third-party apps."""
        ignored_list = [
            "Python",
            "python3",
            "python",
            "BiometricGuard",
            "SecurityAgent",
            "System Events",
            "osascript",
            "loginwindow",
            "",
            "   ",
            "Dock",
            "SystemUIServer",
            "NotificationCenter",
            "ControlCenter",
            "Spotlight",
            "WindowManager",
            "applet",
            "CoreServicesUIAgent",
            "python3.12",
            "Python3",
        ]
        for app in ignored_list:
            self.assertTrue(
                app_guard.is_ignored_switch_app(app),
                f"Expected '{app}' to be recognized as an ignored process",
            )

        third_party_apps = ["Mail", "Safari", "Finder", "Google Chrome", "Slack", "Notes", "Spotify"]
        for app in third_party_apps:
            self.assertFalse(
                app_guard.is_ignored_switch_app(app),
                f"Expected '{app}' NOT to be recognized as an ignored process",
            )

    @patch("app_guard.dismiss_lingering_dialogs")
    @patch("app_guard.unhide_app")
    @patch("app_guard.hide_app")
    @patch("app_guard.elevate_to_foreground")
    @patch("app_guard.authenticate_face")
    @patch("app_guard.ask_password_failsafe")
    @patch("app_guard.get_frontmost_app")
    @patch("app_guard.time.sleep")
    def test_ignored_switch_apps_keeps_session_unlocked(
        self,
        mock_sleep,
        mock_get_app,
        mock_failsafe,
        mock_auth_face,
        mock_elevate,
        mock_hide,
        mock_unhide,
        mock_dismiss,
    ):
        """Internal/transient processes (Python, System Events, osascript, etc.) do NOT relock the session."""
        mock_get_app.side_effect = [
            "Mail",
            "Python",
            "python3",
            "System Events",
            "osascript",
            "loginwindow",
            "BiometricGuard",
            "Mail",
        ]
        mock_auth_face.return_value = True
        mock_sleep.side_effect = [
            None,  # unlock 1.0s
            None,  # Mail loop end 0.5s
            None,  # Python ignore continue
            None,  # python3 ignore continue
            None,  # System Events ignore continue
            None,  # osascript ignore continue
            None,  # loginwindow ignore continue
            None,  # BiometricGuard ignore continue
            KeyboardInterrupt,  # Mail continue
        ]

        try:
            app_guard.main()
        except KeyboardInterrupt:
            pass

        # authenticate_face was called ONLY ONCE initially; ignored processes never triggered relock
        mock_auth_face.assert_called_once()
        mock_failsafe.assert_not_called()

    @patch("app_guard.time.time")
    @patch("app_guard.dismiss_lingering_dialogs")
    @patch("app_guard.unhide_app")
    @patch("app_guard.hide_app")
    @patch("app_guard.elevate_to_foreground")
    @patch("app_guard.authenticate_face")
    @patch("app_guard.ask_password_failsafe")
    @patch("app_guard.get_frontmost_app")
    @patch("app_guard.time.sleep")
    def test_transient_switch_away_within_grace_period_does_not_relock(
        self,
        mock_sleep,
        mock_get_app,
        mock_failsafe,
        mock_auth_face,
        mock_elevate,
        mock_hide,
        mock_unhide,
        mock_dismiss,
        mock_time,
    ):
        """A transient switch away to Safari/Finder (< 2.0s) followed by returning to Mail does NOT relock."""
        mock_get_app.side_effect = ["Mail", "Safari", "Mail"]
        mock_auth_face.return_value = True
        # Time progression:
        # Loop 1 (Mail unlock): 4 calls to time.time() at 100.0
        # Loop 2 (Safari start debounce): 1 call to time.time() at 100.0 (away_start = 100.0)
        # Loop 3 (Mail returns): 1 call to time.time() at 100.5 (elapsed = 0.5s < 2.0s -> cancels relock)
        mock_time.side_effect = [100.0, 100.0, 100.0, 100.0, 100.0, 100.5, 100.5]
        mock_sleep.side_effect = [None, None, None, KeyboardInterrupt]

        try:
            app_guard.main()
        except KeyboardInterrupt:
            pass

        # authenticate_face was called ONLY ONCE because the transient switch was absorbed by the grace period
        mock_auth_face.assert_called_once()
        mock_failsafe.assert_not_called()

    @patch("app_guard.time.time")
    @patch("app_guard.dismiss_lingering_dialogs")
    @patch("app_guard.unhide_app")
    @patch("app_guard.hide_app")
    @patch("app_guard.elevate_to_foreground")
    @patch("app_guard.authenticate_face")
    @patch("app_guard.ask_password_failsafe")
    @patch("app_guard.get_frontmost_app")
    @patch("app_guard.time.sleep")
    def test_sustained_switch_away_exceeding_grace_period_relocks(
        self,
        mock_sleep,
        mock_get_app,
        mock_failsafe,
        mock_auth_face,
        mock_elevate,
        mock_hide,
        mock_unhide,
        mock_dismiss,
        mock_time,
    ):
        """Switching away to Safari for >= 2.0s confirms departure, relocks, and requires re-auth upon returning."""
        mock_get_app.side_effect = ["Mail", "Safari", "Safari", "Mail"]
        mock_auth_face.return_value = True
        # Time progression:
        # Loop 1 (Mail unlock): 4 calls to time.time() at 100.0
        # Loop 2 (Safari start debounce): 1 call to time.time() at 100.0 (away_start = 100.0)
        # Loop 3 (Safari sustained >2.0s): 1 call to time.time() at 102.5 (2.5s >= 2.0s -> relocks)
        # Loop 4 (Mail re-focus): 1 call to time.time() at 103.0 -> triggers re-authentication
        mock_time.side_effect = [
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            102.5,
            103.0,
            103.0,
            103.0,
            103.0,
        ]
        mock_sleep.side_effect = [None, None, None, None, KeyboardInterrupt]

        try:
            app_guard.main()
        except KeyboardInterrupt:
            pass

        # authenticate_face was called twice: once initially, and once after returning from sustained Safari focus
        self.assertEqual(mock_auth_face.call_count, 2)
        mock_failsafe.assert_not_called()

    def test_cancel_pending_failsafe_timers_cancels_active_timer(self):
        """cancel_pending_failsafe_timers successfully cancels any active Timer threads."""
        import threading
        timer = threading.Timer(60.0, lambda: None)
        timer.start()
        self.assertTrue(timer.is_alive())
        app_guard.cancel_pending_failsafe_timers()
        self.assertFalse(timer.is_alive())

    def test_build_app_and_entitlements_content(self):
        """Verify entitlements.plist and Info.plist configurations."""
        entitlements_path = os.path.join(os.path.dirname(__file__), "..", "entitlements.plist")
        self.assertTrue(os.path.exists(entitlements_path))
        with open(entitlements_path, "r") as f:
            content = f.read()
        self.assertIn("com.apple.security.device.camera", content)

        build_app_path = os.path.join(os.path.dirname(__file__), "..", "build_app.sh")
        with open(build_app_path, "r") as f:
            build_content = f.read()
        self.assertIn("Biometric App Guard requires camera access for facial recognition authentication.", build_content)
        self.assertIn("Hardware requirement.", build_content)
        self.assertIn("--entitlements", build_content)

    def test_singleton_lockfile(self):
        """acquire_lock acquires /tmp/biometric_guard.lock and release_lock cleans it up."""
        # Ensure clean state
        app_guard.release_lock()
        res1 = app_guard.acquire_lock()
        self.assertTrue(res1)
        self.assertTrue(os.path.exists(app_guard.LOCK_FILE_PATH))

        # Re-acquiring in same process or another should succeed or fail appropriately
        app_guard.release_lock()
        self.assertIsNone(app_guard._lock_file)

    @patch("subprocess.run")
    def test_hide_app_uses_finder_not_kill(self, mock_subp):
        """hide_app uses Finder to hide Mail and never calls killall or terminate."""
        app_guard.hide_app("Mail")
        mock_subp.assert_called_once()
        cmd = mock_subp.call_args[0][0]
        self.assertIn('tell application "Finder" to set visible of process "Mail" to false', cmd)
        self.assertNotIn("kill", cmd.lower())
        self.assertNotIn("quit", cmd.lower())


if __name__ == "__main__":
    unittest.main()
