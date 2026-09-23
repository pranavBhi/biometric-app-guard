#!/usr/bin/env python3
"""macOS Biometric Application Guard with Passcode Failsafe.

Monitors active frontmost applications on macOS. When a protected application
(e.g., Mail) is focused, it immediately minimizes/hides the app, prompts for
biometric face authentication and natural blink liveness check via webcam.
If biometric verification times out (10s), is cancelled, or fails to recognize the
user, it falls back to a native macOS passcode dialog.
Relocks when focus shifts away.
"""

import os
import subprocess
import sys
import time
from typing import Optional, Tuple
import cv2
import face_recognition
import mediapipe as mp
from utils.liveness import is_blinking

# ================= Configuration =================
PROTECTED_APP = "Mail"  # Change to any app: "Messages", "Notes", "Spotify", etc.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE_PATH = os.path.join(BASE_DIR, "data", "authorized", "me.jpg")
CAMERA_INDEX = 1  # 1 = Built-in FaceTime HD, 0 = Primary/Continuity
FAILSAFE_PASSCODE = os.environ.get("GUARD_PASSCODE", "0713")
TIMEOUT_SECONDS = 10
# =================================================

if not os.path.exists(REF_IMAGE_PATH):
    raise FileNotFoundError(
        f"Missing authorized face profile at {REF_IMAGE_PATH}. "
        f"Run 'python launcher.py --register' or place an image at {REF_IMAGE_PATH}."
    )

print(f"[Guard] Loading face encodings from {REF_IMAGE_PATH}...")
known_img = face_recognition.load_image_file(REF_IMAGE_PATH)
known_encodings = face_recognition.face_encodings(known_img)
if not known_encodings:
    raise ValueError(f"No face detected in reference image '{REF_IMAGE_PATH}'.")
known_encoding = known_encodings[0]

# Initialize MediaPipe Face Mesh
mp_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_frontmost_app() -> str:
    """Returns the name of the currently active/focused application process on macOS."""
    cmd = 'osascript -e \'tell application "System Events" to get name of first application process whose frontmost is true\''
    try:
        result = subprocess.check_output(cmd, shell=True, text=True).strip()
        return result
    except Exception:
        return ""


def hide_app(app_name: str) -> None:
    """Instantly hides the targeted app so the user cannot view or interact with it."""
    cmd = f'osascript -e \'tell application "System Events" to set visible of process "{app_name}" to false\''
    subprocess.run(cmd, shell=True)


def unhide_app(app_name: str) -> None:
    """Brings the protected app back to the front once authenticated."""
    cmd = f'osascript -e \'tell application "{app_name}" to activate\''
    subprocess.run(cmd, shell=True)


def ask_password_failsafe(app_name: str = PROTECTED_APP) -> bool:
    """Displays a native macOS password prompt dialog using AppleScript."""
    apple_script = f'''
    try
        set dialogResult to display dialog "Biometric authentication failed for {app_name}.\\nEnter passcode to unlock:" default answer "" with hidden answer with title "App Guard Passcode Failsafe" with icon caution buttons {{"Cancel", "Unlock"}} default button "Unlock"
        return text returned of dialogResult
    on error number -128
        return ""
    end try
    '''
    try:
        output = subprocess.check_output(
            ["osascript", "-e", apple_script],
            text=True,
        ).strip()
        if output == FAILSAFE_PASSCODE:
            print("[Guard] Passcode correct! Unlocking...")
            return True
        else:
            if output:
                print("[Guard] Incorrect passcode entered.")
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'display notification "Incorrect passcode." with title "App Guard"',
                    ]
                )
            else:
                print("[Guard] Passcode entry cancelled.")
            return False
    except Exception as e:
        print(f"[Guard] Passcode dialog error: {e}", file=sys.stderr)
        return False


def open_camera_with_fallback(
    primary_index: int = 1,
    fallback_index: int = 0,
    timeout: float = 1.5,
) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """Open camera using Apple's native AVFoundation backend with fallback.

    If primary_index fails to open or grab a frame within timeout (1.5 seconds),
    immediately falls back to fallback_index.
    """
    candidates = [primary_index]
    if fallback_index not in candidates:
        candidates.append(fallback_index)

    for idx in candidates:
        print(f"[Guard] Connecting to camera {idx} via AVFoundation (cv2.CAP_AVFOUNDATION)...")
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"[Guard] Camera index {idx} failed to open.")
            cap.release()
            continue

        # Test grabbing a frame within timeout
        start_wait = time.time()
        frame_grabbed = False
        while time.time() - start_wait < timeout:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frame_grabbed = True
                break
            time.sleep(0.05)

        if frame_grabbed:
            print(f"[Guard] Camera index {idx} acquired successfully.")
            return cap, idx
        else:
            print(f"[Guard] Camera index {idx} timed out grabbing frame ({timeout}s). Falling back...")
            cap.release()

    return None, None


def authenticate_face() -> bool:
    """Pops up webcam feed, verifies identity and blink, returns True/False."""
    fallback_idx = 0 if CAMERA_INDEX == 1 else 1
    cap, active_cam = open_camera_with_fallback(CAMERA_INDEX, fallback_idx, timeout=1.5)
    if cap is None:
        print("[Guard] Error: Unable to acquire a functional camera.", file=sys.stderr)
        return False

    blink_consecutive_frames = 0
    verified = False
    start_time = time.time()

    print(f"\n[Guard] Locking {PROTECTED_APP}. Authenticate with your face + blink...")

    try:
        while cap.isOpened():
            # Auto-timeout after 10 seconds if no one is at the desk
            if time.time() - start_time > TIMEOUT_SECONDS:
                print(f"[Guard] Biometric authentication timed out ({TIMEOUT_SECONDS}s).")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Check Liveness (Blink Detection)
            mesh_result = mp_mesh.process(rgb_frame)
            if mesh_result.multi_face_landmarks:
                landmarks = mesh_result.multi_face_landmarks[0].landmark
                closed, ear = is_blinking(landmarks, w, h)

                if closed:
                    blink_consecutive_frames += 1
                else:
                    if blink_consecutive_frames >= 2:
                        # 2. Check Identity with face_recognition (tolerance 0.48)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        if encodings:
                            match = face_recognition.compare_faces(
                                [known_encoding], encodings[0], tolerance=0.48
                            )
                            if match[0]:
                                print("[Guard] Identity & Liveness Verified! Unlocking...")
                                verified = True
                                break
                    blink_consecutive_frames = 0

            # UI feedback window
            display_frame = cv2.flip(frame, 1)
            cv2.putText(
                display_frame,
                f"LOCKED: {PROTECTED_APP}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                display_frame,
                "Blink naturally to unlock",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Biometric App Guard", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Guard] Authentication aborted by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return verified


def main() -> None:
    """Main Sentinel Loop."""
    print(f"[Guard] Running in background. Watching for '{PROTECTED_APP}'...")
    is_unlocked_for_session = False

    try:
        while True:
            front_app = get_frontmost_app()

            if front_app == PROTECTED_APP and not is_unlocked_for_session:
                # 1. Instantly hide the app so nobody reads the screen
                hide_app(PROTECTED_APP)

                # 2. Require Face ID verification
                success = authenticate_face()

                # 3. If biometric fails, timed out, or cancelled, fallback to passcode
                if not success:
                    print(f"[Guard] Biometric check failed or timed out. Triggering passcode failsafe...")
                    success = ask_password_failsafe(PROTECTED_APP)

                if success:
                    # Reveal the app and mark it authenticated for this session
                    unhide_app(PROTECTED_APP)
                    is_unlocked_for_session = True
                else:
                    # Denied or timed out: minimize it again and reset focus
                    hide_app(PROTECTED_APP)
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            'display notification "Authentication Failed" with title "App Guard"',
                        ]
                    )

            # Relock if the user switches away to another app
            elif front_app not in [PROTECTED_APP, "", "Python", "Python3", "Terminal", "iTerm2", "Antigravity", "Electron"]:
                if is_unlocked_for_session:
                    print(f"[Guard] Left {PROTECTED_APP} (active: '{front_app}'). Relocking...")
                    is_unlocked_for_session = False

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[Guard] Guard terminated.")


if __name__ == "__main__":
    main()