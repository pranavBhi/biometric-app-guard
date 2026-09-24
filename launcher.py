#!/usr/bin/env python3
"""Biometric Application Launcher for macOS.

Captures webcam frames using OpenCV, performs facial recognition verification
against an authorized user image (data/authorized/me.jpg), verifies liveness
via natural eye blink detection (EAR with MediaPipe 468 Face Mesh), and launches
the specified macOS application upon successful authentication.
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Optional, Tuple
import cv2
import face_recognition
import numpy as np

from utils.liveness import LivenessDetector, LivenessResult


DEFAULT_APP = os.environ.get("TARGET_APP", "Google Chrome")
DEFAULT_AUTH_IMAGE = "data/authorized/me.jpg"
DEFAULT_TOLERANCE = 0.55
DEFAULT_EAR_THRESHOLD = 0.21


def load_authorized_encoding(image_path: str) -> Optional[np.ndarray]:
    """Load authorized image and compute 128D face encoding."""
    if not os.path.exists(image_path):
        return None

    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"[ERROR] No face detected in authorized image '{image_path}'.", file=sys.stderr)
            return None
        return encodings[0]
    except Exception as e:
        print(f"[ERROR] Failed to load authorized image: {e}", file=sys.stderr)
        return None


def register_user(image_path: str, camera_id: int = 0) -> bool:
    """Capture a reference photo from webcam and save as authorized user."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera {camera_id} for registration.", file=sys.stderr)
        return False

    print("\n--- BIOMETRIC ENROLLMENT ---")
    print(f"Align your face in the center of the camera.")
    print("Press SPACE to capture and save, or 'q' to cancel.")

    saved = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            display = frame.copy()

            # Center guideline box
            cv2.rectangle(
                display,
                (int(w * 0.25), int(h * 0.15)),
                (int(w * 0.75), int(h * 0.85)),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "Press SPACE to capture authorized photo",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Biometric Registration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                # Validate face presence
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                if encodings:
                    cv2.imwrite(image_path, frame)
                    print(f"[SUCCESS] Reference photo saved to '{image_path}'.")
                    saved = True
                    break
                else:
                    print("[WARNING] No face detected. Please position yourself clearly and try again.")
            elif key == ord("q"):
                print("[INFO] Registration cancelled.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved


def launch_application(target_app: str) -> bool:
    """Execute macOS open command to launch target application."""
    print(f"\n[SUCCESS] Authentication verified! Launching '{target_app}'...")
    try:
        subprocess.run(["open", "-a", target_app], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to launch application '{target_app}': {e}", file=sys.stderr)
        return False


def draw_hud(
    frame: np.ndarray,
    target_app: str,
    is_match: bool,
    distance: Optional[float],
    liveness: LivenessResult,
    blink_confirmed: bool,
    success: bool,
) -> np.ndarray:
    """Draw a clean, informative HUD overlay on the webcam frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Draw semi-transparent header panel
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Title & Target Application
    cv2.putText(
        frame,
        f"Biometric Launcher -> Target: {target_app}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # Identity status badge
    if is_match:
        dist_str = f"({distance:.2f})" if distance is not None else ""
        cv2.putText(
            frame,
            f"Face Match: CONFIRMED {dist_str}",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    else:
        status = "UNKNOWN / SEARCHING" if distance is None else f"NO MATCH ({distance:.2f})"
        cv2.putText(
            frame,
            f"Face Match: {status}",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if distance is not None else (0, 200, 255),
            2,
        )

    # Liveness / Blink status badge
    ear_str = f"EAR: {liveness.ear_avg:.2f}" if liveness.face_detected else "EAR: --"
    if blink_confirmed:
        cv2.putText(
            frame,
            f"Liveness: BLINK CONFIRMED ({ear_str})",
            (15, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    else:
        blink_prompt = "PLEASE BLINK NATURALLY" if is_match else "WAITING"
        cv2.putText(
            frame,
            f"Liveness: {blink_prompt} ({ear_str})",
            (15, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255) if is_match else (180, 180, 180),
            2,
        )

    # Draw eye contour points
    for pt in liveness.left_eye_coords + liveness.right_eye_coords:
        cv2.circle(frame, pt, 2, (0, 255, 255), -1)

    # Large verification success banner
    if success:
        cv2.rectangle(frame, (int(w * 0.1), int(h * 0.4)), (int(w * 0.9), int(h * 0.6)), (0, 180, 0), -1)
        cv2.putText(
            frame,
            "VERIFIED! UNLOCKING...",
            (int(w * 0.18), int(h * 0.52)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )

    # Instructions at bottom
    cv2.putText(
        frame,
        "Press 'q' to quit",
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    return frame


def ask_password_failsafe(target_app: str = DEFAULT_APP) -> bool:
    """Displays a native macOS password prompt dialog using AppleScript."""
    apple_script = f'''
    try
        set dialogResult to display dialog "Biometric authentication failed for {target_app}.\\nEnter passcode to unlock:" default answer "" with hidden answer with title "Biometric Launcher Passcode Failsafe" with icon caution buttons {{"Cancel", "Unlock"}} default button "Unlock"
        return text returned of dialogResult
    on error number -128
        return ""
    end try
    '''
    failsafe_passcode = os.environ.get("GUARD_PASSCODE", "0713")
    try:
        output = subprocess.check_output(
            ["osascript", "-e", apple_script],
            text=True,
        ).strip()
        if output == failsafe_passcode:
            print("[Launcher] Passcode correct! Unlocking...")
            return True
        else:
            if output:
                print("[Launcher] Incorrect passcode entered.")
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'display notification "Incorrect passcode." with title "Biometric Launcher"',
                    ]
                )
            else:
                print("[Launcher] Passcode entry cancelled.")
            return False
    except Exception as e:
        print(f"[Launcher] Passcode dialog error: {e}", file=sys.stderr)
        return False


def authenticate_face(
    target_app: str = DEFAULT_APP,
    authorized_image_path: str = DEFAULT_AUTH_IMAGE,
    tolerance: float = DEFAULT_TOLERANCE,
    ear_threshold: float = DEFAULT_EAR_THRESHOLD,
    camera_id: int = 0,
    timeout: float = 60.0,
    headless: bool = False,
) -> bool:
    """Verifies user identity and natural blink liveness via webcam.

    Returns:
        bool: Strictly True upon successful face match AND blink verification,
              False on timeout, cancellation, or failure.
    """
    print("[INFO] Loading authorized biometric profile...")
    auth_encoding = load_authorized_encoding(authorized_image_path)
    if auth_encoding is None:
        return False
    print("[INFO] Authorized profile loaded successfully.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not access camera (id={camera_id}).", file=sys.stderr)
        return False

    detector = LivenessDetector(ear_threshold=ear_threshold)

    start_time = time.time()
    frame_counter = 0
    is_face_match = False
    face_distance: Optional[float] = None
    blink_confirmed = False

    print("\n[INFO] Starting camera feed. Look into camera and blink to authenticate...")

    try:
        while True:
            if timeout > 0 and (time.time() - start_time) > timeout:
                print(f"[TIMEOUT] Authentication timed out after {timeout} seconds.", file=sys.stderr)
                return False

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_counter += 1

            # 1. Process Liveness & EAR on every frame
            liveness = detector.process_frame(frame)

            # 2. Perform Face Recognition periodically (e.g. every 3 frames for high responsiveness)
            if frame_counter % 3 == 0:
                # Downsample 1/2 for fast feature extraction
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small)

                if face_locations:
                    encodings = face_recognition.face_encodings(rgb_small, face_locations)
                    if encodings:
                        distances = face_recognition.face_distance([auth_encoding], encodings[0])
                        face_distance = float(distances[0])
                        is_face_match = face_distance <= tolerance
                    else:
                        is_face_match = False
                        face_distance = None
                else:
                    is_face_match = False
                    face_distance = None

            # 3. Check for blink completion
            if liveness.blink_occurred:
                if is_face_match:
                    blink_confirmed = True
                    print(f"[LIVENESS] Natural blink confirmed! (EAR: {liveness.ear_avg:.3f})")

            # 4. Check if both criteria met
            if is_face_match and blink_confirmed:
                print("\n[MATCH] Identity match confirmed AND natural blink verified!")

                if not headless:
                    # Show success frame briefly
                    display_frame = draw_hud(
                        frame,
                        target_app,
                        is_face_match,
                        face_distance,
                        liveness,
                        blink_confirmed,
                        success=True,
                    )
                    cv2.imshow("macOS Biometric Launcher", display_frame)
                    cv2.waitKey(500)

                # Release all camera resources immediately upon match
                cap.release()
                detector.close()
                if not headless:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)

                return True

            # 5. Render GUI if not headless
            if not headless:
                display_frame = draw_hud(
                    frame,
                    target_app,
                    is_face_match,
                    face_distance,
                    liveness,
                    blink_confirmed,
                    success=False,
                )
                cv2.imshow("macOS Biometric Launcher", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Authentication cancelled by user.")
                    return False
            else:
                # Terminal log in headless mode
                if frame_counter % 30 == 0:
                    status_str = f"Match={is_face_match}, EAR={liveness.ear_avg:.2f}, Blink={blink_confirmed}"
                    print(f"[STATUS] {status_str}")

    finally:
        # Guarantee resources are cleaned up if exited without return True
        cap.release()
        detector.close()
        if not headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    return False


def run_launcher(
    target_app: str = DEFAULT_APP,
    authorized_image_path: str = DEFAULT_AUTH_IMAGE,
    tolerance: float = DEFAULT_TOLERANCE,
    ear_threshold: float = DEFAULT_EAR_THRESHOLD,
    camera_id: int = 0,
    timeout: float = 60.0,
    headless: bool = False,
    dry_run: bool = False,
    enable_failsafe: bool = False,
) -> int:
    """Main biometric launcher loop."""
    print("=" * 60)
    print("           macOS Biometric Application Launcher")
    print("=" * 60)
    print(f"  Target Application : {target_app}")
    print(f"  Authorized Photo   : {authorized_image_path}")
    print(f"  Face Tolerance     : {tolerance}")
    print(f"  EAR Threshold      : {ear_threshold}")
    print("=" * 60)

    # Verify authorized image exists
    if not os.path.exists(authorized_image_path):
        print(f"\n[ERROR] Authorized reference image not found at '{authorized_image_path}'.", file=sys.stderr)
        print("Please place your reference photo at data/authorized/me.jpg, or enroll using:", file=sys.stderr)
        print("    python launcher.py --register\n", file=sys.stderr)
        return 1

    auth_success = authenticate_face(
        target_app=target_app,
        authorized_image_path=authorized_image_path,
        tolerance=tolerance,
        ear_threshold=ear_threshold,
        camera_id=camera_id,
        timeout=timeout,
        headless=headless,
    )

    if auth_success:
        # Camera resources are already released by authenticate_face()
        if not dry_run:
            launch_application(target_app)
        else:
            print(f"[DRY-RUN] Would execute: open -a '{target_app}'")
        return 0

    # Guarded failsafe invocation: ONLY executed if authenticate_face() evaluates to False
    if enable_failsafe:
        print("[Launcher] Biometric check failed or timed out. Triggering passcode failsafe...")
        failsafe_success = ask_password_failsafe(target_app)
        if failsafe_success:
            if not dry_run:
                launch_application(target_app)
            else:
                print(f"[DRY-RUN] Would execute: open -a '{target_app}'")
            return 0

    return 2 if timeout > 0 else 1


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="macOS Biometric Application Launcher using Face Recognition and EAR Blink Liveness."
    )
    parser.add_argument(
        "-a",
        "--app",
        type=str,
        default=DEFAULT_APP,
        help=f"Target macOS application to launch upon match (default: '{DEFAULT_APP}').",
    )
    parser.add_argument(
        "-i",
        "--authorized-image",
        type=str,
        default=DEFAULT_AUTH_IMAGE,
        help=f"Path to reference image of authorized user (default: '{DEFAULT_AUTH_IMAGE}').",
    )
    parser.add_argument(
        "-r",
        "--register",
        action="store_true",
        help="Register/enroll authorized face photo via webcam before launching.",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Face recognition Euclidean distance threshold (default: {DEFAULT_TOLERANCE}).",
    )
    parser.add_argument(
        "-e",
        "--ear-threshold",
        type=float,
        default=DEFAULT_EAR_THRESHOLD,
        help=f"Eye Aspect Ratio cutoff for blink closure (default: {DEFAULT_EAR_THRESHOLD}).",
    )
    parser.add_argument(
        "-c",
        "--camera-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds to wait for verification before exiting (default: 60s, 0=infinite).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without GUI window.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify credentials and blink without actually launching the application.",
    )
    parser.add_argument(
        "--failsafe",
        action="store_true",
        help="Enable passcode failsafe dialog if biometric verification fails or times out.",
    )
    return parser.parse_args(args)


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    if args.register:
        success = register_user(args.authorized_image, camera_id=args.camera_id)
        if not success:
            sys.exit(1)
        print("[INFO] Registration complete. Proceeding to launcher...\n")

    code = run_launcher(
        target_app=args.app,
        authorized_image_path=args.authorized_image,
        tolerance=args.tolerance,
        ear_threshold=args.ear_threshold,
        camera_id=args.camera_id,
        timeout=args.timeout,
        headless=args.headless,
        dry_run=args.dry_run,
        enable_failsafe=args.failsafe,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
