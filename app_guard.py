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
from typing import Optional
import cv2
import face_recognition
import mediapipe as mp
from utils.liveness import is_blinking

# ================= Configuration =================
PROTECTED_APP = "Mail"  # Change to any app: "Messages", "Notes", "Spotify", etc.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE_PATH = os.path.join(BASE_DIR, "data", "authorized", "me.jpg")
FAILSAFE_PASSCODE = os.environ.get("GUARD_PASSCODE", "0713")
TIMEOUT_SECONDS = 10
IDLE_TIMEOUT_SECONDS = int(os.environ.get("GUARD_IDLE_TIMEOUT", "300"))  # 5 minutes idle timeout
RELOCK_GRACE_PERIOD = float(os.environ.get("GUARD_RELOCK_GRACE_PERIOD", "2.0"))  # Grace-period debounce duration in seconds
# Camera window visibility: display explicitly by default, suppress only if --headless or GUARD_HEADLESS=1
INTERACTIVE_DISPLAY = (
    not any(arg in sys.argv for arg in ("--headless", "-H"))
    and os.environ.get("GUARD_HEADLESS", "0").lower() not in ("1", "true", "yes")
)

# Internal, transient, or system processes that should NEVER trigger a "Left Mail / Relock" event
IGNORED_SWITCH_APPS = {
    "Python",
    "python3",
    "python",
    "BiometricGuard",
    "SecurityAgent",
    "System Events",
    "osascript",
    "loginwindow",
    "",
    "Dock",
    "SystemUIServer",
    "NotificationCenter",
    "ControlCenter",
    "Spotlight",
    "WindowManager",
    "applet",
    "CoreServicesUIAgent",
}
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


# ================= Logging & Status Tracking =================
LOG_FILE = "/tmp/app_guard.log"

AUTH_REASON_SUCCESS = "SUCCESS"
AUTH_REASON_CAMERA_PERMISSION_DENIED = "CAMERA_PERMISSION_DENIED"
AUTH_REASON_USER_FAILED_VERIFICATION = "USER_FAILED_VERIFICATION"
AUTH_REASON_USER_CANCELLED = "USER_CANCELLED"

LAST_AUTH_FAILURE_REASON = AUTH_REASON_USER_FAILED_VERIFICATION


def log_message(msg: str) -> None:
    """Print log message to stdout and append to /tmp/app_guard.log with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def dismiss_lingering_dialogs() -> None:
    """Cleanly dismisses or kills any lingering AppleScript dialogs (osascript processes)."""
    try:
        subprocess.run(
            ["killall", "osascript"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(0.05)
    except Exception as e:
        log_message(f"[Guard] Note dismissing lingering dialogs: {e}")


def get_frontmost_app() -> str:
    """Returns the name of the currently active/focused application process on macOS."""
    cmd = 'osascript -e \'tell application "System Events" to get name of first application process whose frontmost is true\''
    try:
        result = subprocess.check_output(cmd, shell=True, text=True).strip()
        return result
    except Exception:
        return ""


def is_ignored_switch_app(app_name: str) -> bool:
    """Checks whether an application process is an internal, transient, or system background process
    that should NEVER trigger a "Left <PROTECTED_APP> / Relock" event.
    """
    if not app_name:
        return True
    if app_name in IGNORED_SWITCH_APPS:
        return True
    app_clean = app_name.strip()
    if app_clean in IGNORED_SWITCH_APPS:
        return True
    app_lower = app_clean.lower()
    if app_lower in {s.lower() for s in IGNORED_SWITCH_APPS}:
        return True
    if app_lower.startswith("python"):
        return True
    return False


LOCK_FILE_PATH = "/tmp/biometric_guard.lock"
_lock_file = None


def acquire_lock() -> bool:
    """Acquires a non-blocking exclusive lock on /tmp/biometric_guard.lock.

    Returns True if the lock is successfully acquired, or False if another
    instance of biometric guard is already running.
    """
    global _lock_file
    import fcntl

    try:
        _lock_file = open(LOCK_FILE_PATH, "a+")
        fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_file.seek(0)
        _lock_file.truncate()
        _lock_file.write(f"{os.getpid()}\n")
        _lock_file.flush()
        return True
    except (BlockingIOError, IOError, OSError):
        return False


def release_lock() -> None:
    """Releases the lock on /tmp/biometric_guard.lock."""
    global _lock_file
    import fcntl

    try:
        if _lock_file is not None:
            fcntl.flock(_lock_file.fileno(), fcntl.LOCK_UN)
            _lock_file.close()
            _lock_file = None
            if os.path.exists(LOCK_FILE_PATH):
                try:
                    os.remove(LOCK_FILE_PATH)
                except OSError:
                    pass
    except Exception:
        pass


def hide_app(app_name: str) -> None:
    """Instantly hides the targeted app via Finder so the user cannot view or interact with it."""
    cmd = f'osascript -e \'tell application "Finder" to set visible of process "{app_name}" to false\''
    subprocess.run(cmd, shell=True)


def unhide_app(app_name: str) -> None:
    """Brings the protected app back to the front once authenticated."""
    cmd = f'osascript -e \'tell application "{app_name}" to activate\''
    subprocess.run(cmd, shell=True)


def cancel_pending_failsafe_timers() -> None:
    """Clear any lingering background threads, timer callbacks, or pending failsafe alerts."""
    try:
        import threading
        for thread in threading.enumerate():
            if isinstance(thread, threading.Timer) and thread.is_alive():
                thread.cancel()
                thread.join(timeout=0.05)
    except Exception:
        pass


def elevate_to_foreground() -> None:
    """Elevate the current process to the frontmost GUI application.

    Sets NSApplicationActivationPolicyRegular (0) on Cocoa NSApplication and calls
    activateIgnoringOtherApps(True) via ctypes and AppleScript, ensuring macOS
    WindowServer, AVFoundation, and TCC recognize the process as the active
    frontmost GUI app without spawning child processes.
    """
    log_message("[Guard] [FOREGROUND_HANDSHAKE] Elevating biometric verification process to active foreground...")

    # 1. Cocoa NSApplication setActivationPolicy and activateIgnoringOtherApps via ctypes
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))

        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]

        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        ns_app_cls = objc.objc_getClass(b"NSApplication")
        shared_app_sel = objc.sel_registerName(b"sharedApplication")
        app_instance = objc.objc_msgSend(ns_app_cls, shared_app_sel)

        # Set activation policy to NSApplicationActivationPolicyRegular (0)
        set_policy_sel = objc.sel_registerName(b"setActivationPolicy:")
        set_policy_fn = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long),
        )
        policy_res = set_policy_fn(app_instance, set_policy_sel, 0)
        log_message(f"[Guard] [FOREGROUND_HANDSHAKE] NSApplicationActivationPolicyRegular configured (res={policy_res}).")

        # Activate application ignoring other apps
        activate_sel = objc.sel_registerName(b"activateIgnoringOtherApps:")
        activate_fn = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool),
        )
        activate_fn(app_instance, activate_sel, True)
    except Exception as e:
        log_message(f"[Guard] [FOREGROUND_HANDSHAKE] Cocoa elevation note: {e}")

    # 2. AppleScript activate on current PID
    try:
        pid = os.getpid()
        apple_script = f'''
        tell application "System Events"
            set procList to every process whose unix id is {pid}
            if (count of procList) > 0 then
                set frontmost of (item 1 of procList) to true
            end if
        end tell
        '''
        subprocess.run(["osascript", "-e", apple_script], check=False)
    except Exception as e:
        log_message(f"[Guard] [FOREGROUND_HANDSHAKE] AppleScript elevation note: {e}")

    # 3. 0.3s delay allowing macOS WindowServer and AVFoundation session to bind to the display
    log_message("[Guard] [FOREGROUND_HANDSHAKE] Waiting 0.3s for WindowServer and AVFoundation display binding...")
    time.sleep(0.3)


def ask_password_failsafe(app_name: str = PROTECTED_APP, reason: str = "") -> bool:
    """Displays a native macOS password prompt dialog using AppleScript."""
    detail_msg = reason if reason else f"Biometric authentication failed for {app_name}."
    apple_script = f'''
    try
        set dialogResult to display dialog "{detail_msg}\\nEnter passcode to unlock:" default answer "" with hidden answer with title "App Guard Passcode Failsafe" with icon caution buttons {{"Cancel", "Unlock"}} default button "Unlock"
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
            log_message("[Guard] Passcode correct! Unlocking...")
            return True
        else:
            if output:
                log_message("[Guard] Incorrect passcode entered.")
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'display notification "Incorrect passcode." with title "App Guard"',
                    ]
                )
            else:
                log_message("[Guard] Passcode entry cancelled.")
            return False
    except Exception as e:
        log_message(f"[Guard] Passcode dialog error: {e}")
        return False


def get_working_camera(timeout_per_index: float = 2.5) -> Optional[cv2.VideoCapture]:
    """Robust camera discovery iterating through indices [0, 1] using AVFoundation.

    For each camera index in [0, 1]:
      - Initializes using AVFoundation: cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
      - Polls cap.isOpened() for up to 1.5s with time.sleep(0.05) to allow hardware power-on
      - Once open, polls ret, frame = cap.read() for the remaining timeout window
      - Ensures cv2.VideoCapture does not abort if initial frame read fails during app switch
      - As soon as a valid frame is captured, flushes 3-5 buffer frames to clear
        exposure/white-balance lag and returns the active cap object
      - Only declares failure after the full probe window (at least 1.5s) without success

    Returns None if all indices fail to yield a usable video frame.
    """
    global LAST_AUTH_FAILURE_REASON

    probe_timeout = max(1.5, timeout_per_index)

    for idx in [0, 1]:
        log_message(f"[Guard] [CAMERA_AUTH_STATUS: PROBING] Testing camera index {idx} with AVFoundation (timeout={probe_timeout}s)...")
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)

        # Allow up to 1.5 seconds with 0.05s intervals for hardware initialization / isOpened()
        open_start = time.time()
        while time.time() - open_start < 1.5:
            if cap.isOpened():
                break
            time.sleep(0.05)

        if not cap.isOpened():
            log_message(f"[Guard] [CAMERA_AUTH_STATUS: PROBING] Camera index {idx}: isOpened remained False after 1.5s probe.")
            cap.release()
            continue

        log_message(f"[Guard] [CAMERA_AUTH_STATUS: AUTHORIZED] Camera {idx} handle opened. Polling frames (with warmup retry)...")

        # Hardware warmup & polling loop (up to probe_timeout seconds total)
        start_time = time.time()
        frame_acquired = False
        attempts = 0

        while time.time() - start_time < probe_timeout:
            if not cap.isOpened():
                break

            attempts += 1
            ret, frame = cap.read()
            # Do NOT immediately abort if first frame read fails during app switch
            if ret and frame is not None and frame.size > 0:
                frame_acquired = True
                break

            time.sleep(0.1)

        if frame_acquired:
            # Flush initial buffer frames to clear exposure/white-balance lag
            for _ in range(4):
                cap.read()

            h, w = frame.shape[:2]
            elapsed = time.time() - start_time
            log_message(f"[Guard] [CAMERA_AUTH_STATUS: STREAMING] Acquired functional video stream on index {idx} in {elapsed:.2f}s (attempt {attempts}, {w}x{h}).")
            return cap
        else:
            log_message(f"[Guard] [CAMERA_AUTH_STATUS: PROBING] Camera index {idx} timed out after {probe_timeout}s without valid frames (attempts={attempts}). Releasing...")
            cap.release()

    error_msg = (
        "[Guard] [CAMERA_AUTH_STATUS: DENIED] No functional camera stream found on indices 0 or 1. "
        "Check macOS Camera privacy permissions."
    )
    log_message(error_msg)
    LAST_AUTH_FAILURE_REASON = AUTH_REASON_CAMERA_PERMISSION_DENIED
    return None


def authenticate_face(interactive_display: Optional[bool] = None) -> bool:
    """Verifies identity and natural blink liveness via webcam.

    Displays an explicit camera preview window with status overlay ("Blink to verify..."),
    brief green confirmation banner ("Verified!") on success, and cleanly closes resources.

    Parameters:
        interactive_display: If True or None, renders OpenCV preview window ("Biometric Authentication").
                             If False, runs headless verification reading frames without cv2.imshow/waitKey.

    Returns:
        bool: Strictly True upon verified face match AND blink, False otherwise.
    """
    global LAST_AUTH_FAILURE_REASON

    show_gui = INTERACTIVE_DISPLAY if interactive_display is None else bool(interactive_display)

    cap = get_working_camera(timeout_per_index=2.5)

    if cap is None:
        log_message("[Guard] Direct camera stream unavailable. Check camera privacy permissions.")
        LAST_AUTH_FAILURE_REASON = AUTH_REASON_CAMERA_PERMISSION_DENIED
        return False

    window_name = "Biometric Authentication"
    if show_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    blink_consecutive_frames = 0
    blink_confirmed = False
    is_face_match = False
    face_distance: Optional[float] = None
    verified = False
    start_time = time.time()
    frame_counter = 0

    log_message(f"[Guard] Locking {PROTECTED_APP}. Authenticate with your face + blink (interactive={show_gui})...")

    try:
        while cap.isOpened():
            # Auto-timeout after TIMEOUT_SECONDS (10s)
            if time.time() - start_time > TIMEOUT_SECONDS:
                log_message(f"[Guard] Biometric authentication timed out ({TIMEOUT_SECONDS}s).")
                LAST_AUTH_FAILURE_REASON = AUTH_REASON_USER_FAILED_VERIFICATION
                break

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            frame_counter += 1
            h, w = frame.shape[:2]

            # 1. Process Liveness (Blink Detection) on every frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_result = mp_mesh.process(rgb_frame)

            if mesh_result.multi_face_landmarks:
                landmarks = mesh_result.multi_face_landmarks[0].landmark
                closed, ear = is_blinking(landmarks, w, h)

                if closed:
                    blink_consecutive_frames += 1
                else:
                    if blink_consecutive_frames >= 2:
                        blink_confirmed = True
                        log_message(f"[Guard] Blink verified! (EAR: {ear:.3f})")
                    blink_consecutive_frames = 0

            # 2. Check Identity on downscaled frame (0.5x) every 2 frames for fast 30+ FPS
            if frame_counter % 2 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small)

                if face_locations:
                    encodings = face_recognition.face_encodings(rgb_small, face_locations)
                    if encodings:
                        distances = face_recognition.face_distance([known_encoding], encodings[0])
                        face_distance = float(distances[0])
                        is_face_match = face_distance <= 0.55
                    else:
                        is_face_match = False
                else:
                    is_face_match = False

            # 3. Success condition: BOTH face match AND natural blink confirmed
            if is_face_match and blink_confirmed:
                log_message(f"[Guard] Identity match confirmed AND natural blink verified! Unlocking {PROTECTED_APP}...")
                verified = True
                if show_gui:
                    display_frame = cv2.flip(frame, 1)
                    # Brief green confirmation banner ("Verified!") for 0.4 seconds
                    cv2.rectangle(display_frame, (0, h - 65), (w, h), (0, 180, 0), -1)
                    cv2.putText(
                        display_frame,
                        "Verified!",
                        (max(15, int(w / 2) - 65), h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow(window_name, display_frame)
                    cv2.waitKey(400)
                break

            # 4. Interactive UI feedback or headless sleep
            if show_gui:
                display_frame = cv2.flip(frame, 1)

                # Header background
                cv2.rectangle(display_frame, (0, 0), (w, 75), (20, 20, 20), -1)

                # Target app label
                cv2.putText(
                    display_frame,
                    f"LOCKED: {PROTECTED_APP}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Status indicator
                if is_face_match:
                    status_txt = "Face matched - Blink to verify..." if not blink_confirmed else "Verified!"
                    status_color = (0, 255, 255) if not blink_confirmed else (0, 255, 0)
                else:
                    dist_txt = f" ({face_distance:.2f})" if face_distance is not None else ""
                    status_txt = f"Looking for authorized user{dist_txt}..."
                    status_color = (200, 200, 200)

                cv2.putText(
                    display_frame,
                    status_txt,
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                )

                cv2.imshow(window_name, display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log_message("[Guard] Authentication aborted by user ('q').")
                    LAST_AUTH_FAILURE_REASON = AUTH_REASON_USER_CANCELLED
                    break
            else:
                # Headless verification path: yield CPU briefly without GUI calls
                time.sleep(0.01)

    finally:
        if cap is not None:
            cap.release()
        if show_gui:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    if not verified and LAST_AUTH_FAILURE_REASON not in (AUTH_REASON_USER_CANCELLED, AUTH_REASON_CAMERA_PERMISSION_DENIED):
        LAST_AUTH_FAILURE_REASON = AUTH_REASON_USER_FAILED_VERIFICATION

    return bool(verified)


def main() -> None:
    """Main Sentinel Loop."""
    if not acquire_lock():
        log_message(f"[Guard] Another instance is already running (locked via {LOCK_FILE_PATH}). Exiting.")
        return

    log_message(
        f"[Guard] Running in background. Watching for '{PROTECTED_APP}' "
        f"(idle_timeout={IDLE_TIMEOUT_SECONDS}s, relock_grace_period={RELOCK_GRACE_PERIOD}s)..."
    )
    session_unlocked = False
    unlocked_session_active = False
    last_unlocked_timestamp = 0.0
    last_active_timestamp = 0.0
    away_start_timestamp: Optional[float] = None

    try:
        while True:
            front_app = get_frontmost_app()
            now = time.time()

            # 1. Filter Out Internal / Transient Processes in Frontmost App Detection:
            # If the frontmost app is in IGNORED_SWITCH_APPS, completely ignore the switch
            # and keep the existing session state intact.
            if is_ignored_switch_app(front_app):
                time.sleep(0.5)
                continue

            if front_app == PROTECTED_APP:
                # If focus returned to protected app, cancel any pending away debounce
                if away_start_timestamp is not None:
                    log_message(
                        f"[Guard] Focus returned to {PROTECTED_APP} within grace period. Cancelling relock."
                    )
                    away_start_timestamp = None

                if session_unlocked or unlocked_session_active:
                    # Check idle duration (e.g. 5 minutes)
                    if last_active_timestamp > 0 and (now - last_active_timestamp > IDLE_TIMEOUT_SECONDS):
                        log_message(
                            f"[Guard] Session idle for >{IDLE_TIMEOUT_SECONDS}s. Relocking {PROTECTED_APP}..."
                        )
                        session_unlocked = False
                        unlocked_session_active = False
                        cancel_pending_failsafe_timers()
                        # Fall through to lock and re-authenticate
                    else:
                        # Session remains authenticated and active; maintain unlock without re-prompting
                        last_active_timestamp = now
                        time.sleep(0.5)
                        continue

                log_message(f"[Guard] Transition: Detected frontmost protected app '{PROTECTED_APP}'.")
                # 0. Cleanly dismiss any lingering dialogs and pending timers from previous cycles
                cancel_pending_failsafe_timers()
                dismiss_lingering_dialogs()

                # 1. Instantly hide the app so nobody reads the screen
                hide_app(PROTECTED_APP)

                # 2. Foreground Window & TCC Context Handshake:
                elevate_to_foreground()

                # 3. Require Face ID verification strictly synchronously
                auth_success = authenticate_face()

                # 4. ONLY invoke passcode failsafe if biometric authentication failed AND session not unlocked
                if not auth_success and not session_unlocked:
                    cancel_pending_failsafe_timers()
                    if LAST_AUTH_FAILURE_REASON == AUTH_REASON_CAMERA_PERMISSION_DENIED:
                        log_message("[Guard] Failsafe trigger reason: CAMERA_PERMISSION_DENIED. Prompting passcode...")
                        fail_reason = "Camera permission denied or camera unavailable."
                    elif LAST_AUTH_FAILURE_REASON == AUTH_REASON_USER_CANCELLED:
                        log_message("[Guard] Failsafe trigger reason: USER_CANCELLED. Prompting passcode...")
                        fail_reason = "Biometric authentication cancelled."
                    else:
                        log_message("[Guard] Failsafe trigger reason: USER_FAILED_VERIFICATION. Prompting passcode...")
                        fail_reason = f"Biometric verification failed for {PROTECTED_APP}."

                    failsafe_result = ask_password_failsafe(PROTECTED_APP, reason=fail_reason)
                    dismiss_lingering_dialogs()
                    auth_success = failsafe_result

                # 5. Handle result
                if auth_success:
                    log_message(f"[Guard] Access granted to {PROTECTED_APP}. Session unlocked.")
                    # The unlock routine should ONLY bring Mail to the front:
                    # osascript -e 'tell application "Mail" to activate'
                    # Ensure it NEVER invokes quit, killall, terminate(), or close on the target application process.
                    cancel_pending_failsafe_timers()
                    unhide_app(PROTECTED_APP)
                    session_unlocked = True
                    unlocked_session_active = True
                    away_start_timestamp = None
                    last_unlocked_timestamp = time.time()
                    last_active_timestamp = time.time()
                    # Transition the watcher to a passive polling state. DO NOT call ask_password_failsafe() or exit the process.
                    time.sleep(1.0)
                    last_active_timestamp = time.time()
                else:
                    log_message(f"[Guard] Access denied. Keeping {PROTECTED_APP} locked.")
                    session_unlocked = False
                    unlocked_session_active = False
                    away_start_timestamp = None
                    cancel_pending_failsafe_timers()
                    dismiss_lingering_dialogs()
                    hide_app(PROTECTED_APP)
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            'display notification "Authentication Failed" with title "App Guard"',
                        ]
                    )
                    time.sleep(1.0)

            elif session_unlocked or unlocked_session_active:
                # Only consider the user as having "left Mail" if the frontmost app changes
                # to a legitimate third-party application (e.g. Finder, Safari, Chrome, Slack, etc.).
                # Grace-Period Debounce: verify that the frontmost app remains NOT Mail
                # for at least RELOCK_GRACE_PERIOD consecutive seconds before relocking.
                if away_start_timestamp is None:
                    away_start_timestamp = now
                    log_message(
                        f"[Guard] Detected switch away from {PROTECTED_APP} to '{front_app}'. "
                        f"Starting {RELOCK_GRACE_PERIOD}s relock debounce..."
                    )
                elif (now - away_start_timestamp) >= RELOCK_GRACE_PERIOD:
                    elapsed = now - away_start_timestamp
                    log_message(
                        f"[Guard] Left {PROTECTED_APP} (confirmed frontmost: '{front_app}' for {elapsed:.1f}s >= {RELOCK_GRACE_PERIOD}s). Relocking session..."
                    )
                    cancel_pending_failsafe_timers()
                    dismiss_lingering_dialogs()
                    session_unlocked = False
                    unlocked_session_active = False
                    away_start_timestamp = None
                    last_unlocked_timestamp = 0.0
                    last_active_timestamp = 0.0
                else:
                    # Debounce grace period in progress; maintain unlocked state
                    pass
            else:
                away_start_timestamp = None

            time.sleep(0.5)

    except KeyboardInterrupt:
        log_message("[Guard] Guard terminated by KeyboardInterrupt.")
    finally:
        release_lock()


if __name__ == "__main__":
    main()