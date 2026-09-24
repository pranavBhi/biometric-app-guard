#!/usr/bin/env python3
"""Diagnostic script to test camera access with hardware warmup via Apple's AVFoundation."""

import sys
import time
import cv2


def test_cameras():
    print("=" * 60)
    print("      macOS Camera Diagnostic Probe (AVFoundation Warmup)")
    print("=" * 60)

    found_working = False

    for idx in [0, 1]:
        print(f"\n[Probe] Testing camera index {idx} with cv2.CAP_AVFOUNDATION...")
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)

        if not cap.isOpened():
            print(f"  [-] Camera index {idx}: Failed to open (isOpened = False).")
            cap.release()
            continue

        print(f"  [+] Camera index {idx}: Device opened. Running hardware warmup loop (up to 2.5s)...")

        # Hardware warmup polling loop (up to 2.5s, 0.1s interval)
        start_time = time.time()
        ret = False
        frame = None
        attempts = 0

        while time.time() - start_time < 2.5:
            attempts += 1
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                break
            time.sleep(0.1)

        elapsed = time.time() - start_time

        if ret and frame is not None and frame.size > 0:
            # Flush initial buffer frames
            for _ in range(4):
                cap.read()

            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  [SUCCESS] Camera {idx} is functional!")
            print(f"            Resolution: {w}x{h}, Channels: {frame.shape[2]}, FPS: {fps:.1f}")
            print(f"            Frame acquired on attempt {attempts} in {elapsed:.2f}s.")
            found_working = True
        else:
            print(f"  [-] Camera index {idx}: Timed out after {elapsed:.2f}s without valid frames (attempts={attempts}).")

        cap.release()

    print("\n" + "=" * 60)
    if found_working:
        print("[RESULT] Camera subsystem verified! Active camera identified.")
        print("=" * 60)
        return 0
    else:
        print("[ERROR] No functional camera stream found on indices 0 or 1.")
        print("[ACTION REQUIRED] Check macOS Camera privacy permissions:")
        print("  1. Open System Settings -> Privacy & Security -> Camera.")
        print("  2. Ensure permission is granted to Terminal, Python, or BiometricGuard.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(test_cameras())
