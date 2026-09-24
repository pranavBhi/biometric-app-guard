"""Liveness detection using MediaPipe 468 Face Mesh landmarks and Eye Aspect Ratio (EAR).

Computes Eye Aspect Ratio (EAR) from facial landmarks and tracks eye closure state
over time to reliably detect natural human blinks for biometric authentication.
"""

from dataclasses import dataclass, field
import math
from typing import Any, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# Standard 468/478 MediaPipe landmark indices for eye contours
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22

__all__ = [
    "is_blinking",
    "calculate_ear",
    "extract_point",
    "euclidean_distance",
    "BlinkDetector",
    "LivenessDetector",
    "LivenessResult",
    "LEFT_EYE_INDICES",
    "RIGHT_EYE_INDICES",
    "EAR_THRESHOLD",
]


@dataclass
class LivenessResult:
    """Dataclass holding liveness detection output for a single video frame."""

    face_detected: bool = False
    ear_avg: float = 0.0
    ear_left: float = 0.0
    ear_right: float = 0.0
    blink_occurred: bool = False
    total_blinks: int = 0
    is_eye_closed: bool = False
    left_eye_coords: List[Tuple[int, int]] = field(default_factory=list)
    right_eye_coords: List[Tuple[int, int]] = field(default_factory=list)
    raw_landmarks: Optional[Any] = None


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return float(dist.euclidean(p1, p2))


def extract_point(
    landmark: Any,
    w: Optional[float] = None,
    h: Optional[float] = None,
) -> Tuple[float, float]:
    """Safely extracts coordinates from each landmark point.

    Supports objects with .x and .y attributes (e.g. MediaPipe landmarks),
    tuples/lists, and dictionaries with 'x' and 'y' keys.
    """
    scale_w = float(w) if w is not None else 1.0
    scale_h = float(h) if h is not None else 1.0

    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        raw_x = landmark.x
        raw_y = landmark.y
    elif isinstance(landmark, (list, tuple)):
        raw_x = landmark[0]
        raw_y = landmark[1]
    elif isinstance(landmark, dict):
        raw_x = landmark["x"]
        raw_y = landmark["y"]
    else:
        try:
            raw_x = landmark[0]
            raw_y = landmark[1]
        except Exception:
            raise ValueError(f"Unknown landmark point format: {type(landmark)}")

    scaled_x = raw_x * scale_w
    scaled_y = raw_y * scale_h

    # Convert to int with round(..., 5) to eliminate float32 precision artifacts (e.g. 24.9999994 -> 25)
    if w is not None and h is not None:
        return (int(round(scaled_x, 5)), int(round(scaled_y, 5)))
    elif scaled_x > 1.0 or scaled_y > 1.0:
        return (int(round(scaled_x, 5)), int(round(scaled_y, 5)))
    else:
        return (float(scaled_x), float(scaled_y))


def calculate_ear(
    landmarks: Any,
    eye_indices: List[int],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> float:
    """Calculates Eye Aspect Ratio (EAR) accepting native MediaPipe protobuf or list/dict.

    Duck-types indexing into landmarks container (such as protobuf
    RepeatedCompositeContainer, NormalizedLandmarkList, list, or dict)
    and computes EAR via NumPy Euclidean distances without division-by-zero errors.
    """
    pts = []

    # Duck-type access into container:
    # If landmarks has a .landmark attribute (e.g. NormalizedLandmarkList), access it;
    # otherwise access landmarks directly (e.g. RepeatedCompositeContainer, list, dict)
    container = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks

    for idx in eye_indices:
        try:
            lm = container[idx]
        except (TypeError, KeyError, IndexError) as e:
            try:
                lm = landmarks[idx]
            except Exception:
                raise ValueError(
                    f"Unsupported landmarks structure or missing index {idx}: {type(landmarks)}"
                ) from e
        pts.append(extract_point(lm, w=image_width, h=image_height))

    pts_arr = np.array(pts, dtype=np.float64)

    # Vertical eye distances:
    # v1 = distance between p2 (pts_arr[1]) and p6 (pts_arr[5])
    # v2 = distance between p3 (pts_arr[2]) and p4 (pts_arr[4])
    v1 = np.linalg.norm(pts_arr[1] - pts_arr[5])
    v2 = np.linalg.norm(pts_arr[2] - pts_arr[4])

    # Horizontal eye distance:
    # horiz = distance between p1 (pts_arr[0]) and p4 (pts_arr[3])
    horiz = np.linalg.norm(pts_arr[0] - pts_arr[3])

    if horiz <= 1e-6:
        return 0.0

    return float((v1 + v2) / (2.0 * horiz))


def is_blinking(
    landmarks: Any,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
    ear_threshold: float = EAR_THRESHOLD,
) -> Tuple[bool, float]:
    """Returns (is_closed_boolean, average_ear_value)."""
    ear_left = calculate_ear(
        landmarks, LEFT_EYE_INDICES, image_width=frame_w, image_height=frame_h
    )
    ear_right = calculate_ear(
        landmarks, RIGHT_EYE_INDICES, image_width=frame_w, image_height=frame_h
    )
    avg_ear = (ear_left + ear_right) / 2.0
    return (avg_ear < ear_threshold), float(avg_ear)


class BlinkDetector:
    """State machine that detects natural blinks from a stream of EAR values."""

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        consec_frames_min: int = 1,
        consec_frames_max: int = 15,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.consec_frames_min = consec_frames_min
        self.consec_frames_max = consec_frames_max
        self.consecutive_closed: int = 0
        self.total_blinks: int = 0
        self.is_closed: bool = False

    def reset(self) -> None:
        self.consecutive_closed = 0
        self.total_blinks = 0
        self.is_closed = False

    def update(self, ear: float) -> Tuple[bool, bool]:
        blink_completed = False
        if ear < self.ear_threshold:
            self.consecutive_closed += 1
            self.is_closed = True
        else:
            if self.is_closed:
                if self.consec_frames_min <= self.consecutive_closed <= self.consec_frames_max:
                    self.total_blinks += 1
                    blink_completed = True
                self.consecutive_closed = 0
                self.is_closed = False
        return blink_completed, self.is_closed


class LivenessDetector:
    """Full liveness detector integrating MediaPipe Face Mesh and EAR blink tracking."""

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        consec_frames_min: int = 1,
        consec_frames_max: int = 15,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.blink_detector = BlinkDetector(
            ear_threshold=ear_threshold,
            consec_frames_min=consec_frames_min,
            consec_frames_max=consec_frames_max,
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        if hasattr(self, "face_mesh") and self.face_mesh is not None:
            self.face_mesh.close()

    def __enter__(self) -> "LivenessDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def reset(self) -> None:
        self.blink_detector.reset()

    def process_frame(self, frame_bgr: np.ndarray) -> LivenessResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return LivenessResult()

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return LivenessResult(
                face_detected=False,
                total_blinks=self.blink_detector.total_blinks,
            )

        landmarks = results.multi_face_landmarks[0]
        ear_left = calculate_ear(landmarks, LEFT_EYE_INDICES, image_width=width, image_height=height)
        ear_right = calculate_ear(landmarks, RIGHT_EYE_INDICES, image_width=width, image_height=height)
        ear_avg = (ear_left + ear_right) / 2.0

        blink_completed, is_closed = self.blink_detector.update(ear_avg)

        def get_coords(indices: List[int]) -> List[Tuple[int, int]]:
            coords = []
            for idx in indices:
                lm = landmarks.landmark[idx]
                coords.append((int(lm.x * width), int(lm.y * height)))
            return coords

        return LivenessResult(
            face_detected=True,
            ear_avg=ear_avg,
            ear_left=ear_left,
            ear_right=ear_right,
            blink_occurred=blink_completed,
            total_blinks=self.blink_detector.total_blinks,
            is_eye_closed=is_closed,
            left_eye_coords=get_coords(LEFT_EYE_INDICES),
            right_eye_coords=get_coords(RIGHT_EYE_INDICES),
            raw_landmarks=landmarks,
        )