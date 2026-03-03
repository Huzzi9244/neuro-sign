"""
Neuro-Sign | Phase 1: Data Collector
=====================================
Uses the NEW MediaPipe Tasks API (HandLandmarker) — compatible with
mediapipe >= 0.10.31 which removed the old `mp.solutions` interface.

Requires:
    hand_landmarker.task  →  save to:
        backend/ml_pipeline/models/hand_landmarker.task

    Download from:
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Folder output:
    data/
    └── <GestureName>/
        ├── sequence_0.npy   shape: (30, 126)
        ├── sequence_1.npy
        └── ...

Controls:
    G  → Set / change the current gesture label  (typed in terminal)
    R  → 3-second countdown, then record 30 frames
    Q  → Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

from mediapipe.tasks            import python           as mp_python       # noqa: F401
from mediapipe.tasks.python     import vision           as mp_vision
from mediapipe.tasks.python.core import base_options    as base_options_module

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(_SCRIPT_DIR, "models", "hand_landmarker.task")
DATA_DIR        = os.path.join(_SCRIPT_DIR, "data")

SEQUENCE_LENGTH = 30        # frames per recorded sequence
NUM_HANDS       = 2
CAM_INDEX       = 0
CAM_WIDTH       = 1280
CAM_HEIGHT      = 720

# Drawing colours (BGR)
COLOR_LANDMARK   = (0, 255, 128)
COLOR_CONNECTION = (255, 255, 255)
COLOR_DOT_REC    = (0,   0,   255)

# MediaPipe 21-landmark hand skeleton connection pairs
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),            # thumb
    (0,5),(5,6),(6,7),(7,8),            # index
    (5,9),(9,10),(10,11),(11,12),       # middle
    (9,13),(13,14),(14,15),(15,16),     # ring
    (13,17),(17,18),(18,19),(19,20),    # pinky
    (0,17),                             # palm base
]

# ──────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE TASKS SETUP
# ──────────────────────────────────────────────────────────────────────────────
def build_landmarker(model_path: str) -> mp_vision.HandLandmarker:
    """Creates a HandLandmarker in VIDEO mode (frame-by-frame, synchronous)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"\n[ERROR] Model file not found:\n  {model_path}\n\n"
            "  ► Download it from:\n"
            "    https://storage.googleapis.com/mediapipe-models/hand_landmarker"
            "/hand_landmarker/float16/1/hand_landmarker.task\n\n"
            "  ► Save it to:\n"
            "    backend/ml_pipeline/models/hand_landmarker.task\n"
        )

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ──────────────────────────────────────────────────────────────────────────────
# KEYPOINT EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def extract_keypoints(detection_result) -> np.ndarray:
    """
    Returns a flat ndarray of shape (126,):
        2 hands × 21 landmarks × (x, y, z)
    Absent hands are represented as zeros.
    """
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if detection_result is None:
        return np.concatenate([left, right])

    for hand_landmarks, handedness in zip(
        detection_result.hand_landmarks,
        detection_result.handedness,
    ):
        label  = handedness[0].category_name           # "Left" or "Right"
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        ).flatten()

        if label == "Left":
            left = coords
        else:
            right = coords

    return np.concatenate([left, right])


# ──────────────────────────────────────────────────────────────────────────────
# LANDMARK DRAWING  (manual — mp.solutions.drawing_utils no longer exists)
# ──────────────────────────────────────────────────────────────────────────────
def draw_landmarks_on_frame(frame, detection_result) -> None:
    """Draws hand skeleton directly onto the frame (in-place)."""
    if detection_result is None or not detection_result.hand_landmarks:
        return

    h, w = frame.shape[:2]

    for hand_landmarks in detection_result.hand_landmarks:
        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]
        for start_idx, end_idx in _HAND_CONNECTIONS:
            cv2.line(frame, pts[start_idx], pts[end_idx],
                     COLOR_CONNECTION, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 4, COLOR_LANDMARK, -1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# HUD OVERLAY
# ──────────────────────────────────────────────────────────────────────────────
def draw_ui(frame, gesture: str, state: str, countdown: int,
            seq_idx: int, frame_idx: int) -> None:
    h, w = frame.shape[:2]

    # top banner
    cv2.rectangle(frame, (0, 0), (w, 56), (20, 20, 20), -1)
    gesture_text = f"Gesture: {gesture if gesture else 'None  (press G to set)'}"
    cv2.putText(frame, gesture_text, (12, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

    # bottom status bar
    cv2.rectangle(frame, (0, h - 52), (w, h), (20, 20, 20), -1)
    state_colors = {
        "IDLE":      (180, 180, 180),
        "COUNTDOWN": (0,   220, 255),
        "RECORDING": (0,    60, 255),
        "SAVED":     (0,   220, 100),
    }
    color = state_colors.get(state, (255, 255, 255))

    if state == "IDLE":
        msg = "Press [R] to record  |  [G] to set gesture  |  [Q] to quit"
    elif state == "COUNTDOWN":
        msg = f"Get ready in...  {countdown}"
    elif state == "RECORDING":
        filled = int((frame_idx / SEQUENCE_LENGTH) * 22)
        bar    = "█" * filled + "░" * (22 - filled)
        msg    = f"● REC  [{bar}]  {frame_idx}/{SEQUENCE_LENGTH}"
    elif state == "SAVED":
        msg = f"✔  Saved → sequence_{seq_idx}.npy    (R = next  |  G = change gesture)"
    else:
        msg = state

    cv2.putText(frame, msg, (12, h - 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1, cv2.LINE_AA)

    if state == "RECORDING":
        cv2.circle(frame, (w - 28, 28), 10, COLOR_DOT_REC, -1)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def next_sequence_index(gesture_dir: str) -> int:
    existing = [
        f for f in os.listdir(gesture_dir)
        if f.startswith("sequence_") and f.endswith(".npy")
    ]
    return len(existing)


def prompt_gesture_name(current: str) -> str:
    print("\n" + "─" * 52)
    name = input("  Enter gesture label (e.g. Hello, ThumbsUp): ").strip()
    print("─" * 52 + "\n")
    return name if name else current


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "=" * 52)
    print("  Neuro-Sign  |  Data Collector  (Tasks API)")
    print("=" * 52)
    print("  G  →  set gesture label")
    print("  R  →  record a 30-frame sequence")
    print("  Q  →  quit")
    print("=" * 52 + "\n")

    landmarker = build_landmarker(MODEL_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(f"Cannot open webcam at index {CAM_INDEX}.")

    gesture_label  = ""
    state          = "IDLE"     # IDLE | COUNTDOWN | RECORDING | SAVED
    sequence: list = []
    seq_idx        = 0
    frame_idx      = 0
    countdown_end  = 0.0
    last_result    = None       # most recent HandLandmarkerResult

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel

        # ── MediaPipe Tasks inference ──────────────────────────────────────
        # VIDEO mode requires a monotonically increasing timestamp in milliseconds
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        mp_image    = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        last_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # ── Draw skeleton ──────────────────────────────────────────────────
        draw_landmarks_on_frame(frame, last_result)

        # ── State machine ──────────────────────────────────────────────────
        keypoints = extract_keypoints(last_result)

        if state == "COUNTDOWN":
            remaining = max(int(countdown_end - time.time()) + 1, 0)
            draw_ui(frame, gesture_label, state, remaining, seq_idx, 0)
            if time.time() >= countdown_end:
                state     = "RECORDING"
                sequence  = []
                frame_idx = 0

        elif state == "RECORDING":
            sequence.append(keypoints)
            frame_idx += 1
            draw_ui(frame, gesture_label, state, 0, seq_idx, frame_idx)

            if frame_idx >= SEQUENCE_LENGTH:
                gesture_dir = os.path.join(DATA_DIR, gesture_label)
                os.makedirs(gesture_dir, exist_ok=True)
                seq_idx   = next_sequence_index(gesture_dir)
                save_path = os.path.join(gesture_dir, f"sequence_{seq_idx}.npy")
                arr       = np.array(sequence, dtype=np.float32)
                np.save(save_path, arr)
                print(f"[SAVED]  {save_path}  shape: {arr.shape}")
                state = "SAVED"

        else:   # IDLE or SAVED
            draw_ui(frame, gesture_label, state, 0, seq_idx, 0)

        cv2.imshow("Neuro-Sign | Data Collector  (Q to quit)", frame)

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        elif key in (ord('g'), ord('G')):
            cv2.destroyWindow("Neuro-Sign | Data Collector  (Q to quit)")
            gesture_label = prompt_gesture_name(gesture_label)
            state = "IDLE"

        elif key in (ord('r'), ord('R')):
            if not gesture_label:
                print("[WARN]  No gesture label set — press G first.")
            elif state not in ("RECORDING", "COUNTDOWN"):
                countdown_end = time.time() + 3
                state         = "COUNTDOWN"
                print(f"[INFO]  Recording '{gesture_label}' in 3 s...")

    # ── Cleanup ────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\n[DONE]  Data collector closed.\n")


if __name__ == "__main__":
    main()
