"""
ISL Hand Landmark Data Collector (MediaPipe Tasks API Version)
"""

import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Imports for new Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_FILE = "isl_dataset.csv"
FRAMES_PER_GESTURE = 50          # Number of frames to capture per spacebar press
WEBCAM_INDEX = 0                 # Default laptop webcam
NUM_LANDMARKS = 21               # 21 landmarks per hand

# Path to your downloaded .task file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


def build_csv_header() -> list[str]:
    header = ["label", "recording_id", "frame_idx"]
    for i in range(NUM_LANDMARKS):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    return header


def ensure_csv_exists(filepath: str) -> None:
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(build_csv_header())


def append_frame_to_csv(
    filepath: str,
    label: str,
    recording_id: str,
    frame_idx: int,
    landmarks: list[tuple[float, float, float]],
) -> None:
    row = [label, recording_id, frame_idx]
    for x, y, z in landmarks:
        row.extend([x, y, z])

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def draw_overlay(
    frame: np.ndarray,
    detection_result,
    label: str,
    saved_frames: int,
    recording: bool,
    recording_progress: int,
) -> None:
    # Draw simple keypoints if landmarks are detected
    if detection_result and detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Semi-transparent header box
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    status_lines = [
        f"Label: {label}",
        f"Saved frames (this session): {saved_frames}",
        "SPACEBAR = record 50 frames  |  Q = quit",
    ]

    if recording:
        status_lines.append(f"Recording... {recording_progress}/{FRAMES_PER_GESTURE}")

    for i, line in enumerate(status_lines):
        cv2.putText(
            frame,
            line,
            (10, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def prompt_for_label() -> str:
    while True:
        label = input("\nEnter gesture label (e.g., Help, Question, Yes, No): ").strip()
        if label:
            return label
        print("Label cannot be empty. Please try again.")


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please make sure 'hand_landmarker.task' is inside a 'models' folder!"
        )

    # Initialize HandLandmarker using the Tasks API
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {WEBCAM_INDEX}.")

    ensure_csv_exists(CSV_FILE)

    current_label = prompt_for_label()
    total_saved_frames = 0
    is_recording = False
    recording_frame_count = 0
    recording_id = ""

    print(f"\nCollecting data for label: '{current_label}'")
    print("Press SPACEBAR in the webcam window to record. Press Q to quit.\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam. Exiting.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert frame to MediaPipe Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Get current timestamp in milliseconds
                frame_timestamp_ms = int(time.time() * 1000)

                # Detect landmarks
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                if is_recording and result.hand_landmarks:
                    hand_lm = result.hand_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lm]
                    
                    append_frame_to_csv(
                        CSV_FILE,
                        current_label,
                        recording_id,
                        recording_frame_count,
                        landmarks,
                    )
                    recording_frame_count += 1
                    total_saved_frames += 1

                    if recording_frame_count >= FRAMES_PER_GESTURE:
                        is_recording = False
                        print(f"Saved {FRAMES_PER_GESTURE} frames for '{current_label}'.")

                draw_overlay(
                    frame,
                    result,
                    current_label,
                    total_saved_frames,
                    is_recording,
                    recording_frame_count,
                )

                cv2.imshow("ISL Landmark Collector", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Quit requested. Exiting.")
                    break

                if key == ord(" ") and not is_recording:
                    is_recording = True
                    recording_frame_count = 0
                    recording_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    print(f"Recording {FRAMES_PER_GESTURE} frames for '{current_label}'...")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nDone. Dataset saved to: {os.path.abspath(CSV_FILE)}")


if __name__ == "__main__":
    main()