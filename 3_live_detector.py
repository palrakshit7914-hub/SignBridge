"""
Real-time Indian Sign Language gesture recognition from webcam.
Uses MediaPipe hand landmarks + trained Random Forest model (isl_model.pkl).
"""

import os
import sys
import threading
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "isl_model.pkl")
HAND_LANDMARKER_MODEL = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")
WEBCAM_INDEX = 0
NUM_HANDS = 2
NUM_LANDMARKS = 21

CONFIDENCE_THRESHOLD = 0.75
STABLE_FRAMES_REQUIRED = 8
SPEECH_COOLDOWN_SEC = 2.0

WINDOW_TITLE = "ISL Live Detector"

# MediaPipe hand landmark connections (thumb → pinky + palm)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class SpeechEngine:
    """Debounced text-to-speech so confident predictions are spoken once."""

    def __init__(self) -> None:
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 165)
        self._lock = threading.Lock()
        self._last_spoken = ""
        self._last_spoken_at = 0.0

    def speak_if_allowed(self, text: str) -> None:
        now = time.time()
        if text == self._last_spoken and (now - self._last_spoken_at) < SPEECH_COOLDOWN_SEC:
            return

        self._last_spoken = text
        self._last_spoken_at = now
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str) -> None:
        with self._lock:
            self._engine.say(text)
            self._engine.runAndWait()


def load_classifier(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Trained model not found: {filepath}\n"
            "Run train_isl_model.py first to create isl_model.pkl."
        )
    return joblib.load(filepath)


def landmarks_to_features(landmarks) -> np.ndarray:
    """Flatten 21 (x, y, z) landmarks into the feature vector the model expects."""
    row = []
    for lm in landmarks[:NUM_LANDMARKS]:
        row.extend([lm.x, lm.y, lm.z])
    return np.array(row, dtype=np.float32).reshape(1, -1)


def predict_gesture(model, landmarks) -> tuple[str, float]:
    features = landmarks_to_features(landmarks)
    probabilities = model.predict_proba(features)[0]
    best_idx = int(np.argmax(probabilities))
    label = model.classes_[best_idx]
    confidence = float(probabilities[best_idx])
    return label, confidence


def draw_hand_skeleton(
    frame: np.ndarray,
    landmarks,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], color, 2, cv2.LINE_AA)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)


def draw_prediction(
    frame: np.ndarray,
    label: str,
    confidence: float,
    hand_count: int,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if label:
        display_text = f"{label}  ({confidence * 100:.0f}%)"
        cv2.putText(
            frame,
            display_text,
            (20, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            1.8,
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Show your hand",
            (20, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (0, 200, 255),
            3,
            cv2.LINE_AA,
        )

    help_text = f"Hands tracked: {hand_count}/{NUM_HANDS}  |  Press Q to quit"
    cv2.putText(
        frame,
        help_text,
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    try:
        if not os.path.exists(HAND_LANDMARKER_MODEL):
            raise FileNotFoundError(
                f"MediaPipe model not found: {HAND_LANDMARKER_MODEL}\n"
                "Place hand_landmarker.task inside the models/ folder."
            )

        classifier = load_classifier(MODEL_FILE)
        speech = SpeechEngine()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {WEBCAM_INDEX}.")

        print("ISL Live Detector started.")
        print("Show a sign to the camera. Press Q in the video window to exit.\n")

        stable_label = ""
        stable_count = 0
        displayed_label = ""
        displayed_confidence = 0.0

        skeleton_colors = [(0, 255, 0), (255, 180, 0)]

        with HandLandmarker.create_from_options(options) as landmarker:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read from webcam. Exiting.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int(time.time() * 1000)

                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0

                    for idx, hand_landmarks in enumerate(result.hand_landmarks or []):
                        color = skeleton_colors[idx % len(skeleton_colors)]
                        draw_hand_skeleton(frame, hand_landmarks, color)

                    if result.hand_landmarks:
                        # Model was trained on the primary (first) detected hand.
                        label, confidence = predict_gesture(classifier, result.hand_landmarks[0])

                        if confidence >= CONFIDENCE_THRESHOLD:
                            if label == stable_label:
                                stable_count += 1
                            else:
                                stable_label = label
                                stable_count = 1

                            displayed_label = label
                            displayed_confidence = confidence

                            if stable_count >= STABLE_FRAMES_REQUIRED:
                                speech.speak_if_allowed(label)
                        else:
                            stable_label = ""
                            stable_count = 0
                            displayed_label = label
                            displayed_confidence = confidence
                    else:
                        stable_label = ""
                        stable_count = 0
                        displayed_label = ""
                        displayed_confidence = 0.0

                    draw_prediction(frame, displayed_label, displayed_confidence, hand_count)
                    cv2.imshow(WINDOW_TITLE, frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Quit requested. Exiting.")
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
