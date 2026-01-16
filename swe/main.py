import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "hand_landmarker.task"
CAM_INDEX = 1

# -----------------------------
# FINGER GROUPS + COLORS
# (ring & pinky REMOVED)
# -----------------------------
FINGERS = {
    "thumb":  ([1, 2, 3, 4],    (255, 0, 0)),    # Blue
    "index":  ([5, 6, 7, 8],    (0, 255, 0)),    # Green
    "middle": ([9, 10, 11, 12], (0, 255, 255))   # Yellow
}

PALM_INDICES = [0]          # Wrist / palm base
PALM_COLOR = (0, 0, 255)    # Red
DOT_RADIUS = 7              # Bigger dots
BONE_THICKNESS = 3

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open webcam {CAM_INDEX}")

# -----------------------------
# HAND LANDMARKER (VIDEO MODE)
# -----------------------------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

# -----------------------------
# MAIN LOOP
# -----------------------------
timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms += 33  # ~30 FPS
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # -----------------------------
    # DRAW LANDMARKS
    # -----------------------------
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = []

            # Convert normalized landmarks → pixel coords
            for lm in hand_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                points.append((x, y))

            # Draw palm (red)
            for idx in PALM_INDICES:
                cv2.circle(frame, points[idx], DOT_RADIUS, PALM_COLOR, -1)

            # Draw fingers
            for indices, color in FINGERS.values():
                # Draw joints
                for idx in indices:
                    cv2.circle(frame, points[idx], DOT_RADIUS, color, -1)

                # Draw bones (connected to wrist)
                prev = 0
                for idx in indices:
                    cv2.line(frame, points[prev], points[idx], color, BONE_THICKNESS)
                    prev = idx

    cv2.imshow("MediaPipe Tasks – Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
