from __future__ import annotations
import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
]

latest_hand_result = None
latest_pose_result = None

def handle_hand_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_hand_result
    latest_hand_result = result

def handle_pose_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_pose_result
    latest_pose_result = result

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=handle_hand_result
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_pose_result
)

with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
     PoseLandmarker.create_from_options(pose_options) as pose_landmarker:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        # Send the same frame to both landmarkers
        hand_landmarker.detect_async(mp_image, timestamp_ms)
        pose_landmarker.detect_async(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        # --- Draw Pose (arms/shoulders) ---
        if latest_pose_result and latest_pose_result.pose_landmarks:
            for pose_landmarks in latest_pose_result.pose_landmarks:
                for start_idx, end_idx in POSE_CONNECTIONS:
                    start = pose_landmarks[start_idx]
                    end = pose_landmarks[end_idx]
                    if start.visibility < 0.5 or end.visibility < 0.5:
                        continue
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Yellow lines

                for idx in set(i for pair in POSE_CONNECTIONS for i in pair):
                    lm = pose_landmarks[idx]
                    if lm.visibility < 0.5:
                        continue
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 7, (255, 200, 0), -1)  # Yellow dots

        # --- Draw Hands (fingers) ---
        if latest_hand_result and latest_hand_result.hand_landmarks:
            for hand_landmarks in latest_hand_result.hand_landmarks:
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines

                for landmark in hand_landmarks:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # Red dots

        cv2.imshow('Hand + Pose Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()