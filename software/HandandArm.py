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

LEFT_ARM_CONNECTIONS  = [(11, 13), (13, 15)]
RIGHT_ARM_CONNECTIONS = [(12, 14), (14, 16)]

latest_hand_result = None
latest_pose_result = None

def handle_hand_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_hand_result
    latest_hand_result = result

def handle_pose_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_pose_result
    latest_pose_result = result

def pick_one_hand(hand_result):
    if not hand_result or not hand_result.hand_landmarks:
        return None, None
    if len(hand_result.hand_landmarks) == 1:
        return hand_result.hand_landmarks[0], hand_result.handedness[0][0]
    handedness_scores = [
        hand_result.handedness[i][0].score
        for i in range(len(hand_result.handedness))
    ]
    best_index = handedness_scores.index(max(handedness_scores))
    return hand_result.hand_landmarks[best_index], hand_result.handedness[best_index][0]

def pick_one_arm(pose_landmarks, selected_handedness):
    if selected_handedness is None:
        return RIGHT_ARM_CONNECTIONS, [12, 14, 16]
    if selected_handedness.display_name == 'Left':
        return LEFT_ARM_CONNECTIONS, [11, 13, 15]
    else:
        return RIGHT_ARM_CONNECTIONS, [12, 14, 16]

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

        hand_landmarker.detect_async(mp_image, timestamp_ms)
        pose_landmarker.detect_async(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        selected_hand, selected_handedness = pick_one_hand(latest_hand_result)

        # --- Draw Pose (one arm only, no shoulder bar) ---
        if latest_pose_result and latest_pose_result.pose_landmarks:
            for pose_landmarks in latest_pose_result.pose_landmarks:
                arm_connections, arm_indices = pick_one_arm(pose_landmarks, selected_handedness)

                for start_idx, end_idx in arm_connections:
                    start = pose_landmarks[start_idx]
                    end = pose_landmarks[end_idx]
                    if start.visibility < 0.5 or end.visibility < 0.5:
                        continue
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)

                for idx in arm_indices:
                    lm = pose_landmarks[idx]
                    if lm.visibility < 0.5:
                        continue
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 7, (255, 200, 0), -1)

        # --- Draw one Hand only ---
        if selected_hand:
            for start_idx, end_idx in HAND_CONNECTIONS:
                start = selected_hand[start_idx]
                end = selected_hand[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for landmark in selected_hand:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            wrist = selected_hand[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f'Controlling: {selected_handedness.display_name}',
                        (wx - 30, wy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Hand + Pose Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()