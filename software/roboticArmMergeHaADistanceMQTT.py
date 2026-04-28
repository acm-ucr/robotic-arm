from __future__ import annotations
import cv2
import mediapipe as mp
import time
import json
import math
import paho.mqtt.client as mqtt
import numpy as np

# --- MQTT Setup ---
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_PUB = "arm/servos"
TOPIC_SUB = "arm/feedback"

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

def on_connect(client, userdata, flags, rc):
    print(f"MQTT connected (rc={rc})")
    client.subscribe(TOPIC_SUB)

def on_message(client, userdata, msg):
    print(f"Arm feedback: {msg.payload.decode()}")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(BROKER, PORT)
mqtt_client.loop_start()

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connection pairs for drawing (21 keypoints)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# Simplified hand connections for pinky detection
HAND_CONNECTIONS_SIMPLIFIED = [
    (0,4),(0,8),(0,12),(4,8),(8,12),(4,12)
]

# Arm connections
LEFT_ARM_CONNECTIONS = [(11, 13), (13, 15)]
RIGHT_ARM_CONNECTIONS = [(12, 14), (14, 16)]

# Global variables
latest_hand_result = None
latest_pose_result = None
min_scale = None
max_scale = None
min_inv_scale = 9
max_inv_scale = 16

def compute_hand_scale(hand_landmarks):
    """Compute hand scale based on distance between landmarks."""
    p0 = hand_landmarks[0]
    p1 = hand_landmarks[1]
    dx = p0.x - p1.x
    dy = p0.y - p1.y
    return (dx**2 + dy**2)**0.5

def compute_grip(hand_landmarks):
    palm = hand_landmarks[0]
    fingertips = [hand_landmarks[i] for i in [4, 8, 12, 16, 20]]
    MAX_DIST = 0.35
    avg_dist = sum(
        ((f.x - palm.x)**2 + (f.y - palm.y)**2)**0.5
        for f in fingertips
    ) / len(fingertips)
    return round(1.0 - min(avg_dist / MAX_DIST, 1.0), 3)

def compute_rotation(world_landmarks):
    # Use three landmarks from the palm: wrist, thumb MCP, pinky MCP
    points = np.array([[world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z],
                       [world_landmarks[5].x, world_landmarks[5].y, world_landmarks[5].z],
                       [world_landmarks[17].x, world_landmarks[17].y, world_landmarks[17].z]])

    wrist = points[0]
    thumb = points[1]
    pinky = points[2]

    # Compute palm normal from three co-planar points
    v1 = pinky - wrist
    v2 = thumb - pinky
    normal = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 0:
        normal /= normal_norm

    # Use X and Z components to determine left/right and front/back
    front_score = -normal[2]
    side_score = normal[0]

    # Signed angle from front origin, positive to the right, negative to the left
    angle = math.degrees(math.atan2(side_score, front_score))
    orientation_angle = round(angle, 1)

    # Determine the orientation label in 8 sectors centered on front/right/back/left
    if -22.5 <= angle <= 22.5:
        orientation = 'front'
    elif 22.5 < angle <= 67.5:
        orientation = 'front right'
    elif 67.5 < angle <= 112.5:
        orientation = 'right'
    elif 112.5 < angle <= 157.5:
        orientation = 'back right'
    elif angle > 157.5 or angle < -157.5:
        orientation = 'back'
    elif -157.5 <= angle < -112.5:
        orientation = 'back left'
    elif -112.5 <= angle < -67.5:
        orientation = 'left'
    else:
        orientation = 'front left'

    return orientation, orientation_angle

def handle_hand_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_hand_result
    latest_hand_result = result

def handle_pose_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_pose_result
    latest_pose_result = result

def pick_one_hand(hand_result):
    """Select the best right hand from detected hands."""
    if not hand_result or not hand_result.hand_landmarks:
        return None, None, None
    
    # Filter for right hands only
    right_hand_indices = [
        i for i in range(len(hand_result.handedness))
        if hand_result.handedness[i][0].display_name == 'Right'
    ]
    
    if not right_hand_indices:
        return None, None, None
    
    if len(right_hand_indices) == 1:
        idx = right_hand_indices[0]
        return hand_result.hand_landmarks[idx], hand_result.hand_world_landmarks[idx], hand_result.handedness[idx][0]
    
    # Multiple right hands: select the one with highest confidence
    right_scores = [
        hand_result.handedness[i][0].score
        for i in right_hand_indices
    ]
    best_right_index = right_hand_indices[right_scores.index(max(right_scores))]
    return hand_result.hand_landmarks[best_right_index], hand_result.hand_world_landmarks[best_right_index], hand_result.handedness[best_right_index][0]

def pick_one_arm(pose_landmarks, selected_handedness):
    """Select arm based on hand handedness."""
    if selected_handedness is None:
        return RIGHT_ARM_CONNECTIONS, [12, 14, 16]
    if selected_handedness.display_name == 'Left':
        return LEFT_ARM_CONNECTIONS, [11, 13, 15]
    else:
        return RIGHT_ARM_CONNECTIONS, [12, 14, 16]

# Configure MediaPipe options
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

# Main webcam loop
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

        selected_hand, selected_world_hand, selected_handedness = pick_one_hand(latest_hand_result)

        # --- Draw Pose (one arm only) ---
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

                # Print arm coordinates
                arm_name = "Left Arm" if selected_handedness and selected_handedness.display_name == 'Left' else "Right Arm"
                print(f"\n{arm_name} Coordinates:")
                arm_labels = ["Shoulder", "Elbow", "Wrist"]
                for label, idx in zip(arm_labels, arm_indices):
                    lm = pose_landmarks[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    z = lm.z  # Z coordinate from MediaPipe (normalized)
                    print(f"  {label} (idx {idx}): ({x}, {y}, {z:.3f}) | Visibility: {lm.visibility:.3f}")

        # --- Draw Hand ---
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

            # Compute hand scale and Z value
            scale = compute_hand_scale(selected_hand)
            inv_scale = 1.0 / scale if scale > 0 else 0

            if min_scale is None or scale < min_scale:
                min_scale = scale
            if max_scale is None or scale > max_scale:
                max_scale = scale

            z_value = 0.5
            if max_inv_scale > min_inv_scale:
                z_value = 1 - (inv_scale - min_inv_scale) / (max_inv_scale - min_inv_scale)
            z_value = max(0, min(1, z_value))

            # Compute hand center
            hand_center_x = sum(lm.x for lm in selected_hand) / len(selected_hand)
            hand_center_y = sum(lm.y for lm in selected_hand) / len(selected_hand)

            # Display hand control info
            wrist = selected_hand[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f'Controlling: {selected_handedness.display_name}',
                        (wx - 30, wy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Publish MQTT data
            grip = compute_grip(selected_hand)
            orientation, orientation_angle = compute_rotation(selected_world_hand)
            payload = json.dumps({
                "x": round(wrist.x, 3),
                "y": round(wrist.y, 3),
                "z": round(z_value, 3),
                "grip": grip,
                "palm_orientation": orientation,
                "orientation_angle": orientation_angle,
            })
            mqtt_client.publish(TOPIC_PUB, payload)

            # Print debug info
            print(f"Hand: {selected_handedness.display_name} | Z: {z_value:.3f} | Scale: {scale:.4f} | Grip: {grip:.3f} | Orientation: {orientation} | Angle: {orientation_angle}°")

        cv2.imshow('Hand + Pose Tracking with Distance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    mqtt_client.loop_stop()
    mqtt_client.disconnect()

    cap.release()
    cv2.destroyAllWindows()
