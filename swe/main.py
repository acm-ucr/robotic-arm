import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from mqtt_sender import send_coordinates

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "hand_landmarker.task"
CAM_INDEX = 0

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
# HELPER FUNCTION: CALCULATE DISTANCE
# -----------------------------
def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# -----------------------------
# HELPER FUNCTION: CALCULATE HAND OPENNESS
# -----------------------------
def calculate_hand_openness(points):
    """
    Calculate how open the hand is (0% = fully open, 100% = fully closed)
    Based on distances between the three fingertips
    """
    # Fingertip indices
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    
    # Calculate distances between each pair of fingertips
    thumb_to_index = calculate_distance(thumb_tip, index_tip)
    thumb_to_middle = calculate_distance(thumb_tip, middle_tip)
    #index_to_middle = calculate_distance(index_tip, middle_tip)
    
    # Average distance between fingertips (took out the index_to_middle distane)
    avg_distance = (thumb_to_index + thumb_to_middle) / 2
    
    # Normalize to percentage (these values may need calibration)
    # Fully open hand: avg_distance ≈ 150-200 pixels (fingers spread apart)
    # Fully closed hand: avg_distance ≈ 20-40 pixels (fingers together)
    
    # Define min/max distances (adjust based on your camera distance)
    MAX_DISTANCE = 300  # Fully open
    MIN_DISTANCE = 50   # Fully closed
    
    # Calculate percentage (inverted: small distance = high percentage)
    if avg_distance >= MAX_DISTANCE:
        percentage = 0
    elif avg_distance <= MIN_DISTANCE:
        percentage = 100
    else:
        # Linear interpolation
        percentage = 100 - ((avg_distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE) * 100)
    
    return int(max(0, min(100, percentage)))  # Clamp between 0-100

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
    num_hands=1  # Track only one hand
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

    timestamp_ms += 33 # ~30 FPS
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # -----------------------------
    # DRAW LANDMARKS & CALCULATE OPENNESS
    # -----------------------------
    if result.hand_landmarks:
        # Only process the first hand (since num_hands=1)
        hand_landmarks = result.hand_landmarks[0]
        points = []

        # Convert normalized landmarks → pixel coords
        for lm in hand_landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            points.append((x, y))

# --- ARM MOVEMENT LOGIC ---
        
        # 1. POSITIONAL MOVEMENT (Up/Down, Left/Right)
        # Your original logic: wrist is the anchor
        palm_x = points[0][0]
        palm_y = points[0][1]
        frame_height, frame_width = frame.shape[:2]
        coord_x = palm_x
        coord_y = frame_height - palm_y # Inverted so bottom is 0

        # 2. REACH MOVEMENT (Forward/Backward)
        # We measure the distance from Wrist (0) to Middle Finger Base (9)
        # As the arm extends toward the camera, this "arm_reach" value increases.
        arm_reach = int(calculate_distance(points[0], points[9]))

        # Calculate hand openness percentage
        openness_percentage = calculate_hand_openness(points)
        
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
        
        cv2.putText(frame, f"Wrist Y: {coord_y}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Wrist X: {coord_x}", (50,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Front/Back: {arm_reach}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display percentage on screen
        # Position text near the wrist
        text_position = (points[0][0] - 50, points[0][1] - 30)
        
        # Draw background rectangle for better readability
        text = f"{openness_percentage}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 3
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (text_position[0] - 10, text_position[1] - text_height - 10),
                     (text_position[0] + text_width + 10, text_position[1] + baseline + 10),
                     (0, 0, 0), -1)

        # Draw text
        cv2.putText(frame, text, text_position, font, font_scale, 
                   (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # Optional: Display status text
        status = "CLOSED" if openness_percentage > 90 else "OPEN" if openness_percentage < 30 else "PARTIAL"
        status_position = (text_position[0], text_position[1] + 40)
        cv2.putText(frame, status, status_position, font, 0.6, 
                   (255, 255, 255), 2, cv2.LINE_AA)

        # Calculate palm center coordinates (wrist is point 0)
        palm_x = points[0][0]
        palm_y = points[0][1]
        
        # Convert to bottom-left origin (0,0)
        # Bottom-left means: x increases rightward, y increases upward
        frame_height, frame_width = frame.shape[:2]
        coord_x = palm_x
        coord_y = frame_height - palm_y
        
        # Display coordinates in bottom-right corner of screen
        coord_text = f"Hand: ({coord_x}, {coord_y})"
        coord_position = (frame_width - 300, frame_height - 20)
        cv2.putText(frame, coord_text, coord_position, font, 0.7, 
                   (0, 255, 255), 2, cv2.LINE_AA)
        
        # Send coordinates via MQTT with a lag of 5 frames
        if timestamp_ms % 5 == 0:
            send_coordinates(coord_x, coord_y, openness_percentage)

    cv2.imshow("MediaPipe Tasks – Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()