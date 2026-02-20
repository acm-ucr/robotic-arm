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

PALM_INDICES = [0]
PALM_COLOR = (0, 0, 255)
DOT_RADIUS = 7
BONE_THICKNESS = 3

# -----------------------------
# HELPER: EUCLIDEAN DISTANCE
# -----------------------------
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# -----------------------------
# HAND OPENNESS % (thumb, index, middle)
# 0% = fully closed, 100% = fully open
# -----------------------------
def calculate_openness(hand_landmarks):
    """
    Uses extension ratios of thumb, index finger, and middle finger.
    Each finger contributes equally to the final percentage.
    """
    lm = hand_landmarks
    wrist_x = lm[0].x

    # Index: tip (8) above PIP (6) → extended
    index_diff  = lm[6].y - lm[8].y
    index_ratio = max(0.0, min(1.0, index_diff / 0.12))

    # Middle: tip (12) above PIP (10) → extended
    middle_diff  = lm[10].y - lm[12].y
    middle_ratio = max(0.0, min(1.0, middle_diff / 0.12))

    # Thumb: tip (4) further from wrist x than base (2) → extended
    thumb_diff  = abs(lm[4].x - wrist_x) - abs(lm[2].x - wrist_x)
    thumb_ratio = max(0.0, min(1.0, thumb_diff / 0.08))

    return int((index_ratio + middle_ratio + thumb_ratio) / 3 * 100)

# -----------------------------
# PALM FACING DIRECTION
# 100% = fully facing camera (palm toward you)
#   0% = fully facing away  (back of hand toward you)
#
# Method: cross product of two palm vectors gives the palm normal.
# The z-component tells us which way the palm is pointing.
# -----------------------------
def calculate_palm_facing(hand_landmarks):
    """
    Returns:
      direction (str)  — "FACING CAMERA", "FACING AWAY", or "SIDE-ON"
      facing_pct (int) — 100% = palm toward camera, 0% = back of hand toward camera
    """
    lm = hand_landmarks

    # V1: wrist (0) → index MCP (5)
    # V2: wrist (0) → pinky MCP (17)
    v1x = lm[5].x - lm[0].x
    v1y = lm[5].y - lm[0].y
    v1z = lm[5].z - lm[0].z

    v2x = lm[17].x - lm[0].x
    v2y = lm[17].y - lm[0].y
    v2z = lm[17].z - lm[0].z

    # Cross product → palm normal vector
    nx = v1y * v2z - v1z * v2y
    ny = v1z * v2x - v1x * v2z
    nz = v1x * v2y - v1y * v2x

    magnitude = math.sqrt(nx**2 + ny**2 + nz**2)
    if magnitude == 0:
        return "SIDE-ON", 50

    nz_norm = nz / magnitude  # -1.0 to +1.0

    # +1 → palm facing camera (0%), -1 → back of hand (100%)
    facing_pct = int((1 - (nz_norm + 1) / 2) * 100)

    if facing_pct <= 40:
        direction = "FACING AWAY"
    elif facing_pct >= 60:
        direction = "FACING CAMERA"
    else:
        direction = "SIDE-ON"

    return direction, facing_pct

# -----------------------------
# DRAW HUD BOX (top-left)
# -----------------------------
def draw_hud(frame, openness_pct, direction, facing_pct, coord_x, coord_y, arm_reach):
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 14
    line_gap = 10
    box_x, box_y = 12, 12

    palm_color = (0, 255, 120) if direction == "FACING CAMERA" else \
                 (80, 80, 220) if direction == "FACING AWAY" else \
                 (200, 200, 200)

    lines = [
        (f"Openness:   {openness_pct}%",   (0, 220, 255)),
        (f"Palm:       {direction}",         palm_color),
        (f"Facing:     {facing_pct}%",      (0, 220, 255)),
        (f"Wrist X:    {coord_x}",          (255, 255, 255)),
        (f"Wrist Y:    {coord_y}",          (255, 255, 255)),
        (f"Front/Back: {arm_reach}",        (0, 255, 255)),
    ]

    line_sizes = [cv2.getTextSize(text, font, 0.85, 2) for text, _ in lines]
    box_w = max(sz[0][0] for sz in line_sizes) + padding * 2
    box_h = sum(sz[0][1] + line_gap for sz in line_sizes) + padding * 2

    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (80, 80, 80), 1)

    cursor_y = box_y + padding
    for (text, color), (size, baseline) in zip(lines, line_sizes):
        cursor_y += size[1]
        cv2.putText(frame, text, (box_x + padding, cursor_y), font, 0.85, color, 2, cv2.LINE_AA)
        cursor_y += line_gap

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
    num_hands=1
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

    frame_h, frame_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms += 33  # ~30 FPS
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        points = []

        for lm in hand_landmarks:
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            points.append((x, y))

        # --- Calculations ---
        openness_pct          = calculate_openness(hand_landmarks)
        direction, facing_pct = calculate_palm_facing(hand_landmarks)

        # Wrist position (bottom-left origin)
        coord_x   = points[0][0]
        coord_y   = frame_h - points[0][1]

        # Arm reach: wrist (0) → middle finger base (9)
        arm_reach = int(calculate_distance(points[0], points[9]))

        # --- Draw palm dot ---
        for idx in PALM_INDICES:
            cv2.circle(frame, points[idx], DOT_RADIUS, PALM_COLOR, -1)

        # --- Draw fingers ---
        for indices, color in FINGERS.values():
            for idx in indices:
                cv2.circle(frame, points[idx], DOT_RADIUS, color, -1)
            prev = 0
            for idx in indices:
                cv2.line(frame, points[prev], points[idx], color, BONE_THICKNESS)
                prev = idx

        # --- Openness % near wrist ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (points[0][0] - 50, points[0][1] - 30)
        grip_text = f"{openness_pct}%"
        (tw, th), bl = cv2.getTextSize(grip_text, font, 1.2, 3)
        cv2.rectangle(frame,
                      (text_pos[0] - 10, text_pos[1] - th - 10),
                      (text_pos[0] + tw + 10, text_pos[1] + bl + 10),
                      (0, 0, 0), -1)
        cv2.putText(frame, grip_text, text_pos, font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        status = "CLOSED" if openness_pct > 90 else "OPEN" if openness_pct < 30 else "PARTIAL"
        cv2.putText(frame, status, (text_pos[0], text_pos[1] + 40),
                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # --- HUD (top-left) ---
        draw_hud(frame, openness_pct, direction, facing_pct, coord_x, coord_y, arm_reach)

        # --- Coordinates (bottom-right) ---
        cv2.putText(frame, f"Hand: ({coord_x}, {coord_y})",
                    (frame_w - 300, frame_h - 20),
                    font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # --- MQTT ---
        if timestamp_ms % 5 == 0:
            send_coordinates(coord_x, coord_y, openness_pct)

    cv2.imshow("MediaPipe Tasks - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()