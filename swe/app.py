import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import csv
import time
import math

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Initializing CSV outputs
positions = ['thumb_x', 'thumb_y', 'index_x', 'index_y', 'mid_x', 'mid_y', 'wrist_x', 'wrist_y', 'openness_percentage']
csv_file_path = 'positions.csv'
last_save_time = 0
save_time_interval = 0.5 # seconds

# Stationary detection initialization
stationary_start_time = None
required_stationary_duration = 1.0  # seconds
movement_threshold = 5  # pixels
prev_wrist_pos = None
is_stationary = False

# Initialize CSV once
with open(csv_file_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['thumb_x', 'thumb_y', 'index_x', 'index_y', 'mid_x', 'mid_y', 'wrist_x', 'wrist_y', 'openness_percentage'])

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_hand_openness(thumb_tip, index_tip, middle_tip):
    """
    Calculate how open the hand is (0% = fully open, 100% = fully closed)
    Based on distances between the three fingertips
    """
    # Calculate distances between each pair of fingertips
    thumb_to_index = calculate_distance(thumb_tip, index_tip)
    thumb_to_middle = calculate_distance(thumb_tip, middle_tip)
    index_to_middle = calculate_distance(index_tip, middle_tip)
    
    # Average distance between fingertips
    avg_distance = (thumb_to_index + thumb_to_middle + index_to_middle) / 3
    
    # Define min/max distances (adjust based on your camera distance)
    MAX_DISTANCE = 150  # Fully open
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

cam = cv.VideoCapture(0)

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera Frame not available")
        continue

    h, w, c = frame.shape
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    hands_detected = hands.process(frame)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:

            # Define the chains of landmarks for the 3 fingers we want
            fingers_to_draw = [
                [0, 1, 2, 3, 4],       # Thumb
                [0, 5, 6, 7, 8],       # Index
                [0, 9, 10, 11, 12]     # Middle
            ]

            for finger in fingers_to_draw:
        
                poly_coords = []

                for i in finger:
                    lm = hand_landmarks.landmark[i]
                    # Convert to integer pixels
                    px, py = int(lm.x * w), int(lm.y * h)
                    poly_coords.append((px, py))

                    # Draw a small joint circle (Green)
                    cv.circle(frame, (px, py), 5, (0, 255, 0), cv.FILLED)
                
                # Connect the dots with lines (Blue)
                for i in range(len(poly_coords) - 1):
                    cv.line(frame, poly_coords[i], poly_coords[i+1], (255, 0, 0), 2)

            # Access the Tips
            wrist_landmark = hand_landmarks.landmark[0]
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            
            # Convert to Pixels
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            wx, wy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)
            
            # Calculate openness percentage
            openness_percentage = calculate_hand_openness((tx, ty), (cx, cy), (mx, my))
            
            # Check if wrist is stationary
            current_time = time.time()
            if prev_wrist_pos is not None:
                distance_moved = math.sqrt((wx - prev_wrist_pos[0])**2 + (wy - prev_wrist_pos[1])**2)
                
                if distance_moved < movement_threshold:
                    if stationary_start_time is None:
                        stationary_start_time = current_time
                    elif current_time - stationary_start_time >= required_stationary_duration:
                        is_stationary = True
                    else:
                        is_stationary = False
                else:
                    stationary_start_time = None
                    is_stationary = False
            
            # Update previous wrist position
            prev_wrist_pos = (wx, wy)
            
            # Draw big circles on tips
            cv.circle(frame, (tx, ty), 15, (255, 255, 0), cv.FILLED)
            cv.circle(frame, (cx, cy), 15, (0, 255, 255), cv.FILLED)
            cv.circle(frame, (mx, my), 15, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (wx, wy), 15, (255, 255, 255), cv.FILLED)

            # Display Coordinates text
            cv.putText(frame, f"Thumb: ({tx}, {ty})", (tx + 20, ty - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.putText(frame, f"Index: ({cx}, {cy})", (cx + 20, cy - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.putText(frame, f"Middle: ({mx}, {my})", (mx + 20, my - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.putText(frame, f"Wrist: ({wx}, {wy})", (wx + 20, wy - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display openness percentage
            cv.putText(frame, f"Openness: {openness_percentage}%", (50, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Display status text
            status = "CLOSED" if openness_percentage > 70 else "OPEN" if openness_percentage < 30 else "PARTIAL"
            cv.putText(frame, status, (50, 90), cv.FONT_HERSHEY_SIMPLEX, 1.0, 
                       (0, 255, 0), 2)
            
            # Display stationary status
            stationary_text = "STATIONARY" if is_stationary else "MOVING"
            stationary_color = (0, 255, 0) if is_stationary else (0, 0, 255)
            cv.putText(frame, stationary_text, (50, 130), cv.FONT_HERSHEY_SIMPLEX, 1.0, 
                       stationary_color, 2)
            
            # Print to terminal when conditions are met
            if is_stationary and openness_percentage > 70:
                print(f"Hand is STATIONARY and CLOSED ({openness_percentage}%)")
            elif not is_stationary:
                print("Hand is MOVING")

            
            
            if current_time - last_save_time >= save_time_interval:
                last_save_time = current_time
                with open(csv_file_path, mode='a', newline='') as csvfile:
                    # create CSV writer object
                    csv_writer = csv.writer(csvfile)

                    # writing data to file
                    csv_writer.writerow([tx, ty, cx, cy, mx, my, wx, wy, openness_percentage])
            

    else:
        # Reset stationary tracking when no hand is detected
        prev_wrist_pos = None
        stationary_start_time = None
        is_stationary = False

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xff == ord('q'):
        break

cam.release()
