from __future__ import annotations
import cv2
import mediapipe as mp
import time

# --- Setup MediaPipe Tasks and Options ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connection pairs for drawing (21 keypoints)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)     # Pinky
]

# A global variable to store the latest results asynchronously
latest_result = None

# --- Define the Callback Function ---
# In LIVE_STREAM mode, MediaPipe processes frames in the background.
# Whenever it finishes a frame, it calls this function.
def handle_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# --- Configure the Hand Landmarker ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'), # Ensure this matches your downloaded model file name
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2, # You can change this to detect more hands
    result_callback=handle_result)

# --- Start the Webcam Loop ---
with HandLandmarker.create_from_options(options) as landmarker:
    # 0 is usually the default built-in webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # 1. Prepare the Data
        # OpenCV captures in BGR, but MediaPipe expects RGB format.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the numpy array to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 2. Calculate the timestamp
        # The detect_async function requires a monotonically increasing timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        
        # 3. Run the Detection
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        # 4. Handle and Display the Results
        # If our callback has caught a result, draw the landmarks on the current frame
        if latest_result and latest_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in latest_result.hand_landmarks:
                
                # Draw connections
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start_landmark = hand_landmarks[start_idx]
                    end_landmark = hand_landmarks[end_idx]
                    
                    start_x = int(start_landmark.x * w)
                    start_y = int(start_landmark.y * h)
                    end_x = int(end_landmark.x * w)
                    end_y = int(end_landmark.y * h)
                    
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                
                # Draw landmarks (circles)
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        # Show the frame to the user
        cv2.imshow('MediaPipe Hand Landmarker (Live Stream)', frame)
        
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()