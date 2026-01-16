import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

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
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            
            # Convert to Pixels
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            
            # Draw big circles on tips
            cv.circle(frame, (tx, ty), 15, (255, 255, 0), cv.FILLED)
            cv.circle(frame, (cx, cy), 15, (0, 255, 255), cv.FILLED)
            cv.circle(frame, (mx, my), 15, (255, 0, 255), cv.FILLED)

            # Display Coordinates text
            cv.putText(frame, f"Thumb: ({tx}, {ty})", (tx + 20, ty - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.putText(frame, f"Index: ({cx}, {cy})", (cx + 20, cy - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.putText(frame, f"Middle: ({mx}, {my})", (mx + 20, my - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xff == ord('q'):
        break

cam.release()
