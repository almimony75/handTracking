import cv2
import mediapipe as mp
from pynput.mouse import Controller
import time
import pyautogui

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize mouse controller
mouse_controller = Controller()

# Add small delay to allow camera to start
time.sleep(2)

# Constants
SENSITIVITY_FACTOR = 2  # Adjust this value to make cursor more/less sensitive
THUMB_INDEX_THRESHOLD = 0.9 # Distance threshold for left click

# Calibration center (relative to webcam frame)
calibration_radius = 20  # Radius of the calibration circle

# Helper function to calculate Euclidean distance
def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Flag to indicate if calibration is complete
calibrated = False

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    h, w, c = frame.shape

    # Calculate the center of the frame for the calibration circle
    calibration_center = (w // 2, h // 2)

    # Draw the calibration circle only if calibration is not complete
    if not calibrated:
        cv2.circle(frame, calibration_center, calibration_radius, (255, 255, 255), -1)  # White circle for calibration
        cv2.putText(frame, "Touch the circle to start", (calibration_center[0] - 120, calibration_center[1] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    result = hands.process(rgb_frame)

    # If landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb and index finger positions (blue and green circles)
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip (blue circle)
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip (green circle)

            # Get the x, y coordinates of the thumb and index fingers in the image
            x_thumb = int(thumb_tip.x * w)
            y_thumb = int(thumb_tip.y * h)
            x_index = int(index_finger_tip.x * w)
            y_index = int(index_finger_tip.y * h)

            # Draw the thumb (blue circle)
            thumb_radius = 10
            cv2.circle(frame, (x_thumb, y_thumb), thumb_radius, (255, 0, 0), -1)  # Blue circle for thumb

            # Draw the index finger (green circle)
            cursor_radius = 10  # Radius of the cursor (green circle)
            cv2.circle(frame, (x_index, y_index), cursor_radius, (0, 255, 0), -1)  # Green circle for index finger

            # Calculate the distance between thumb and the center of the frame (calibration circle)
            distance_thumb_center = calculate_distance(x_thumb, y_thumb, calibration_center[0], calibration_center[1])

            # Print distance for debugging
            print(f"Distance to center: {distance_thumb_center}")

            # If the thumb touches the calibration circle (within the radius), start the program
            if not calibrated and distance_thumb_center < calibration_radius:
                print("Calibration successful! Starting the program.")
                # Move the cursor to the center of the screen
                mouse_controller.position = (screen_width // 2, screen_height // 2)
                calibrated = True  # Mark as calibrated

            # After calibration is successful, start controlling the mouse with the thumb
            if calibrated:
                # Normalize thumb position to screen coordinates without extra scaling
                x_thumb_screen = int(x_thumb * screen_width / w)
                y_thumb_screen = int(y_thumb * screen_height / h)

                # Move the system mouse pointer to the thumb position (blue circle) directly
                mouse_controller.position = (x_thumb_screen, y_thumb_screen)

    # Show the frame
    cv2.imshow('Hand Tracking', frame)

    # Check for exit key ('q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()