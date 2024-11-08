import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
import time
import pyautogui

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,  # Lower detection confidence for speed
    min_tracking_confidence=0.3    # Lower tracking confidence for speed
)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize the webcam feed with reduced resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the pynput mouse controller
mouse_controller = Controller()

# Add a small delay to allow time for the camera to start
time.sleep(2)

frame_count = 0
skip_frames = 3  # Process every 3rd frame
last_x, last_y = None, None
mouse_is_pressed = False  # To track if the left mouse button is being held
right_mouse_is_pressed = False  # To track if the right mouse button is being pressed
last_click_time = 0  # To track time for double-click detection
double_click_threshold = 0.5  # seconds
thumb_index_contact_time = 0  # Track time for thumb-index touch
thumb_index_threshold = 1  # 1 second threshold for holding left-click

# Smoothing factor (adjust as needed)
alpha = 0.3

# Function to smooth the movement with a more gradual smoothing factor
def smooth_move(last, current, alpha=0.3):
    if last is None:
        return current
    return int(alpha * current + (1 - alpha) * last)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Skip frames to reduce processing load
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Flip the frame horizontally for a mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    result = hands.process(rgb_frame)

    # If landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Optionally: Comment this out to speed up processing
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            # Get the tip of the thumb (landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            # Get the tip of the pinky finger (landmark 20)
            pinky_tip = hand_landmarks.landmark[20]

            # Get the x, y coordinates in image space for the fingers
            h, w, c = frame.shape
            x_index = int(index_finger_tip.x * w)
            y_index = int(index_finger_tip.y * h)

            x_thumb = int(thumb_tip.x * w)
            y_thumb = int(thumb_tip.y * h)

            x_pinky = int(pinky_tip.x * w)
            y_pinky = int(pinky_tip.y * h)

            # Normalize the coordinates to screen resolution for all fingers
            x_index_screen = int(x_index * screen_width / w)
            y_index_screen = int(y_index * screen_height / h)

            x_thumb_screen = int(x_thumb * screen_width / w)
            y_thumb_screen = int(y_thumb * screen_height / h)

            x_pinky_screen = int(x_pinky * screen_width / w)
            y_pinky_screen = int(y_pinky * screen_height / h)

            # Move the system mouse pointer incrementally with smoothing
            x_index_screen = smooth_move(last_x, x_index_screen, alpha)
            y_index_screen = smooth_move(last_y, y_index_screen, alpha)

            # Ensure the mouse stays within screen bounds
            x_index_screen = max(0, min(screen_width - 1, x_index_screen))
            y_index_screen = max(0, min(screen_height - 1, y_index_screen))

            # Set the mouse to the calculated screen position (for the index finger)
            mouse_controller.position = (x_index_screen, y_index_screen)

            # Calculate the distance between thumb and index, and between thumb and pinky
            distance_thumb_index = ((x_index_screen - x_thumb_screen) ** 2 + (y_index_screen - y_thumb_screen) ** 2) ** 0.5
            distance_thumb_pinky = ((x_pinky_screen - x_thumb_screen) ** 2 + (y_pinky_screen - y_thumb_screen) ** 2) ** 0.5

            # Calculate hand width to dynamically adjust the threshold for thumb-index and thumb-pinky distance
            hand_width = ((x_pinky_screen - x_thumb_screen) ** 2 + (y_pinky_screen - y_thumb_screen) ** 2) ** 0.5
            dynamic_threshold = hand_width * 0.1  # Set threshold to 10% of hand's width

            # Check if thumb is touching the index finger for left-click
            if distance_thumb_index < dynamic_threshold:  # If the thumb is near the index finger
                if not mouse_is_pressed:
                    mouse_controller.press(Button.left)  # Left button pressed
                    mouse_is_pressed = True

                # Track how long the thumb is in contact with the index for hold-left-click behavior
                if thumb_index_contact_time == 0:  # Only start the timer when touch is first detected
                    thumb_index_contact_time = time.time()

                # Check if the thumb has stayed in contact for more than 1 second
                if time.time() - thumb_index_contact_time > thumb_index_threshold:
                    if not mouse_is_pressed:
                        mouse_controller.press(Button.left)  # Simulate holding left-click
                        mouse_is_pressed = True
            else:
                if mouse_is_pressed:
                    mouse_controller.release(Button.left)  # Left button released
                    mouse_is_pressed = False
                    thumb_index_contact_time = 0  # Reset the contact time when thumb moves away

            # Check if thumb is touching the pinky finger position for right-click
            if distance_thumb_pinky < dynamic_threshold:  # If the thumb is near the pinky finger
                if not right_mouse_is_pressed:
                    mouse_controller.press(Button.right)  # Right button pressed
                    right_mouse_is_pressed = True
            else:
                if right_mouse_is_pressed:
                    mouse_controller.release(Button.right)  # Right button released
                    right_mouse_is_pressed = False

            # Check for double-click
            current_time = time.time()
            if distance_thumb_index < dynamic_threshold:  # Check if thumb and index are near
                if current_time - last_click_time < double_click_threshold:
                    mouse_controller.click(Button.left, 2)  # Double-click detected
                last_click_time = current_time

            # Draw the visual cursor (green circle) for the index finger
            cursor_radius = 10  # Radius of the cursor (circle)
            cv2.circle(frame, (x_index, y_index), cursor_radius, (0, 255, 0), -1)  # Green circle for index

            # Draw the thumb's position (blue circle)
            thumb_radius = 10  # Radius of the thumb cursor
            cv2.circle(frame, (x_thumb, y_thumb), thumb_radius, (255, 0, 0), -1)  # Blue circle for thumb

            # Draw the pinky finger's position (red circle)
            pinky_radius = 10  # Radius of the pinky cursor
            cv2.circle(frame, (x_pinky, y_pinky), pinky_radius, (0, 0, 255), -1)  # Red circle for pinky

            # Update last_x and last_y for next frame
            last_x, last_y = x_index_screen, y_index_screen

    else:
        # No hand detected, reset the mouse position to avoid erratic behavior
        mouse_controller.position = (screen_width // 2, screen_height // 2)

    # Show the frame with the visual cursors
    cv2.imshow('Hand Tracking', frame)

    # Check if the window is responding (or check for the 'q' key press)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
