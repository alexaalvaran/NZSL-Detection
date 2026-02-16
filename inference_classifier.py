import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# -------------------------------
# Load model
# -------------------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
mp_hands_connections = mp.tasks.vision.HandLandmarksConnections

MODEL_PATH = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# -------------------------------
# Global frame storage
# -------------------------------
latest_frame = None

def extract_hand_landmarks(hand_landmarks):
    x_coords = [lm.x for lm in hand_landmarks]
    y_coords = [lm.y for lm in hand_landmarks]
    x_min, y_min = min(x_coords), min(y_coords)
    data_aux = []
    for lm in hand_landmarks:
        data_aux.append(lm.x - x_min)
        data_aux.append(lm.y - y_min)
    return data_aux, x_coords, y_coords

# -------------------------------
# Callback: update global frame
# -------------------------------
def results_callback(result, input_image, timestamp_ms):
    global latest_frame
    frame = input_image.numpy_view()  # RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    H, W, _ = frame_bgr.shape

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands_connections.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in result.hand_landmarks:
            data_aux, x_coords, y_coords = extract_hand_landmarks(hand_landmarks)
            x1 = int(min(x_coords) * W) - 10
            y1 = int(min(y_coords) * H) - 10
            x2 = int(max(x_coords) * W) + 10
            y2 = int(max(y_coords) * H) + 10

            prediction = model.predict([np.array(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame_bgr, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    latest_frame = frame_bgr  # store for main thread display

# -------------------------------
# Create detector
# -------------------------------
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=results_callback
)
detector = vision.HandLandmarker.create_from_options(options)
start_time = time.time()

# -------------------------------
# Main loop: send frames & display
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)

    detector.detect_async(mp_image, timestamp_ms)

    # Thread-safe display
    if latest_frame is not None:
        cv2.imshow('Hand Prediction', latest_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
