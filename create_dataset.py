import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = './data'
MODEL_PATH = 'hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

data = []
labels = []

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        mp_image = mp.Image.create_from_file(img_path)

        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]

                x_min = min(x_coords)
                y_min = min(y_coords)

                data_aux = []
                for lm in hand_landmarks:
                    data_aux.append(lm.x - x_min)
                    data_aux.append(lm.y - y_min)

                data.append(data_aux)
                labels.append(class_name)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Processed {len(data)} hands from dataset.")
