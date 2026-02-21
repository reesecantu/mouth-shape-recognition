import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import mediapipe as mp
import torch
import numpy as np
import glob
from collections import deque
from utils.model import NMMClassifier
from utils.features import compute_features
from config import HIDDEN_SIZE, INPUT_SIZE, LABELS, IDX_TO_LABEL, WINDOW_SIZE, MODEL_DIR


HIDDEN_LABELS = {"Uncertain", "NEUTRAL", "OTHER"}

# Load most recent model
files = glob.glob(os.path.join(MODEL_DIR, "model_*.pth"))
latest_model = max(files, key=os.path.getmtime)
print(f"Loaded model: {latest_model}")

model = NMMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=len(LABELS),
)
model.load_state_dict(torch.load(latest_model))
model.eval()



# Set up MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh # type: ignore
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=WINDOW_SIZE)
CONFIDENCE_THRESHOLD = 0.7

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        key = cv2.waitKey(5) & 0xFF

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        prediction_text = "Collecting frames..."
        text_color = (255, 255, 255)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                features = compute_features(face_landmarks)
                if features is not None:
                    feature_values = list(features.values())
                    buffer.append(feature_values)
                    if len(buffer) == WINDOW_SIZE:
                        window = np.array(list(buffer))
                        tensor = torch.FloatTensor(window).unsqueeze(0)

                        with torch.no_grad():
                            outputs = model(tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidence, predicted_class = torch.max(probabilities, dim=1)

                        if confidence.item() > CONFIDENCE_THRESHOLD:
                            label = IDX_TO_LABEL[int(predicted_class.item())]
                            if label in HIDDEN_LABELS:
                                prediction_text = label
                            else:
                                prediction_text = f"{label} ({confidence.item():.2f})"
                                text_color = (0, 255, 0)
                        else:
                            prediction_text = "Uncertain"

        if prediction_text not in HIDDEN_LABELS:
            text_x = 30
            text_y = 90
            font_scale = 2.5
            thickness = 5
            padding = 20

            text_size = cv2.getTextSize(
                prediction_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )[0]

            # Rectangle top-left is above the text, bottom-right is below
            cv2.rectangle(
                image,
                (text_x - padding, text_y - text_size[1] - padding),
                (text_x + text_size[0] + padding, text_y + padding),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                image,
                prediction_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        cv2.imshow("NMM Live Detection", image)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()