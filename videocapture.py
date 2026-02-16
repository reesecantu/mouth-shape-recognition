import numpy as np
import cv2
import mediapipe as mp
from landmarks import parts, colors
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh # type: ignore

# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image = cv2.flip(image, 1)
    # Performance improvements, optionally mark the image as not writeable to 
    # pass by reference
    image.flags.writeable = False

    # face mesh expects RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    img_y, img_x, img_channels = image.shape

    # Draw the face mesh annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # Draw relevant landmarks
        # Draw each part with its color
        for indices, name in parts:
          for idx in indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * img_x)
            y = int(landmark.y * img_y)
            cv2.circle(image, (x, y), 3, colors[name], -1)
            cv2.putText(
              image,
              str(idx),
              (x + 4, y - 4),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (255, 255, 255),
              1,
              cv2.LINE_AA
            )
        # calculations
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()