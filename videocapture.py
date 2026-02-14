import numpy as np
import cv2
import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh # type: ignore

# relavent facial landmarks
# see bin.py for more robust landmarks
outer_top = [61, 39, 185, 0, 269, 409, 291]
inner_top = [78, 81, 191, 13, 311, 415, 308]
inner_bottom = [78, 88, 14, 318, 308]
outer_bottom = [61, 91, 17, 321, 291]
left_eyebrow = [70, 63, 105, 66, 107]
right_eyebrow = [336, 296, 334, 293, 300]
upper_lip_nose = [167, 164, 393]
left_eye = [362, 374, 386, 398, 263]
right_eye = [33, 145, 133, 173, 159]


parts = [
  (outer_top, "outer_top"),
  (inner_top, "inner_top"),
  (inner_bottom, "inner_bottom"),
  (outer_bottom, "outer_bottom"),
  (left_eyebrow, "left_eyebrow"),
  (right_eyebrow, "right_eyebrow"),
  (upper_lip_nose, "upper_lip_nose"),
  (left_eye, "left_eye"),
  (right_eye, "right_eye"),
]

# Define colors for each part
colors = {
  "outer_top": (0, 255, 0),      # Green
  "inner_top": (255, 0, 0),      # Blue
  "inner_bottom": (0, 0, 255),   # Red
  "outer_bottom": (0, 255, 255),  # Yellow
  "left_eyebrow": (255, 0, 255),  # Magenta
  "right_eyebrow": (255, 165, 0),  # Orange
  "upper_lip_nose": (128, 0, 128),  # Purple
  "left_eye": (255, 255, 0),     # Cyan
  "right_eye": (128, 128, 0),    # Olive"
}

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
        # Get the two landmarks using their raw normalized coordinates
        top_eyelid = face_landmarks.landmark[159]
        bottom_eyelid = face_landmarks.landmark[145]

        # Compute the distance between them
        eye_openness = np.sqrt(
            (top_eyelid.x - bottom_eyelid.x) ** 2 + 
            (top_eyelid.y - bottom_eyelid.y) ** 2
        )

        # Get the reference distance for normalization (eye corners)
        left_corner = face_landmarks.landmark[33]
        right_corner = face_landmarks.landmark[133]
        eye_width = np.sqrt(
            (left_corner.x - right_corner.x) ** 2 + 
            (left_corner.y - right_corner.y) ** 2
        )

        # Normalize
        normalized_eye_openness = eye_openness / eye_width

        print(f"Eye openness: {normalized_eye_openness:.3f}")
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()