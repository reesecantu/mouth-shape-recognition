# relavent facial landmarks
outer_top = [61, 39, 185, 0, 269, 409, 291]
inner_top = [78, 81, 191, 13, 311, 415, 308]
inner_bottom = [78, 88, 14, 318, 308]
outer_bottom = [61, 91, 17, 321, 291]
left_eyebrow = [70, 63, 105, 66, 107]
right_eyebrow = [336, 296, 334, 293, 300]
upper_lip_nose = [167, 164, 393]
left_eye = [362, 374, 386, 398, 263]
right_eye = [33, 145, 133, 173, 159]

# Verbose version of landmark list
full_outer_top = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
full_inner_top = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
full_inner_bottom = [78, 88, 87, 14, 317, 318, 308]
full_outer_bottom = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
full_left_eyebrow = [70, 63, 105, 66, 107]
full_right_eyebrow = [336, 296, 334, 293, 300]
full_upper_lip_nose = [206, 203, 167, 164, 393, 426, 423]
full_left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
full_right_eye = [33, 161, 246, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160]


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