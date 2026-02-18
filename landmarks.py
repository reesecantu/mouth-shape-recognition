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
full_upper_lip_nose = [206, 203, 167, 164, 393, 426, 423]
full_right_eye = [263, 362, 386, 374, 385, 387, 373, 384, 380]
full_left_eye = [33, 133, 159, 145, 160, 158, 157, 144, 161, 153]
full_left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
full_right_eyebrow = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
full_nose = [1, 2, 98, 327, 168, 6, 195, 5, 4, 19, 97, 49, 131, 279, 360, 326]
full_left_cheek = [116, 123, 147, 213, 192, 187]
full_right_cheek = [345, 352, 376, 433, 416, 411]
full_chin = [136, 150, 149, 176, 148, 365, 379, 378, 400, 377, 172, 397, 152]

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

full_parts = [
    (full_outer_top, "outer_top"),
    (full_inner_top, "inner_top"),
    (full_inner_bottom, "inner_bottom"),
    (full_outer_bottom, "outer_bottom"),
    (full_left_eyebrow, "left_eyebrow"),
    (full_right_eyebrow, "right_eyebrow"),
    (full_upper_lip_nose, "upper_lip_nose"),
    (full_left_eye, "left_eye"),
    (full_right_eye, "right_eye"),    
    (full_nose, "nose"),
    (full_left_cheek, "left_cheek"),
    (full_right_cheek, "right_cheek"),
    (full_chin, "chin"),
]

all_landmarks = set(
    full_outer_top + full_inner_top + full_inner_bottom + 
    full_outer_bottom + full_upper_lip_nose + full_left_eye + 
    full_right_eye + full_left_eyebrow + full_right_eyebrow + 
    full_nose + full_left_cheek + full_right_cheek + full_chin
)
all_landmarks = sorted(all_landmarks)

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
  "right_eye": (128, 128, 0),    # Olive
  "nose": (0, 128, 128),        # Teal
  "left_cheek": (128, 128, 128), # Gray
  "right_cheek": (0, 128, 0),    # Dark Green
  "chin": (128, 0, 0),          # Maroon
}