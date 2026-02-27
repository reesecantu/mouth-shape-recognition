import numpy as np
'''

'''



# ============ Helper Functions ============

def get_point(face_landmarks, idx):
    """Get normalized x, y coordinates for a landmark index."""
    lm = face_landmarks.landmark[idx]
    return np.array([lm.x, lm.y])

def shoelace_area(points):
    """Compute area of a polygon from ordered vertices."""
    n = len(points)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2

# ============ Feature Computation ============

def compute_features(face_landmarks):
    """Compute normalized facial features from MediaPipe landmarks."""

    # Reference distance for normalization (outer eye corners)
    ref_distance = np.linalg.norm(
        get_point(face_landmarks, 33) - get_point(face_landmarks, 263)
    )
    if ref_distance == 0:
        return None

    # --- Mouth shape ---
    mouth_aperture = np.linalg.norm(
        get_point(face_landmarks, 13) - get_point(face_landmarks, 14)
    ) / ref_distance

    mouth_spread = np.linalg.norm(
        get_point(face_landmarks, 61) - get_point(face_landmarks, 291)
    ) / ref_distance

    mouth_ratio = mouth_aperture / mouth_spread if mouth_spread > 0 else 0

    upper_lip_thickness = np.linalg.norm(
        get_point(face_landmarks, 0) - get_point(face_landmarks, 13)
    ) / ref_distance

    lower_lip_thickness = np.linalg.norm(
        get_point(face_landmarks, 17) - get_point(face_landmarks, 14)
    ) / ref_distance

    inner_lip_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 318, 317, 14, 87, 88]
    inner_lip_points = [get_point(face_landmarks, idx) for idx in inner_lip_indices]
    mouth_area = shoelace_area(inner_lip_points) / (ref_distance ** 2)

    # --- Eyes ---
    left_eye_openness = np.linalg.norm(
        get_point(face_landmarks, 386) - get_point(face_landmarks, 374)
    ) / ref_distance

    right_eye_openness = np.linalg.norm(
        get_point(face_landmarks, 159) - get_point(face_landmarks, 145)
    ) / ref_distance

    avg_eye_openness = (left_eye_openness + right_eye_openness) / 2

    # --- Eyebrows ---
    left_brow_height = (
        get_point(face_landmarks, 334)[1] - get_point(face_landmarks, 263)[1]
    ) / ref_distance

    right_brow_height = (
        get_point(face_landmarks, 105)[1] - get_point(face_landmarks, 33)[1]
    ) / ref_distance

    left_brow_to_eye = np.linalg.norm(
        get_point(face_landmarks, 334) - get_point(face_landmarks, 386)
    ) / ref_distance

    right_brow_to_eye = np.linalg.norm(
        get_point(face_landmarks, 105) - get_point(face_landmarks, 159)
    ) / ref_distance

    # --- Nose and cheeks ---
    nostril_width = np.linalg.norm(
        get_point(face_landmarks, 49) - get_point(face_landmarks, 279)
    ) / ref_distance

    nose_bridge_width = np.linalg.norm(
        get_point(face_landmarks, 98) - get_point(face_landmarks, 327)
    ) / ref_distance

    nose_scrunch_ratio = nostril_width / nose_bridge_width if nose_bridge_width > 0 else 0

    left_cheek_height = np.linalg.norm(
        get_point(face_landmarks, 187) - get_point(face_landmarks, 33)
    ) / ref_distance

    right_cheek_height = np.linalg.norm(
        get_point(face_landmarks, 411) - get_point(face_landmarks, 263)
    ) / ref_distance

    left_nasolabial = np.linalg.norm(
        get_point(face_landmarks, 206) - get_point(face_landmarks, 61)
    ) / ref_distance

    right_nasolabial = np.linalg.norm(
        get_point(face_landmarks, 426) - get_point(face_landmarks, 291)
    ) / ref_distance

    # --- Head orientation ---
    nose_to_bridge_vertical = (
        get_point(face_landmarks, 4)[1] - get_point(face_landmarks, 168)[1]
    ) / ref_distance

    face_height = np.linalg.norm(
        get_point(face_landmarks, 152) - get_point(face_landmarks, 168)
    ) / ref_distance

    eye_center_x = (
        get_point(face_landmarks, 33)[0] + get_point(face_landmarks, 263)[0]
    ) / 2
    nose_tip_x = get_point(face_landmarks, 4)[0]
    head_turn = (nose_tip_x - eye_center_x) / ref_distance

    # --- Return all features ---
    return {
        # Mouth
        "mouth_aperture": mouth_aperture,
        "mouth_spread": mouth_spread,
        "mouth_ratio": mouth_ratio,
        "upper_lip_thickness": upper_lip_thickness,
        "lower_lip_thickness": lower_lip_thickness,
        "mouth_area": mouth_area,
        # Eyes
        "left_eye_openness": left_eye_openness,
        "right_eye_openness": right_eye_openness,
        "avg_eye_openness": avg_eye_openness,
        # Eyebrows
        "left_brow_height": left_brow_height,
        "right_brow_height": right_brow_height,
        "left_brow_to_eye": left_brow_to_eye,
        "right_brow_to_eye": right_brow_to_eye,
        # Nose and cheeks
        "nostril_width": nostril_width,
        "nose_scrunch_ratio": nose_scrunch_ratio,
        "left_cheek_height": left_cheek_height,
        "right_cheek_height": right_cheek_height,
        "left_nasolabial": left_nasolabial,
        "right_nasolabial": right_nasolabial,
        # Head orientation
        "nose_to_bridge_vertical": nose_to_bridge_vertical,
        "face_height": face_height,
        "head_turn": head_turn,
    }