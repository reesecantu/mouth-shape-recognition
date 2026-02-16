import numpy as np

def get_point(face_landmarks, idx):
    lm = face_landmarks.landmark[idx]
    return np.array([lm.x, lm.y])

def compute_features(face_landmarks):
    # Reference distance for normalization (outer eye corners)
    ref_distance = np.linalg.norm(
        get_point(face_landmarks, 33) - get_point(face_landmarks, 263)
    )
    # Avoid division by zero
    if ref_distance == 0:
        return None
    

    # Mouth measurements
    mouth_aperture = np.linalg.norm(
        get_point(face_landmarks, 13) - get_point(face_landmarks, 14)
    ) / ref_distance

    mouth_spread = np.linalg.norm(
        get_point(face_landmarks, 61) - get_point(face_landmarks, 291)
    ) / ref_distance

    aspect_ratio = mouth_aperture / mouth_spread if mouth_spread > 0 else 0

    # Eye openness
    left_eye_openness = np.linalg.norm(
        get_point(face_landmarks, 386) - get_point(face_landmarks, 374)
    ) / ref_distance

    right_eye_openness = np.linalg.norm(
        get_point(face_landmarks, 159) - get_point(face_landmarks, 145)
    ) / ref_distance

    # Eyebrow height (relative to eye corners)
    left_brow_height = (
        get_point(face_landmarks, 105)[1] - get_point(face_landmarks, 263)[1]
    ) / ref_distance

    right_brow_height = (
        get_point(face_landmarks, 334)[1] - get_point(face_landmarks, 33)[1]
    ) / ref_distance

    return {
        "mouth_aperture": mouth_aperture,
        "mouth_spread": mouth_spread,
        "aspect_ratio": aspect_ratio,
        "left_eye_openness": left_eye_openness,
        "right_eye_openness": right_eye_openness,
        "left_brow_height": left_brow_height,
        "right_brow_height": right_brow_height,
    }