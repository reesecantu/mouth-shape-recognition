"""
Step 1: Collect raw landmark data from webcam.
Run this first. Press SPACE to start/stop recording.
Output: data/raw/raw_data_<timestamp>.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import mediapipe as mp
import pandas as pd
import time
from utils.landmarks import all_landmarks
from config import LABELS, RAW_DATA_DIR

mp_face_mesh = mp.solutions.face_mesh #type: ignore
cap = cv2.VideoCapture(0)
is_recording = False
collected_data = []
burst_id = int(time.time())
frame_count = 0
samples = 0

current_label_index = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame while collecting data.")
            continue
        
        image = cv2.flip(image, 1)
        key = cv2.waitKey(5) & 0xFF
        
        ###### STATUS TEXT ######
        if key == ord(' '): 
            is_recording = not is_recording
            if is_recording:
                burst_id += 1
                frame_count = 0
            else:
                samples += 1
        if key == ord('n') and not is_recording:
            current_label_index = (current_label_index + 1) % len(LABELS)
        status_text = f"{'RECORDING' if is_recording else 'WAITING'} {LABELS[current_label_index]} {current_label_index + 1}/{len(LABELS)} | samples: {samples}"
        text_color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.putText(
        image,
        status_text,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        text_color,
        5,
        cv2.LINE_AA
        )
        
        ###### FACE LANDMARKS ######
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        img_y, img_x, img_channels = image.shape


        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if is_recording:
                    row = {
                        "label": LABELS[current_label_index],
                        "burst_id": burst_id,
                        "frame_number": frame_count,
                    }
                    for idx in all_landmarks:
                        lm = face_landmarks.landmark[idx]
                        row[f"lm_{idx}_x"] = lm.x
                        row[f"lm_{idx}_y"] = lm.y
                        row[f"lm_{idx}_z"] = lm.z
                    collected_data.append(row)
                    frame_count += 1                
                
        cv2.imshow('Data Collection', image)
        if key == 27:
            break
    
cap.release()
if collected_data:
    df = pd.DataFrame(collected_data)
    df.to_csv(os.path.join(RAW_DATA_DIR, f"collected_data_{int(time.time())}.csv"), index=False)
    print(f"Saved {len(collected_data)} rows and {samples} samples to {os.path.join(RAW_DATA_DIR, f'collected_data_{int(time.time())}.csv')}")
else:
    print("No data collected.")