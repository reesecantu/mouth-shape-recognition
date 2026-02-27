"""
Step 2: Compute features from raw landmark data.
Input:  data/raw/collected_data_*.csv
Output: data/processed/processed_data_<timestamp>.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import glob
import time
from utils.features import compute_features
from utils.face_landmark_struct import FakeFaceLandmarks
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

files = glob.glob(os.path.join(RAW_DATA_DIR, "collected_data_*.csv"))
raw_df = pd.concat([pd.read_csv(f) for f in files])

# Process each row, compute features
processed_rows = []
for idx, row in raw_df.iterrows():
    fake_landmarks = FakeFaceLandmarks(row)
    features = compute_features(fake_landmarks)
    if features is not None:
        features["label"] = row["label"]
        features["burst_id"] = row["burst_id"]
        features["frame_number"] = row["frame_number"]
        processed_rows.append(features)

processed_df = pd.DataFrame(processed_rows)
processed_df.to_csv(os.path.join(PROCESSED_DATA_DIR, f"processed_data_{int(time.time())}.csv"), index=False)
print(f"Processed {len(processed_rows)} rows")
print(f"\nBursts per class:")
print(processed_df.groupby("label")["burst_id"].nunique())
print(processed_df.head())
