import pandas as pd
from features import compute_features
from utils import FakeFaceLandmarks
import glob
import time

files = glob.glob("data/raw/collected_data_*.csv")
raw_df = pd.concat([pd.read_csv(f) for f in files])

# Process each row
processed_rows = []
for idx, row in raw_df.iterrows():
    fake_landmarks = FakeFaceLandmarks(row)
    features = compute_features(fake_landmarks)
    if features is not None:
        features["label"] = row["label"]
        features["burst_id"] = row["burst_id"]
        features["frame_number"] = row["frame_number"]
        processed_rows.append(features)

# Save
processed_df = pd.DataFrame(processed_rows)
processed_df.to_csv(f"data/processed/processed_data_{int(time.time())}.csv", index=False)
print(f"Processed {len(processed_rows)} rows")
print(processed_df["label"].value_counts())
print(processed_df.head())
