import pandas as pd
import numpy as np
import glob
import os
from utils import LABELS, LABEL_TO_IDX, IDX_TO_LABEL
from sklearn.model_selection import train_test_split

files = glob.glob("data/processed/processed_data_*")
latest_file = max(files, key=os.path.getmtime)
df = pd.read_csv(latest_file)
print(f"Loaded: {latest_file}")


WINDOW_SIZE = 30

windows = []
window_labels = []
window_burst_ids = []

for burst_id, burst_df in df.groupby("burst_id"):
    burst_df = burst_df.sort_values("frame_number")
    label = burst_df["label"].iloc[0]

    # Get just the feature columns (drop label, burst_id, frame_number)
    feature_cols = [col for col in burst_df.columns 
                    if col not in ["label", "burst_id", "frame_number"]]
    features = burst_df[feature_cols].values

    # Skip bursts shorter than window size
    if len(features) < WINDOW_SIZE:
        print(f"Skipping burst {burst_id}: only {len(features)} frames")
        continue

    # Slide the window
    for start in range(len(features) - WINDOW_SIZE + 1):
        window = features[start:start + WINDOW_SIZE]
        windows.append(window)
        window_labels.append(LABEL_TO_IDX[label])
        window_burst_ids.append(burst_id)
        
X = np.array(windows)
y = np.array(window_labels)
burst_ids = np.array(window_burst_ids)

print(f"Total windows: {X.shape[0]}")
print(f"Window shape: {X.shape[1:]}")
print(f"Classes: {len(LABELS)}")


unique_bursts = np.unique(burst_ids)

# Get the label for each burst
burst_labels = []
for b in unique_bursts:
    mask = burst_ids == b
    burst_labels.append(y[mask][0])
burst_labels = np.array(burst_labels)

# Print bursts per class
print("\nBursts per class:")
for idx, label in enumerate(LABELS):
    count = np.sum(burst_labels == idx)
    if count > 0:
        print(f"  {label}: {count} bursts")

train_bursts, test_bursts = train_test_split(
    unique_bursts, 
    test_size=0.2, 
    random_state=42,
    stratify=burst_labels,
)

train_mask = np.isin(burst_ids, train_bursts)
test_mask = np.isin(burst_ids, test_bursts)

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"Training windows: {len(X_train)}")
print(f"Test windows: {len(X_test)}")

np.savez(
    "data/prepared_data.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

