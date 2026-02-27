
import glob
import pandas as pd
import numpy as np
import os
from config import RAW_DATA_DIR

files = glob.glob(os.path.join(RAW_DATA_DIR, "collected_data_*.csv"))
raw_df = pd.concat([pd.read_csv(f) for f in files])

# Group by burst_id and label, count frames in each burst
burst_length = (
    raw_df.groupby(['label', 'burst_id'])
    .size()
    .reset_index(name='length')
)

# Group by label and calculate min, max, and mean
stats = (
    burst_length.groupby('label')['length']
    .agg(['mean', 'min', 'max'])
    .reset_index()
)

for _, row in stats.iterrows():
    label = row['label']
    avg = row['mean']
    mn = row['min']
    mx = row['max']
    print(f"{label} average length: {avg:.1f} frames, max: {mx}, min: {mn}")