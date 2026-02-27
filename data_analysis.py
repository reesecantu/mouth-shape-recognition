
import glob
import pandas as pd
import numpy as np
import os
from config import RAW_DATA_DIR


files = glob.glob(os.path.join(RAW_DATA_DIR, "collected_data_*.csv"))
if not files:
    raise FileNotFoundError(f"No collected_data_*.csv files found in {RAW_DATA_DIR}")

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



print(f"{'Label':<12} {'Avg (frames)':>14} {'Min':>6} {'Max':>6}")
print("-" * 42)
for _, row in stats.iterrows():
    print(f"{row['label']:<12} {row['mean']:>7.1f} frames {row['min']:>6} {row['max']:>6}")
