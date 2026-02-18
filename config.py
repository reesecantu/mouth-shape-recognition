import os

# --- Labels ---
LABELS = ["EE", "MM", "OO", "AH", "CHA", "SH", "STA", "SOW", "ARCH", "LIP BUZZ", "OTHER", "NEUTRAL"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# --- Pipeline parameters ---
WINDOW_SIZE = 30
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PREPARED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "prepared")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREPARED_DATA_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)