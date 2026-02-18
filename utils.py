
# These classes are used to create fake landmark objects that mimic MediaPipe Face Mesh objects 
# out of the collected data. 

class FakeLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class FakeFaceLandmarks:
    def __init__(self, row):
        self.landmark = {}
        for col in row.index:
            if col.startswith("lm_"):
                parts = col.split("_")
                idx = int(parts[1])
                coord = parts[2]
                if idx not in self.landmark:
                    self.landmark[idx] = FakeLandmark(0, 0, 0)
                setattr(self.landmark[idx], coord, row[col])



LABELS = ["EE", "MM", "OO", "AH", "CHA", "SH", "STA", "SOW", "ARCH", "LIP BUZZ", "OTHER", "NEUTRAL"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}