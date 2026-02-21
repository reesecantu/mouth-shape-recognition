import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import time
from utils.model import NMMClassifier
from config import LABELS, PREPARED_DATA_DIR, MODEL_DIR, INPUT_SIZE, HIDDEN_SIZE, NUM_EPOCHS, BATCH_SIZE
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import glob

# create model path with timestamp
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pth")

# Load latest prepared data
files = glob.glob(os.path.join(PREPARED_DATA_DIR, "prepared_data_*"))
latest_file = max(files, key=os.path.getmtime)
data = np.load(latest_file)
X_train = torch.FloatTensor(data["X_train"])
y_train = torch.LongTensor(data["y_train"])
X_test = torch.FloatTensor(data["X_test"])
y_test = torch.LongTensor(data["y_test"])

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = NMMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=len(LABELS),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_test_accuracy = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    train_accuracy = correct / total
    avg_loss = total_loss / len(train_loader)

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions = torch.argmax(outputs, dim=1)
            test_correct += (predictions == batch_y).sum().item()
            test_total += batch_y.size(0)

    test_accuracy = test_correct / test_total

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), model_path)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

print(f"\nBest test accuracy: {best_test_accuracy:.4f}")
print("Loading best model for evaluation...")

# Load best model and generate confusion matrix
model.load_state_dict(torch.load(model_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

all_label_indices = list(range(len(LABELS)))

print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=LABELS,
    labels=all_label_indices,
))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds, labels=all_label_indices))