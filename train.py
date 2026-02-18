import torch
import torch.nn as nn
import numpy as np
from model import NMMClassifier
from utils import LABELS
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Load prepared data
data = np.load("data/prepared_data.npz")
X_train = torch.FloatTensor(data["X_train"])
y_train = torch.LongTensor(data["y_train"])
X_test = torch.FloatTensor(data["X_test"])
y_test = torch.LongTensor(data["y_test"])

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

BATCH_SIZE = 32

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = NMMClassifier(
    input_size=22,
    hidden_size=128,
    num_classes=len(LABELS),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 50
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
        torch.save(model.state_dict(), "data/trained_model.pth")

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

print(f"\nBest test accuracy: {best_test_accuracy:.4f}")
print("Loading best model for evaluation...")

# Load best model and generate confusion matrix
model.load_state_dict(torch.load("data/trained_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=LABELS
))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))