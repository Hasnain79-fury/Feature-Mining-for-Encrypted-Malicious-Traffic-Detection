import os
import numpy as np
import torch, torch.nn as nn, torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader

# Resolve paths relative to the project root (not the script location)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure output directories exist
os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'probs'),  exist_ok=True)

# Load pre-processed tensors (already scaled by the ResNet MinMaxScaler)
X_train = np.load(os.path.join(BASE, 'tensors/X_resnet_train.npy'))  # (N, 1, 38, 38)
X_val   = np.load(os.path.join(BASE, 'tensors/X_resnet_val.npy'))
X_test  = np.load(os.path.join(BASE, 'tensors/X_resnet_test.npy'))
y_train = np.load(os.path.join(BASE, 'tensors/y_train.npy'))
y_val   = np.load(os.path.join(BASE, 'tensors/y_val.npy'))

# Convert to torch tensors
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset   = TensorDataset(torch.from_numpy(X_val).float(),   torch.from_numpy(y_val).long())
loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
loader_val   = DataLoader(val_dataset, batch_size=32)

# ResNet-34 adaptation for 1-channel 38x38 images (as described in the paper §5.2)
net = models.resnet34(pretrained=False)
net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 1-channel input
net.maxpool = nn.Identity()  # remove aggressive pooling for small images
net.fc = nn.Linear(512, 2)   # binary classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def train_one_epoch():
    net.train()
    epoch_loss = 0.0
    for xb, yb in loader_train:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = net(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader_train)

def get_probs(loader):
    """Get softmax probabilities for an entire DataLoader (labels ignored)."""
    net.eval()
    all_probs = []
    with torch.no_grad():
        for xb, *_ in loader:
            xb = xb.to(device)
            logits = net(xb)
            all_probs.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)

def evaluate():
    net.eval()
    correct, total = 0, 0
    probs = []
    with torch.no_grad():
        for xb, yb in loader_val:
            xb = xb.to(device)
            logits = net(xb)
            probs.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
            preds = logits.argmax(dim=1).cpu().numpy()
            correct += (preds == yb.numpy()).sum()
            total += yb.size(0)
    acc = correct / total
    return acc, np.concatenate(probs, axis=0)

best_acc = 0.0
patience = 5
wait = 0
for epoch in range(1, 101):
    train_loss = train_one_epoch()
    val_acc, val_probs = evaluate()
    print(f'Epoch {epoch:03d} – Train loss: {train_loss:.4f} – Val acc: {val_acc:.4f}')
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), os.path.join(BASE, 'models/resnet34.pt'))
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print('Early stopping')
            break

# Load best model for inference
net.load_state_dict(torch.load(os.path.join(BASE, 'models/resnet34.pt'), map_location=device))

# Save probabilities for Layer-2 ensemble (train, val, AND test)
train_probs = get_probs(loader_train)
# Re-evaluate val to get probs from best model
_, val_probs = evaluate()
# Test set
test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.zeros(len(X_test)).long())
loader_test  = DataLoader(test_dataset, batch_size=32)
test_probs   = get_probs(loader_test)

np.save(os.path.join(BASE, 'probs/resnet_probs_train.npy'), train_probs)
np.save(os.path.join(BASE, 'probs/resnet_probs_val.npy'),   val_probs)
np.save(os.path.join(BASE, 'probs/resnet_probs_test.npy'),  test_probs)

print('✅ ResNet training complete. Model saved to models/resnet34.pt')
