"""
SIH 26034 -- PyTorch 2D Spatial Classifier (GPU Accelerated)
Uses PyTorch with torch.compile (XLA-like optimization) to classify
2D bounding boxes (X, Y, W, H) natively on the RTX 5060.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# --- 1. DATA GENERATION (Simulating 2D Bounding Boxes) ---
def generate_synthetic_data(num_samples=25000):
    np.random.seed(42)
    X, labels = [], []
    for _ in range(num_samples):
        class_id = np.random.randint(0, 5)
        if class_id == 1:   # MRP
            x, y, w, h = np.random.uniform(0.6, 0.95), np.random.uniform(0.7, 0.95), np.random.uniform(0.1, 0.3), np.random.uniform(0.02, 0.05)
            is_num, length = 1.0, np.random.uniform(0.05, 0.15)
        elif class_id == 2: # Expiry
            x, y, w, h = np.random.uniform(0.5, 0.9), np.random.uniform(0.6, 0.9), np.random.uniform(0.1, 0.25), np.random.uniform(0.02, 0.05)
            is_num, length = 0.8, np.random.uniform(0.05, 0.12)
        elif class_id == 3: # Net Qty
            x, y, w, h = np.random.uniform(0.05, 0.3), np.random.uniform(0.7, 0.9), np.random.uniform(0.05, 0.2), np.random.uniform(0.02, 0.06)
            is_num, length = 0.9, np.random.uniform(0.03, 0.1)
        elif class_id == 4: # FSSAI
            x, y, w, h = np.random.uniform(0.1, 0.5), np.random.uniform(0.85, 0.98), np.random.uniform(0.2, 0.4), np.random.uniform(0.02, 0.04)
            is_num, length = 1.0, np.random.uniform(0.14, 0.18)
        else:               # Junk
            x, y, w, h = np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0), np.random.uniform(0.01, 0.5), np.random.uniform(0.01, 0.2)
            is_num, length = np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)

        # Add OCR noise
        x = np.clip(x + np.random.normal(0, 0.02), 0, 1)
        y = np.clip(y + np.random.normal(0, 0.02), 0, 1)
        X.append([x, y, w, h, is_num, length])
        labels.append(class_id)
        
    return np.array(X, dtype=np.float32), np.array(labels, dtype=np.int64)

# --- 2. PYTORCH NEURAL NETWORK ---
class SpatialNet(nn.Module):
    def __init__(self, input_dim=6, num_classes=5):
        super(SpatialNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing PyTorch on Device: {device}")
    
    if device.type == "cuda":
        print(f"[*] GPU Detected: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    print("[*] Generating 25,000 spatial samples...")
    X_np, y_np = generate_synthetic_data()
    
    # Split: 80% Train, 20% Test
    split_idx = int(0.8 * len(X_np))
    X_train_t = torch.tensor(X_np[:split_idx]).to(device)
    y_train_t = torch.tensor(y_np[:split_idx]).to(device)
    X_test_t  = torch.tensor(X_np[split_idx:]).to(device)
    y_test_t  = torch.tensor(y_np[split_idx:]).to(device)

    # Init Model
    model = SpatialNet().to(device)
    
    # Optional: XLA-like graph compilation for speed (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("[*] torch.compile() enabled (XLA-like optimization active).")
    except Exception as e:
        print("[*] Standard eager execution active.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print(f"\n[*] Training 2D Spatial Model on {device.type.upper()}...")
    epochs = 300
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch [{epoch+1:03d}/{epochs}], Loss: {loss.item():.4f}")

    train_time = time.time() - start_time
    
    if device.type == "cuda":
        mem_used = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"\n[*] Peak GPU VRAM Used: {mem_used:.2f} MB")

    print(f"[*] Training complete in {train_time:.2f} seconds.")

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_out = model(X_train_t)
        test_out = model(X_test_t)
        
        train_acc = (torch.argmax(train_out, 1) == y_train_t).float().mean().item()
        test_acc = (torch.argmax(test_out, 1) == y_test_t).float().mean().item()

    print(f"\n--- ACCURACY RESULTS ---")
    print(f"Train Accuracy: {train_acc * 100:.3f}%")
    print(f"Test Accuracy:  {test_acc * 100:.3f}%")
    
    if train_acc >= 0.9899 and test_acc >= 0.9899:
        print("\n[SUCCESS] Achieved target accuracy of >98.99% on GPU!")

if __name__ == '__main__':
    main()
