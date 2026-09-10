import torch
"""
SIH 26034 -- Real Data NLP Classifier (PyTorch + CUDA Eager Mode)
Solves Rule 6(1)(b): Generic Commodity Name Identification
Trains natively on the RTX 5060 using the real Open Food Facts CSV data.
Eager mode execution ensures compatibility with RTX 5060 on Windows.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import os

def main():
    print(f"[*] PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device}")
    
    if device.type == "cuda":
        print(f"[*] GPU Detected: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # 1. Load Real Data
    print("\n[*] Loading real product data...")
    if not os.path.exists("data/india_products_all.csv"):
        print("[!] Error: data/india_products_all.csv not found.")
        return
        
    df = pd.read_csv("data/india_products_all.csv")

    # Filter out rows missing required text
    df = df.dropna(subset=['product_name', 'category'])
    
    # Create input feature: Product Name + Ingredients
    df['text_feature'] = df['product_name'].astype(str) + " " + df['ingredients'].fillna("").astype(str)
    
    # Get top 20 categories to ensure we have enough samples per class
    top_categories = df['category'].value_counts().nlargest(20).index
    df = df[df['category'].isin(top_categories)]
    
    texts = df['text_feature'].values
    labels_raw = df['category'].values
    
    print(f"[*] Training on {len(texts)} real product entries across {len(top_categories)} categories.")

    # 2. Vectorize Text
    print("[*] Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()
    
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch Tensors and MOVE TO GPU
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    # 3. Define PyTorch Model (Eager mode, no torch.compile)
    class CategoryClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
            
        def forward(self, x):
            return self.net(x)

    input_dim = X_train_t.shape[1]
    num_classes = len(top_categories)
    model = CategoryClassifier(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 4. Training Loop
    print(f"\n[*] Training Model on {device.type.upper()}...")
    epochs = 150
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    train_time = time.time() - start_time
    
    if device.type == "cuda":
        mem_used_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"\n[*] Peak GPU VRAM Used: {mem_used_mb:.2f} MB")

    print(f"[*] Training complete in {train_time:.2f} seconds.")

    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs.data, 1)
        correct = (predicted == y_test_t).sum().item()
        accuracy = (correct / len(y_test_t)) * 100

    print(f"\n--- ACCURACY ON REAL TEST DATA ---")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Show a real prediction example
    print("\n--- REAL PREDICTION EXAMPLES ---")
    sample_indices = [0, 1, 2]
    for idx in sample_indices:
        if idx < len(X_test):
            sample_tensor = X_test_t[idx].unsqueeze(0)
            pred_idx = torch.argmax(model(sample_tensor)).item()
            predicted_category = le.inverse_transform([pred_idx])[0]
            actual_category = le.inverse_transform([y_test[idx]])[0]
            print(f"Product Input: '{texts[idx][:50]}...'")
            print(f"  Predicted: {predicted_category}")
            print(f"  Actual:    {actual_category}\n")

if __name__ == '__main__':
    main()
