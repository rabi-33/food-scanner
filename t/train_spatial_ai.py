"""
SIH 26034 -- Lightweight 2D Spatial AI
Uses a lightweight Neural Network to classify OCR bounding boxes 
based on their 2D spatial layout (X, Y, Width, Height, Type).
Memory footprint: < 50MB.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

def generate_synthetic_data(num_samples=25000):
    """
    Generates synthetic bounding boxes mimicking standard Indian food label layouts.
    Classes: 0: Junk, 1: MRP, 2: Expiry/Mfg Date, 3: Net Qty, 4: FSSAI
    Features: [x_center, y_center, width, height, is_numeric, text_length]
    """
    np.random.seed(42)
    X = []
    labels = []
    
    for _ in range(num_samples):
        class_id = np.random.randint(0, 5)
        
        if class_id == 1: # MRP (bottom right, numeric, short)
            x_c = np.random.uniform(0.6, 0.95)
            y_c = np.random.uniform(0.7, 0.95)
            w = np.random.uniform(0.1, 0.3)
            h = np.random.uniform(0.02, 0.05)
            is_num = 1.0
            t_len = np.random.uniform(0.05, 0.15)
            
        elif class_id == 2: # Expiry Date (near MRP)
            x_c = np.random.uniform(0.5, 0.9)
            y_c = np.random.uniform(0.6, 0.9)
            w = np.random.uniform(0.1, 0.25)
            h = np.random.uniform(0.02, 0.05)
            is_num = 0.8
            t_len = np.random.uniform(0.05, 0.12)
            
        elif class_id == 3: # Net Qty (top left or bottom left)
            x_c = np.random.uniform(0.05, 0.3)
            y_c = np.random.uniform(0.7, 0.9)
            w = np.random.uniform(0.05, 0.2)
            h = np.random.uniform(0.02, 0.06)
            is_num = 0.9
            t_len = np.random.uniform(0.03, 0.1)
            
        elif class_id == 4: # FSSAI (bottom, long 14 digits)
            x_c = np.random.uniform(0.1, 0.5)
            y_c = np.random.uniform(0.85, 0.98)
            w = np.random.uniform(0.2, 0.4)
            h = np.random.uniform(0.02, 0.04)
            is_num = 1.0
            t_len = np.random.uniform(0.14, 0.18)
            
        else: # Background / Junk
            x_c = np.random.uniform(0.0, 1.0)
            y_c = np.random.uniform(0.0, 1.0)
            w = np.random.uniform(0.01, 0.5)
            h = np.random.uniform(0.01, 0.2)
            is_num = np.random.uniform(0.0, 1.0)
            t_len = np.random.uniform(0.0, 1.0)

        # Add slight noise to simulate messy OCR
        x_c = np.clip(x_c + np.random.normal(0, 0.02), 0, 1)
        y_c = np.clip(y_c + np.random.normal(0, 0.02), 0, 1)
        
        X.append([x_c, y_c, w, h, is_num, t_len])
        labels.append(class_id)
        
    return np.array(X, dtype=np.float32), np.array(labels, dtype=np.int32)

def main():
    print("[*] Generating 25,000 spatial samples...")
    X, y = generate_synthetic_data(25000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[*] Training Lightweight Spatial Neural Network...")
    start_time = time.time()
    
    # 3-layer MLP (similar to the JAX architecture but runs purely on CPU in MBs of RAM)
    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), 
                          activation='relu', 
                          solver='adam', 
                          max_iter=300, 
                          random_state=42)
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"[*] Training complete in {train_time:.2f} seconds.")
    
    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"\n--- ACCURACY RESULTS ---")
    print(f"Train Accuracy: {train_acc * 100:.3f}%")
    print(f"Test Accuracy:  {test_acc * 100:.3f}%")
    
    if train_acc >= 0.9899 and test_acc >= 0.9899:
        print("\n[SUCCESS] Achieved target accuracy of >98.99%!")
    else:
        print("\n[WARNING] Did not hit 98.99% target. Need to adjust layers.")

if __name__ == '__main__':
    main()
