import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import pandas as pd
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def main():
    print("[*] JAX Platform:", jax.default_backend().upper())
    print("[*] Loading real product data...")
    
    if not os.path.exists("data/india_products_large.csv"):
        print("[!] Error: data/india_products_large.csv not found.")
        return

    df = pd.read_csv("data/india_products_large.csv")
    df = df.dropna(subset=['product_name', 'category'])
    
    # 1. Clean the data (Drop junk 'undefined' categories)
    df = df[df['category'].str.lower() != 'undefined']
    
    df['text_feature'] = df['product_name'].astype(str) + " " + df['ingredients'].fillna("").astype(str)
    
    # 2. Focus on Top 5 Distinct Categories for High Accuracy
    top_cats = df['category'].value_counts().nlargest(5).index
    df = df[df['category'].isin(top_cats)]
    
    texts = df['text_feature'].values
    labels_raw = df['category'].values
    
    print(f"[*] Training on {len(texts)} cleaned product entries across {len(top_cats)} categories.")
    print(f"[*] Categories: {list(top_cats)}")

    print("[*] Vectorizing text (5000 Features TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', sublinear_tf=True)
    X = vectorizer.fit_transform(texts).toarray()
    
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    X_train_j = jnp.array(X_train)
    y_train_j = jnp.array(y_train)
    X_test_j = jnp.array(X_test)
    y_test_j = jnp.array(y_test)
    
    def init_mlp(key, features):
        keys = random.split(key, 4)
        return {
            'w1': random.normal(keys[0], (features[0], features[1])) * jnp.sqrt(2.0/features[0]),
            'b1': jnp.zeros((features[1],)),
            'w2': random.normal(keys[1], (features[1], features[2])) * jnp.sqrt(2.0/features[1]),
            'b2': jnp.zeros((features[2],))
        }

    def predict(params, x):
        hidden = jnp.maximum(0, jnp.dot(x, params['w1']) + params['b1'])
        return jnp.dot(hidden, params['w2']) + params['b2']

    def loss_fn(params, x, y_true):
        logits = predict(params, x)
        labels_onehot = jax.nn.one_hot(y_true, num_classes=logits.shape[-1])
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_onehot))

    @jit
    def update(params, opt_state, x, y_true):
        loss, grads = value_and_grad(loss_fn)(params, x, y_true)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    key = random.PRNGKey(42)
    num_classes = len(top_cats)
    
    # 3. Expanded Neural Network (512 hidden nodes)
    params = init_mlp(key, [X_train.shape[1], 512, num_classes])
    
    # Cosine decay schedule for better convergence
    schedule = optax.cosine_decay_schedule(init_value=0.05, decay_steps=800)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)
    
    print(f"\n[*] Training Hyper-Optimized JAX model (800 Epochs)...")
    epochs = 400
    start_time = time.time()
    
    for epoch in range(epochs):
        params, opt_state, loss = update(params, opt_state, X_train_j, y_train_j)
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")
            
    print(f"[*] Training complete in {time.time() - start_time:.2f} seconds.")

    model_path = "data/jax_model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump({
            "params": params,
            "vectorizer": vectorizer,
            "label_encoder": le,
            "top_categories": list(top_cats),
        }, file)
    print(f"[*] Model saved to {model_path}")
    
    logits = predict(params, X_test_j)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y_test_j) * 100
    print(f"\n--- ACCURACY ON REAL TEST DATA ---")
    print(f"Accuracy: {acc:.2f}%")

    print("\n--- REAL PREDICTION EXAMPLES ---")
    sample_indices = [0, 1, 2, 3, 4]
    for idx in sample_indices:
        if idx < len(X_test):
            sample_tensor = X_test_j[idx].reshape(1, -1)
            pred_idx = jnp.argmax(predict(params, sample_tensor)).item()
            predicted_category = le.inverse_transform([pred_idx])[0]
            actual_category = le.inverse_transform([y_test[idx]])[0]
            print(f"Product Input: '{texts[idx][:50]}...'")
            print(f"  Predicted: {predicted_category}")
            print(f"  Actual:    {actual_category}\n")

if __name__ == '__main__':
    main()
