Here’s a **Python implementation** of **Full Batch, Stochastic, and Mini-Batch Gradient Descent** using a simple linear regression example.  

---

### **🔹 Problem Setup:**

- We have a dataset with **1,000 examples** of input \( X \) and output \( Y \).
- Our goal is to learn the best weight \( w \) and bias \( b \) to minimize the loss.
- We use **Mean Squared Error (MSE) Loss**:
  \[
  L = \frac{1}{n} \sum (y_{\text{true}} - y_{\text{pred}})^2
  \]
- We compare **Full Batch, SGD, and Mini-Batch Gradient Descent**.

---

### **🔹 Python Code Implementation**

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (y = 2x + 3 + noise)
np.random.seed(42)
X = np.random.rand(1000, 1)  # 1000 data points
Y = 2 * X + 3 + np.random.randn(1000, 1) * 0.1  # Adding noise

# Hyperparameters
learning_rate = 0.1
epochs = 20
batch_size = 100  # Used for Mini-Batch GD

# Initialize weight and bias
w, b = np.random.randn(), np.random.randn()

# Function to compute gradient of the loss
def compute_gradient(X, Y, w, b):
    n = len(X)
    y_pred = w * X + b
    error = y_pred - Y
    dw = (2 / n) * np.sum(error * X)
    db = (2 / n) * np.sum(error)
    return dw, db

# 1️⃣ Full Batch Gradient Descent
def full_batch_gd(X, Y, learning_rate, epochs):
    w, b = np.random.randn(), np.random.randn()
    history = []

    for epoch in range(epochs):
        dw, db = compute_gradient(X, Y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        history.append((w, b))
    
    return w, b, history

# 2️⃣ Stochastic Gradient Descent (SGD)
def stochastic_gd(X, Y, learning_rate, epochs):
    w, b = np.random.randn(), np.random.randn()
    history = []

    for epoch in range(epochs):
        for i in range(len(X)):  # Update after each example
            dw, db = compute_gradient(X[i:i+1], Y[i:i+1], w, b)
            w -= learning_rate * dw
            b -= learning_rate * db
        history.append((w, b))
    
    return w, b, history

# 3️⃣ Mini-Batch Gradient Descent
def mini_batch_gd(X, Y, learning_rate, epochs, batch_size):
    w, b = np.random.randn(), np.random.randn()
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(len(X))  # Shuffle data
        X_shuffled, Y_shuffled = X[indices], Y[indices]

        for i in range(0, len(X), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            dw, db = compute_gradient(X_batch, Y_batch, w, b)
            w -= learning_rate * dw
            b -= learning_rate * db

        history.append((w, b))
    
    return w, b, history

# Run and compare results
w_full, b_full, hist_full = full_batch_gd(X, Y, learning_rate, epochs)
w_sgd, b_sgd, hist_sgd = stochastic_gd(X, Y, learning_rate, epochs)
w_mini, b_mini, hist_mini = mini_batch_gd(X, Y, learning_rate, epochs, batch_size)

# Plot Convergence
epochs_range = np.arange(1, epochs + 1)
plt.plot(epochs_range, [h[0] for h in hist_full], label="Full Batch GD")
plt.plot(epochs_range, [h[0] for h in hist_sgd], label="SGD")
plt.plot(epochs_range, [h[0] for h in hist_mini], label="Mini-Batch GD")

plt.xlabel("Epochs")
plt.ylabel("Weight Value (w)")
plt.title("Comparison of Gradient Descent Methods")
plt.legend()
plt.show()

# Print final weights and bias
print(f"Full Batch GD: w = {w_full:.4f}, b = {b_full:.4f}")
print(f"SGD: w = {w_sgd:.4f}, b = {b_sgd:.4f}")
print(f"Mini-Batch GD: w = {w_mini:.4f}, b = {b_mini:.4f}")
```

---

### **🔹 Explanation of the Code**

1. **Generating Synthetic Data**  
   - We create \( X \) values between 0 and 1.
   - We compute \( Y = 2X + 3 + \) some **random noise**.

2. **Three Gradient Descent Implementations**  
   - **Full Batch Gradient Descent** → Updates once per epoch after computing the gradient for the entire dataset.
   - **Stochastic Gradient Descent (SGD)** → Updates after each individual example.
   - **Mini-Batch Gradient Descent** → Updates after every small batch of size `batch_size = 100`.

3. **Convergence Plot**  
   - We compare how the weight \( w \) evolves over epochs.
   - **Full Batch GD** → Smoothest, but slow to update.
   - **SGD** → Noisy updates, fluctuates a lot.
   - **Mini-Batch GD** → Balanced and efficient.

---

### **🔹 Expected Results**

- **Full Batch GD** converges **slowly but smoothly**.
- **SGD** fluctuates a lot (since updates are random).
- **Mini-Batch GD** balances speed and stability.

Would you like me to explain any part in more detail? 🚀
