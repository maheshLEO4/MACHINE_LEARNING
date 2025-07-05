import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data
X = np.array([2, 4, 6, 8])       # Study hours
Y = np.array([0, 0, 1, 1])       # Passed (0 = Fail, 1 = Pass)

# Step 2: Initialize weights
w = 0.5
b = 0.0
lr = 0.1                        # Learning rate
epochs = 100                   # Number of iterations

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training loop
for epoch in range(epochs):
    z = w * X + b
    p = sigmoid(z)

    # Cross-entropy loss
    loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    # Gradients
    dw = np.mean((p - Y) * X)
    db = np.mean(p - Y)

    # Update parameters
    w -= lr * dw
    b -= lr * db

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

# Plotting the sigmoid curve
x_vals = np.linspace(0, 10, 100)
y_vals = sigmoid(w * x_vals + b)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label="Sigmoid Curve")
plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary')
plt.scatter(X, Y, color='red', label='Training Data')

plt.title("Logistic Regression: Study Hours vs Pass Probability")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.legend()
plt.grid(True)
plt.show()

# Prediction function
def predict(x):
    prob = sigmoid(w * x + b)
    return 1 if prob >= 0.5 else 0

# Take input from user
h = float(input("\nEnter study hours: "))

# Predict
prediction = predict(h)
print(f"\nStudy Hours: {h} => Prediction: {'Pass' if prediction == 1 else 'Fail'}")
