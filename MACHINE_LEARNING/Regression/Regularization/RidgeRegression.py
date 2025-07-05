import numpy as np
import matplotlib.pyplot as plt

# Step 1: Dataset with irrelevant feature
data = {
    'EngineSize': [1.0, 1.6, 2.0, 2.4, 3.0, 3.5, 4.0, 4.2, 4.5, 5.0],
    'Cylinders':  [4,   4,   4,   6,   6,   6,   8,   8,   8,   8],
    'FuelRate':   [6.5, 7.0, 8.0, 9.0, 10.0, 11.5, 12.5, 13.0, 14.0, 15.0],
    'Noise':      [7,   3,   8,   2,   4,   5,    6,    9,    1,    0],
    'CO2':        [130, 145, 160, 175, 190, 210, 230, 240, 260, 280]
}

# Step 2: Convert to NumPy arrays
X = np.array([ [row[i] for i in range(4)] for row in zip(*data.values()) ])
y = np.array(data['CO2'])

# Step 3: Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Step 4: Parameters
n, d = X.shape
alpha = 0.01
epochs = 5000
lambda_l2 = 0.1 # Regularization strength

# ----------------------------- #
# 🟩 BEFORE RIDGE: No Regularization
# ----------------------------- #
m_plain = np.zeros(d)
b_plain = 0

for _ in range(epochs):
    y_pred = np.dot(X, m_plain) + b_plain
    error = y - y_pred

    dm = (-2/n) * np.dot(X.T, error)
    db = (-2/n) * np.sum(error)

    m_plain -= alpha * dm
    b_plain -= alpha * db

y_pred_plain = np.dot(X, m_plain) + b_plain

# ----------------------------- #
# 🟦 AFTER RIDGE: L2 Regularization
# ----------------------------- #
m_ridge = np.zeros(d)
b_ridge = 0

for _ in range(epochs):
    y_pred = np.dot(X, m_ridge) + b_ridge
    error = y - y_pred

    # L2: Add λ * w to gradient (not sign)
    dm = (-2/n) * np.dot(X.T, error) + 2 * lambda_l2 * m_ridge
    db = (-2/n) * np.sum(error)

    m_ridge -= alpha * dm
    b_ridge -= alpha * db

y_pred_ridge = np.dot(X, m_ridge) + b_ridge

# ----------------------------- #
# 🎯 Print Results
# ----------------------------- #
features = ['EngineSize', 'Cylinders', 'FuelRate', 'Noise']
print("\n📊 Coefficients Comparison:\n")
print(f"{'Feature':<15} | {'Before Ridge':>15} | {'After Ridge':>15}")
print("-" * 50)
for i in range(d):
    print(f"{features[i]:<15} | {m_plain[i]:>15.4f} | {m_ridge[i]:>15.4f}")
print("-" * 50)
print(f"{'Intercept':<15} | {b_plain:>15.4f} | {b_ridge:>15.4f}")

# ----------------------------- #
# 📈 Plotting
# ----------------------------- #
plt.figure(figsize=(8, 5))
plt.plot(y, label="Actual CO₂", marker='o')
plt.plot(y_pred_plain, label="Before Ridge", linestyle='--', marker='x')
plt.plot(y_pred_ridge, label="After Ridge (L2)", linestyle='-.', marker='s')
plt.title("Ridge Regression (L2) using Gradient Descent")
plt.xlabel("Sample Index")
plt.ylabel("CO₂ Emissions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
