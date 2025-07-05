import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Sample dataset — Engine efficiency case
data = {
    'RPM': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
    'Temp': [70, 75, 80, 85, 90, 95, 100, 105],
    'FuelFlow': [2.5, 2.7, 3.0, 3.4, 3.6, 3.9, 4.2, 4.5],
    'Efficiency': [25, 30, 40, 45, 50, 52, 53, 51]  # Output variable
}
df = pd.DataFrame(data)

# Step 2: Polynomial feature generation
X_raw = df[['RPM', 'Temp', 'FuelFlow']].values
y = df['Efficiency'].values

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_raw = poly.fit_transform(X_raw)

# Step 3: Normalize
X_mean = X_poly_raw.mean(axis=0)
X_std = X_poly_raw.std(axis=0)
X = (X_poly_raw - X_mean) / X_std

# Step 4: Initialize
m = np.zeros(X.shape[1])
b = 0
alpha = 0.01
epochs = 10000
n = len(X)

# Step 5: Gradient Descent
for i in range(epochs):
    y_pred = np.dot(X, m) + b
    error = y - y_pred

    dm = (-2 / n) * np.dot(X.T, error)
    db = (-2 / n) * np.sum(error)

    m -= alpha * dm
    b -= alpha * db

    if i % 1000 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {i}: Loss = {loss:.4f}")
# Step 5.1: R² Score Calculation
ss_res = np.sum((y - y_pred) ** 2)          # Residual sum of squares
ss_tot = np.sum((y - np.mean(y)) ** 2)      # Total sum of squares
r2_score = 1 - (ss_res / ss_tot)            # R² formula
print(f"\n🧠 R² Score of the Model: {r2_score:.4f}")


# Step 6: Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(n), y, 'bo-', label='Actual Efficiency')
plt.plot(range(n), y_pred, 'g--', label='Predicted (Polynomial)')
plt.xlabel("Engine Index")
plt.ylabel("Efficiency (%)")
plt.title("Polynomial Regression on Engine Efficiency")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Step 7: Final Coefficients
print("\n🎯 Final Model Coefficients:")
feature_names = poly.get_feature_names_out()
for i in range(len(m)):
    print(f"{feature_names[i]}: {m[i]:.4f}")
print(f"b (intercept): {b:.4f}")

# Step 8: User Input Prediction
try:
    rpm = float(input("\nEnter engine RPM: "))
    temp = float(input("Enter engine temperature (°C): "))
    flow = float(input("Enter fuel flow rate (L/s): "))

    x_input = np.array([[rpm, temp, flow]])
    x_poly_input = poly.transform(x_input)
    x_scaled = (x_poly_input - X_mean) / X_std

    predicted_efficiency = np.dot(x_scaled, m) + b
    print(f"\n🎯 Predicted Engine Efficiency: {predicted_efficiency[0]:.2f}%")

except ValueError:
    print("⚠️ Please enter valid numeric inputs.")
# Note: This code is a complete example of polynomial regression applied to a sample dataset.