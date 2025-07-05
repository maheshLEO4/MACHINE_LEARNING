import numpy as np
import matplotlib.pyplot as plt
#train a simple linear regression model using gradient descent to predict house prices based on area.
# Step 1: Original Dataset
X_raw = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500], dtype=np.float64)
y = np.array([40, 50, 60, 72, 85, 95, 105, 120], dtype=np.float64)

# Step 2: Normalize X for stability
X_mean = np.mean(X_raw)
X_std = np.std(X_raw)
X = (X_raw - X_mean) / X_std  # Scaled feature

# Step 3: Hyperparameters
m = 0
b = 0
alpha = 0.01  # Safe because input is normalized
epochs = 10000
n = len(X)

# Step 4: Gradient Descent Loop
for i in range(epochs):
    y_pred = m * X + b
    error = y - y_pred

    dm = (-2 / n) * np.sum(X * error)
    db = (-2 / n) * np.sum(error)

    m = m - alpha * dm
    b = b - alpha * db

    if i % 1000 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {i}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")

# Step 5: Plot
plt.scatter(X_raw, y, color='blue', label='Actual Data')
line_x = np.linspace(min(X_raw), max(X_raw), 100)
line_x_scaled = (line_x - X_mean) / X_std
line_y = m * line_x_scaled + b
plt.plot(line_x, line_y, color='red', label='Prediction Line')
plt.xlabel("Area (Square Feet)")
plt.ylabel("Price (₹ Lakhs)")
plt.title("Gradient Descent: House Price Prediction")
plt.grid(True)
plt.legend()
plt.show()

# Step 6: Final Equation (in normalized form)
print("\nFinal Model (normalized):")
print(f"Price = {m:.4f} * normalized(Area) + {b:.4f}")

#training complete, now we can use the model to make predictions.





# Final Model (in original form)
# Step 7: User Input
try:
    x_test = float(input("\nEnter house area in sq. ft: "))
    x_test_scaled = (x_test - X_mean) / X_std # Normalize input
    y_test = m * x_test_scaled + b
    print(f"Predicted Price for {x_test:.0f} sq. ft: ₹{y_test:.2f} Lakhs")
except ValueError:
    print("Invalid input! Please enter a numeric value.")


# This code implements a simple linear regression model using gradient descent to predict house prices based on area.
# It normalizes the input data for stability, performs gradient descent to find the best-fit line,
# and allows user input for predictions. Finally, it visualizes the results with a plot.
