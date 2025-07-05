import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Dataset using DataFrame
data = {
    'StudyHours': [2, 4, 5, 6, 7, 8, 9, 10],
    'SleepHours': [6, 7, 5, 8, 6, 7, 5, 6],
    'PreviousMarks': [60, 70, 65, 80, 75, 85, 78, 90],
    'FinalMarks': [50, 60, 58, 75, 72, 82, 80, 90]
}
df = pd.DataFrame(data)

# Step 2: Normalize the features
X_raw = df[['StudyHours', 'SleepHours', 'PreviousMarks']].values
y = df['FinalMarks'].values

X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std  # Normalized input
print(X)

# Step 3: Initialize model parameters
m = np.zeros(X.shape[1])  # 3 weights for 3 features
b = 0
alpha = 0.01
epochs = 10000
n = len(X)

# Step 4: Gradient Descent
for i in range(epochs):
    y_pred = np.dot(X, m) + b
    error = y - y_pred

    dm = (-2 / n) * np.dot(X.T, error)
    db = (-2 / n) * np.sum(error)

    m = m - alpha * dm
    b = b - alpha * db

    if i % 1000 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {i}: Loss = {loss:.4f}, m = {[round(val, 4) for val in m]}, b = {b:.4f}")


# Step 5: Plotting (before user input)
plt.figure(figsize=(8, 5))

# Actual marks
plt.scatter(range(n), y, color='blue', label='Actual Marks')

# Predicted marks from model
y_all_pred = np.dot(X, m) + b
plt.plot(range(n), y_all_pred, color='green', linestyle='--', label='Model Prediction')

plt.xlabel("Student Index")
plt.ylabel("Final Marks")
plt.title("Actual vs Predicted Final Marks (Before User Input)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Final Model Coefficients
print("\n🎯 Final Model Coefficients:")
for i in range(len(m)):
    print(f"m{i+1}: {m[i]:.4f}")

print(f"b: {b:.4f}")


# Step 6: User Input for Prediction
try:
    study = float(input("\nEnter study hours: "))
    sleep = float(input("Enter sleep hours: "))
    previous = float(input("Enter previous exam marks: "))

    x_input = np.array([study, sleep, previous])
    x_scaled = (x_input - X_mean) / X_std
    predicted_marks = np.dot(x_scaled, m) + b

    print(f"\n🎯 Predicted Final Marks: {predicted_marks:.2f} / 100")

except ValueError:
    print("⚠️ Please enter valid numeric inputs.")
