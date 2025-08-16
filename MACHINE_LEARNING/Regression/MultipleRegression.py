import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

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

# ✅ Step 4b: Print final coefficients
print("\n🎯 Final Model Coefficients:")
for i in range(len(m)):
    print(f"m{i+1}: {m[i]:.4f}")
print(f"b: {b:.4f}")

# Step 5: 3D Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Actual marks
ax.scatter(df['StudyHours'], df['SleepHours'], df['PreviousMarks'],
           c=y, cmap='viridis', s=50, label='Actual Marks')

# Create prediction surface (fix PreviousMarks as mean)
study_range = np.linspace(df['StudyHours'].min(), df['StudyHours'].max(), 10)
sleep_range = np.linspace(df['SleepHours'].min(), df['SleepHours'].max(), 10)
study_grid, sleep_grid = np.meshgrid(study_range, sleep_range)

previous_mean = df['PreviousMarks'].mean()
X_plane = np.column_stack([study_grid.ravel(), sleep_grid.ravel(), np.full(study_grid.size, previous_mean)])
X_plane_scaled = (X_plane - X_mean) / X_std
y_plane = np.dot(X_plane_scaled, m) + b
y_plane = y_plane.reshape(study_grid.shape)

# Plot model prediction surface
ax.plot_surface(study_grid, sleep_grid, y_plane, alpha=0.5, color='green')

# Step 6: User input
try:
    study = float(input("\nEnter study hours: "))
    sleep = float(input("Enter sleep hours: "))
    previous = float(input("Enter previous exam marks: "))

    x_input = np.array([study, sleep, previous])
    x_scaled = (x_input - X_mean) / X_std
    predicted_marks = np.dot(x_scaled, m) + b

    print(f"\n🎯 Predicted Final Marks: {predicted_marks:.2f} / 100")

    # Plot user prediction in 3D
    ax.scatter(study, sleep, previous, color='red', s=100, marker='X', label='Your Prediction')

except ValueError:
    print("⚠️ Please enter valid numeric inputs.")

ax.set_xlabel('Study Hours')
ax.set_ylabel('Sleep Hours')
ax.set_zlabel('Previous Marks / Final Marks')
ax.set_title("3D Visualization: Actual Marks, Model Prediction & User Input")
ax.legend()
plt.show()
