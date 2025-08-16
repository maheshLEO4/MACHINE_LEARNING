import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from collections import Counter
from math import log2

# Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Convert target to numeric
df['PlayTennis'] = df['PlayTennis'].map({'Yes': 1, 'No': 0})

# One-hot encode features
X = pd.get_dummies(df.drop('PlayTennis', axis=1))
y = df['PlayTennis']

# 🌟 Custom entropy function
def entropy(y_vals):
    counts = Counter(y_vals)
    total = sum(counts.values())
    ent = -sum((count/total) * log2(count/total) for count in counts.values())
    return ent

# 🔍 Entropy for whole dataset
print(f"\n🧠 Initial Entropy (PlayTennis): {entropy(y):.4f}")

# 🔍 Entropy after split by each feature
print("\n📊 Feature-wise Entropy Breakdown:")
for col in df.columns[:-1]:
    weighted_entropy = 0
    for val, subset in df.groupby(col):
        ent = entropy(subset['PlayTennis'])
        weight = len(subset)/len(df)
        weighted_entropy += weight * ent
        print(f"  ➤ {col} = {val}: Entropy = {ent:.4f}, Weight = {weight:.2f}")
    gain = entropy(y) - weighted_entropy
    print(f"✅ Info Gain for {col}: {gain:.4f}\n")

# 🏗 Train Decision Tree
model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(X, y)

# 🌳 Plot the Tree using matplotlib
plt.figure(figsize=(14, 8))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=["No", "Yes"], 
          filled=True, 
          rounded=True)
plt.title("🎯 Decision Tree - Play Tennis (Entropy Based)")
plt.show()

# 🧑‍💻 User Input
print("\n📥 Enter weather conditions to check if you can Play Tennis:")
outlook = input("Outlook (Sunny / Overcast / Rain): ").capitalize()
temperature = input("Temperature (Hot / Mild / Cool): ").capitalize()
humidity = input("Humidity (High / Normal): ").capitalize()
wind = input("Wind (Weak / Strong): ").capitalize()

# Create user input DataFrame
user_input = pd.DataFrame([{
    'Outlook': outlook,
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind': wind
}])

# One-hot encode user input using same columns as X
user_input_encoded = pd.get_dummies(user_input)
user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

# Predict
prediction = model.predict(user_input_encoded)[0]
print(f"\n🎾 You can Play Tennis: {'Yes ✅' if prediction == 1 else 'No ❌'}")
