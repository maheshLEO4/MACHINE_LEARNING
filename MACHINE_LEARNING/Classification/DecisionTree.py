import pandas as pd
from collections import Counter


data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)


# Gini helper functions
def gini(column):
    '''Calculate Gini impurity for a column'''
    counts = Counter(column)
    total = len(column)
    return 1 - sum((count / total) ** 2 for count in counts.values())  #Gini(S)=1−∑​p(i)^2​

def gini_gain(data, feature, target):
    '''Calculate Gini gain of a feature relative to the target'''
    total_gini = gini(data[target])
    weighted_gini = 0
    for val in set(data[feature]):
        subset = data[data[feature] == val]
        weight = len(subset) / len(data)
        weighted_gini += weight * gini(subset[target])
    return total_gini - weighted_gini   #GiniGain(S,A)=Gini(S)−∑​( |Sv| / |S| ) * Gini(Sv)

def best_feature(data, target):
    '''Select the feature with the highest Gini gain'''
    max_gain, best = -1, None
    for feature in data.columns:
        if feature == target: 
            continue
        gain = gini_gain(data, feature, target)
        if gain > max_gain:             
            max_gain, best = gain, feature
    return best


# Build Decision Tree using Gini
def build_tree(data, target):
    if len(set(data[target])) == 1:
        return list(data[target])[0]
    if len(data.columns) == 1:
        counts = Counter(data[target])
        return max(counts, key=counts.get)

    feature = best_feature(data, target)
    tree = {feature: {}}
    for val in set(data[feature]):
        subset = data[data[feature] == val].drop(columns=[feature])
        tree[feature][val] = build_tree(subset, target)
    return tree

tree = build_tree(df, 'PlayTennis')


# Prediction function
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = sample.get(feature)
    if value not in tree[feature]:
        return None  # Unknown path
    return predict(tree[feature][value], sample)


# Train/Test split (manual)
train_df = df.iloc[:10]  # first 10 rows as training data
test_df = df.iloc[10:]   # last 4 rows as test data

tree = build_tree(train_df, 'PlayTennis')

# 6. Evaluate on test data
y_true = list(test_df['PlayTennis'])
y_pred = [predict(tree, row) for _, row in test_df.drop(columns=['PlayTennis']).iterrows()]

# Handle unknown predictions (None → majority class from train)
majority_class = train_df['PlayTennis'].mode()[0]
y_pred = [p if p is not None else majority_class for p in y_pred]

# Confusion matrix and accuracy
tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'Yes' and yp == 'Yes')
tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'No' and yp == 'No')
fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'No' and yp == 'Yes')
fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'Yes' and yp == 'No')

#tp=true positive, tn=true negative, fp=false positive, fn=false negative

accuracy = (tp + tn) /(tp+tn+fp+fn) # Accuracy = (TP + TN) / Total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision = TP / (TP + FP)
recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)


# Print results
print("Decision Tree (Gini):")
print(tree)
print("\nConfusion Matrix:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")  
print(f"F1 Score: {f1_score:.2f}")