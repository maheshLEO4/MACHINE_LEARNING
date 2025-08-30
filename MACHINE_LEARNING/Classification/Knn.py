import numpy as np
from sklearn.datasets import make_moons

# ==============================
# 1. DATA PREPARATION
# ==============================
def prepare_data(n_samples=300, noise=0.4, split_ratio=0.7, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    data = np.column_stack((X, y))
    np.random.shuffle(data)

    split_idx = int(split_ratio * len(data))
    train, test = data[:split_idx], data[split_idx:]

    X_train, y_train = train[:, :-1], train[:, -1].astype(int)
    X_test, y_test = test[:, :-1], test[:, -1].astype(int)

    return X_train, y_train, X_test, y_test


# ==============================
# 2. KNN IMPLEMENTATION
# ==============================
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_indices = np.argsort(distances)[:k] # returns the indices of the k nearest neighborsgit 
        k_labels = y_train[k_indices]
        predictions.append(np.bincount(k_labels).argmax()) # most common class label among the neighbors
    return np.array(predictions)


# ==============================
# 3. EVALUATION
# ==============================
def confusion_matrix_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))#no of true positives that is predicted as positive and actually positive
    tn = np.sum((y_true == 0) & (y_pred == 0))#no of true negatives that is predicted as negative and actually negative
    fp = np.sum((y_true == 0) & (y_pred == 1))#no of false positives that is predicted as positive but actually negative
    fn = np.sum((y_true == 1) & (y_pred == 0))#no of false negatives that is predicted as negative but actually positive

    accuracy = (tp + tn) / len(y_true)#from total no of predictions how many are correct
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0 #from predicted positives how many are actually positive
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0   #from actual positives how many are predicted positive
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 #harmonic mean of precision and recall

    print("Confusion Matrix:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"F1 Score : {f1:.2%}\n")

    return accuracy


def hyperparameter_tuning(X_train, y_train, X_test, y_test, k_values=[1, 3, 5, 7, 9]):
    print("Hyperparameter Tuning:")
    for k in k_values:
        y_pred = knn_predict(X_train, y_train, X_test, k=k)
        acc = np.mean(y_pred == y_test)
        print(f"k={k}, Accuracy: {acc:.2%}")
    print()


def k_fold_cv(X, y, k_values, n_folds=5):
    fold_size = len(X) // n_folds
    results = {}

    for k in k_values:
        acc_list = []
        for fold in range(n_folds):
            start, end = fold * fold_size, (fold + 1) * fold_size
            X_test, y_test = X[start:end], y[start:end]
            X_train = np.concatenate((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))

            y_pred = knn_predict(X_train, y_train, X_test, k)
            acc_list.append(np.mean(y_pred == y_test))
            print(f"k={k}, Fold {fold+1}, Accuracy={acc_list[-1]:.2f}")

        avg_acc = np.mean(acc_list)
        results[k] = avg_acc
        print(f"Average Accuracy for k={k}: {avg_acc:.2f}\n")

    best_k = max(results, key=results.get)
    print(f"Best k={best_k} with Accuracy={results[best_k]:.2f}")
    return results


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()

    # Predictions with k=5
    y_pred = knn_predict(X_train, y_train, X_test, k=5)
    confusion_matrix_metrics(y_test, y_pred)

    # Hyperparameter tuning
    hyperparameter_tuning(X_train, y_train, X_test, y_test)

    # k-Fold Cross Validation
    print("Cross Validation Results:")
    X, y = np.vstack((X_train, X_test)), np.concatenate((y_train, y_test))
    k_fold_cv(X, y, [1, 3, 5, 7, 9], n_folds=5)
