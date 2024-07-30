import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Iris Dataset/iris.csv')

# Split dataset into features (X) and labels (y)
X = df.iloc[:, :-1].values # features
y = df.iloc[:,-1].values   # labels

# Split data into 50% training and 50% test sets
train_data, test_data = train_test_split(X, y, train_size=0.5, random_state=42)

X_train = train_data[:, :-1].values
y_train = train_data[:, -1].values

X_test = test_data[:, :-1].values
y_test = test_data[:, -1].values

# List of k_values
k_values = list(range(1,20,2))

# Dictionary to store error rates for each k
error_rates = {}

for k in k_values:
    # INitialise knn classifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Predict on teh test set
    y_pred= knn.predict(X_test)

    # Calculate accuracy (which is 1 - classification error rate)
    accuracy = accuracy_score(y_test, y_pred)

    # Compute classification error rate
    error_rate = 1-accuracy

    # Store error rate for this k
    error_rates[k] = error_rate

for k, error_rate in error_rates.items():
    print(f"K = {k}, Error rate = {error_rate:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(list(error_rates.keys()), list(error_rates.values()), marker='o', linestyle='-', color='b')
plt.xlabel('Number of Neighbours (k)')
plt.ylabel('Accuracy')
plt.title('KNN Classification Accuracy vs. k')
plt.grid(True)
plt.xticks(k_values)
plt.show()
