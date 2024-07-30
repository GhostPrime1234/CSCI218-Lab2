import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
dataset_folder = "Iris Dataset"

headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(f'{dataset_folder}/iris.csv', header=None, names=headers)

train_data, test_data = train_test_split(df, train_size=0.5, shuffle=True)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# List of k_values
k_values = list(range(1,20,2))

# Dictionary to store error rates for each k
accuracy_dict = {}

for k in k_values:
    # Initialise knn classifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Predict on teh test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy (which is 1 - classification error rate)
    accuracy = accuracy_score(y_test, y_pred)

    # Store error rate for this k
    accuracy_dict[k] = accuracy

for k, error_rate in accuracy_dict.items():
    print(f"K = {k}, Accuracy = {error_rate:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(list(accuracy_dict.keys()), list(accuracy_dict.values()), marker='o', linestyle='-', color='b')
plt.xlabel('Number of Neighbours (k)')
plt.ylabel('Accuracy')
plt.title('KNN Classification Accuracy vs. k')
plt.grid(True)
plt.xticks(k_values)
plt.show()
