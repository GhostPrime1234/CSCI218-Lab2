import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def euclidean_distance(n1, n2):
    """
    Calculates the Euclidean distance between two vectors.
    :param n1:
    :param n2:
    :return:
    """
    total = 0
    for i in range(len(n1)):
        total += np.square(n1[i] - n2[i])
    return np.sqrt(total)


def find_neighbours(dataset: pd.DataFrame, sample, k: int):
    """
    Finds the neighbours of a given sample in a given dataset.
    :param dataset:
    :param sample:
    :param k:
    :return:
    """
    dists = []
    for i in range(len(dataset)):
        dist = euclidean_distance(sample, dataset[i])
        dists.append(dist)
    return np.argsort(dists)[:k]


def vote(neighbours: np.ndarray, y):
    """
    Determine the most common class among the neighbours.
    :param neighbours:
    :param y:
    :return:
    """
    votes = {}

    for i in neighbours:
        label = y[i]
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)


def load_and_split_data(file_path, headers: list, train_size=0.5, random_state=42):
    """
    Loads the dataset from a CSV file and split it into training and test sets.
    :param file_path:
    :param headers:
    :param train_size:
    :param random_state:
    :return:
    """
    df = pd.read_csv(file_path, header=None, names=headers)
    train_data, test_data = train_test_split(df, train_size=train_size, random_state=random_state)
    return train_data, test_data


def calculate_accuracy(k_list: list, X_train, y_train, X_test, y_test) -> dict:
    """
    Calculates the accuracy of the KNN classifier for each k value in k_list.
    :param k_list:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    accuracy_dict = {}
    for k_val in k_list:
        preds = []
        for X in X_test:
            neighbours = find_neighbours(X_train, X, k_val)
            y_pred = vote(neighbours, y_train)
            preds.append(y_pred)
        accuracy_dict[k_val] = accuracy_score(y_test, preds)

    return accuracy_dict


def plot_accuracy(accuracy_dict: dict):
    """
    Plots the accuracy of a KNN classifier as a function of k.
    :param accuracy_dict:
    :return:
    """
    plt.figure(figsize=(10, 6))
    plt.plot(list(accuracy_dict.keys()), list(accuracy_dict.values()))
    plt.xlabel("Number of Neighbours (K)")
    plt.ylabel("Accuracy")
    plt.title("KNN Classification Accuracy vs l")
    plt.grid(True)
    plt.xticks(list(accuracy_dict.keys()))
    plt.show()


if __name__ == "__main__":
    dataset_folder = "Iris Dataset"
    file_path = dataset_folder + "/Iris.csv"
    headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    # Load and split the dataset
    print("Loading and splitting the dataset...")
    train_data, test_data = load_and_split_data(file_path, headers)

    # Extract features and labels from training and test data
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Define teh list of k values to test
    k_list = [index for index in range(1, 30, 2)]
    print(f"Testing k values: {k_list}")

    # Calculating accuracy for each k value
    print(f"Calculating accuracy for each k value...")
    accuracy_dict = calculate_accuracy(k_list, X_train, y_train, X_test, y_test)

    # Print the accuracy results
    print("Accuracy results:")
    for k, acc in accuracy_dict.items():
        print(f"k = {k}, Accuracy = {acc:.4f}")

    # Plot the results
    plot_accuracy(accuracy_dict)