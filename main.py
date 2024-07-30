import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def euclidean_distance(n1, n2):
    total = 0
    for i in range(len(n1)):
        total += np.square(n1[i] - n2[i])
    return np.sqrt(total)

def find_neighbours(dataset: pd.DataFrame, sample: int, k: int):
    dists = []
    for i in range(len(dataset)):
        dist = euclidean_distance(sample, dataset[i])
        dists.append(dist)
    return np.argsort(dists)[:k]    

def vote(neighbours: list, y):
    votes = {}

    for i in neighbours:
        label = y[i]
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
        
    return max(votes, key=votes.get)
    

if __name__ == "__main__":
    dataset_folder = "Iris Dataset"

    headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(dataset_folder + "/Iris.csv", header=None, names=headers)
    train_data, test_data = train_test_split(df,  train_size=0.5, shuffle=True, random_state=42)
  
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # k_list = [i for i in range(1,16, 2)]
    k_list = range(1,16,2)

    print(k_list)
    accuracy_dict = {}
    
    for k_value in k_list:
        predictions = []
        for x in X_test:
            neighbours = find_neighbours(X_train, x, k_value)
            prediction = vote(neighbours, y_train)
            predictions.append(prediction)
    
        accuracy = accuracy_score(y_test, predictions)
        accuracy_dict[k_value] = accuracy

        
    for k, acc in accuracy_dict.items():
        print(f"k = {k}, Accuracy = {acc}")
