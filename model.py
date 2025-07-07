import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Generate Data

def boundary(slope, x):
    return slope*(x**2) + x

def generate_datapoints(slope, lower, upper, n_datapoints, class_1_ratio):

    random.seed(100)
    np.random.seed(100)

    X = np.zeros((n_datapoints, 2))
    labels = np.zeros(n_datapoints)

    n_class_1 = int(n_datapoints * class_1_ratio)
    n_class_0 = n_datapoints - n_class_1  
    counter_c1 = 0
    counter_c0 = 0

    for i in range(n_datapoints):
        while True:
            x1_i = random.uniform(lower, upper)
            x2_i = random.uniform(lower, upper)
            y_boundary = boundary(slope, x1_i)

            if x2_i > y_boundary and counter_c1 < n_class_1:
                labels[i] = 1
                counter_c1 += 1
            elif x2_i < y_boundary and counter_c0 < n_class_0:
                labels[i] = 0
                counter_c0 += 1
            else:
                continue

            X[i][0] = x1_i
            X[i][1] = x2_i

            break

    return X, labels

# Logistic Regression Model

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,precision_score, recall_score

def plot_boundary(X, y, model, title, ax=None, show_colorbar=False):
    if ax is None:
        f, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.mgrid[x_min:x_max:0.05, y_min:y_max:0.05]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    
    if show_colorbar:
        ax_c = plt.colorbar(contour, ax=ax)
        ax_c.set_label("$P(y = 1)$")

    ax.scatter(X[:,0], X[:, 1], c=y, cmap="RdBu", edgecolor="white", s=30 )
    ax.set_title(title, fontsize=10)

def logistic(slope, lower, upper, n_datapoints, class_1_ratio):

    random.seed(100)
    np.random.seed(100)

    train_acc = []
    test_acc = []

    train_precision = []
    train_recall = []
    test_precision = []
    test_recall = []
    contour = []
    c_matrix = []

    for n in n_datapoints:

        X, y = generate_datapoints(slope= 0.1, lower=low, upper=up,
                                        n_datapoints=n, class_1_ratio=0.5)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        contour.append((X, y, model))
        c_matrix.append(confusion_matrix(y_test, y_pred))

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

        train_precision.append(precision_score(y_train, train_pred))
        train_recall.append(recall_score(y_train, train_pred))

        test_precision.append(precision_score(y_test, y_pred))
        test_recall.append(recall_score(y_test, y_pred))
        
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Logistic Regression Metrics", fontsize=16)

    metrics = [
        ("Accuracy", train_acc, test_acc),
        ("Precision", train_precision, test_precision),
        ("Recall", train_recall, test_recall)
    ]

    for i, (title, train_vals, test_vals) in enumerate(metrics):
        axs[i].plot(n_datapoints, train_vals, label=f'Train {title}')
        axs[i].plot(n_datapoints, test_vals, label=f'Test {title}')
        
        axs[i].set_title(f"{title} for different number of data points")
        axs[i].set_xlabel("# Data points")
        axs[i].set_ylabel(f"{title} Score")
        axs[i].set_xticks(n_datapoints)
        axs[i].set_xticklabels([str(int(n)) for n in n_datapoints])
        axs[i].grid(True)
        axs[i].legend()

    plt.show()

    return c_matrix, contour

# Neural Network

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def load_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size=2, hidden_1=16, hidden_2=10, output=2):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_1)
        self.h2 = nn.Linear(hidden_1, hidden_2)
        self.output = nn.Linear(hidden_2, output)
        self.activation = nn.ReLU()

    def forward(self, X):
        out = self.h1(X)    
        out = self.activation(out)
        out = self.h2(out)
        out = self.activation(out)
        out = self.output(out)

        return out