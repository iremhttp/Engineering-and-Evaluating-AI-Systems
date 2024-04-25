import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy import *
import random
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray):
        super(DecisionTree, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = DecisionTreeClassifier(random_state=0)
        self.predictions = None

    def train(self, data):
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        if data.y_test.ndim == 1:
            print("Classification Report:")
            print(classification_report(data.y_test, self.predictions, zero_division=0))
        else:
            for i in range(data.y_test.shape[1]):
                print(f"Accuracy for output {i}: {accuracy_score(data.y_test[:, i], self.predictions[:, i]):.2f}")
                print(classification_report(data.y_test[:, i], self.predictions[:, i], zero_division=0))
