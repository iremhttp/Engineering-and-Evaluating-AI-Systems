import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy import *
import random
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

# encapsulates functionality related to training, predicting, and evaluating a random forest classifier model.
class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        # Determine the dimensionality of y_test and handle accordingly
        if data.y_test.ndim == 1:

            print("* Classification Report:")
            print(classification_report(data.y_test, self.predictions, zero_division=0))
        else:
            for i in range(data.y_test.shape[1]):
                print(f"Accuracy for output {i}: {accuracy_score(data.y_test[:, i], self.predictions[:, i]):.2f}")
                print(f"* Classification report for output {i}:")
                print(classification_report(data.y_test[:, i], self.predictions[:, i], zero_division=0))


    def data_transform(self) -> None:
        ...