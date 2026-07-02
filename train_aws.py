"""
train.py
Training Random Forest (best tuned model).
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class Trainer:

    def __init__(self):

        self.label_encoder = LabelEncoder()

        self.models = {}
        self.results = {}

    def set_data(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_random_forest(self):

        # Best parameters hasil GridSearchCV
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)

        self.models["random_forest"] = model

        y_pred = model.predict(self.X_test)

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(
                self.y_test,
                y_pred,
                average="weighted"
            ),
            "recall": recall_score(
                self.y_test,
                y_pred,
                average="weighted"
            ),
            "weighted_f1": f1_score(
                self.y_test,
                y_pred,
                average="weighted"
            )
        }

        self.results["random_forest"] = metrics

        return model