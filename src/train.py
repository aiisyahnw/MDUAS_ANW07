"""
train.py
Training pipeline for the Credit Score classification model.
"""

import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class Trainer:
    def __init__(self, experiment_name: str = "credit_score_classification"):
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment(experiment_name)

        self.label_encoder = LabelEncoder()

        self.models = {}
        self.results = {}

    def set_data(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        self.y_train_xgb = self.label_encoder.fit_transform(y_train)
        self.y_test_xgb = self.label_encoder.transform(y_test)

    def train_logistic_regression(self):
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs'],
            'max_iter': [1000]
        }

        grid = GridSearchCV(
            estimator=LogisticRegression(class_weight='balanced', random_state=42),
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=5,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        self.models['logistic_regression'] = grid.best_estimator_
        self._log_run('logistic_regression', grid.best_estimator_, grid.best_params_)

        return grid.best_estimator_

    def train_random_forest(self):
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid = GridSearchCV(
            estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=5,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        self.models['random_forest'] = grid.best_estimator_
        self._log_run('random_forest', grid.best_estimator_, grid.best_params_)

        return grid.best_estimator_

    def train_xgboost(self):
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1]
        }

        grid = GridSearchCV(
            estimator=XGBClassifier(random_state=42, eval_metric='mlogloss'),
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=5,
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train_xgb)

        self.models['xgboost'] = grid.best_estimator_
        self._log_run('xgboost', grid.best_estimator_, grid.best_params_, is_xgb=True)

        return grid.best_estimator_

    def _log_run(self, model_name, model, best_params, is_xgb=False):
        y_pred = model.predict(self.X_test)

        if is_xgb:
            y_pred = self.label_encoder.inverse_transform(y_pred)

        y_true = self.y_test

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted')
        }

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=model_name)

        self.results[model_name] = metrics

        return metrics

    def train_all(self):
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()

        best_name = max(self.results, key=lambda name: self.results[name]['weighted_f1'])
        best_metrics = self.results[best_name]

        return best_name, self.models[best_name], best_metrics, self.results