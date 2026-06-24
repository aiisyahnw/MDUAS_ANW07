"""
pipeline.py

End-to-end pipeline for the Credit Score classification project.
"""

import os
import joblib

from sklearn.model_selection import train_test_split

from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from train import Trainer


class Pipeline:

    def __init__(self):

        self.ingestion = DataIngestion()

        self.preprocessor = Preprocessor()

        self.trainer = Trainer()

    def run(self):

        # Load data

        print("1. Loading data...")

        df = self.ingestion.load_data()

        # Preprocessing

        print("2. Preprocessing...")

        df = self.preprocessor.clean(df)

        df = self.preprocessor.feature_engineering(
            df,
            fit=True
        )

        # Split data

        print("3. Splitting data...")

        X = df.drop(
            columns=self.preprocessor.TARGET_COL
        )

        y = df[
            self.preprocessor.TARGET_COL
        ]

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

        # Encode

        print("4. Encoding...")

        X_train, X_test = self.preprocessor.encode(

            X_train,

            X_test,

            fit=True

        )

        # Scale

        print("5. Scaling...")

        X_train, X_test = self.preprocessor.scale(

            X_train,

            X_test,

            fit=True

        )

        # Training

        print("6. Training models...")

        self.trainer.set_data(

            X_train,

            X_test,

            y_train,

            y_test

        )

        best_name, best_model, best_metrics, all_results = (

            self.trainer.train_all()

        )

        # Save artifacts

        print("7. Saving models...")

        os.makedirs(

            "models",

            exist_ok=True

        )

        joblib.dump(

            best_model,

            "models/best_model.pkl"

        )

        joblib.dump(

            self.preprocessor,

            "models/preprocessor.pkl"

        )

        joblib.dump(

            self.trainer.label_encoder,

            "models/label_encoder.pkl"

        )

        print("Done!")

        return (

            best_name,

            best_metrics,

            all_results

        )


if __name__ == "__main__":

    pipeline = Pipeline()

    best_name, best_metrics, all_results = pipeline.run()

    print("\nBest Model:", best_name)

    print("\nBest Metrics:")

    for key, value in best_metrics.items():

        print(f"{key}: {value:.4f}")