"""
AWS_pipeline.py
Training pipeline untuk Credit Score.
Melatih model Random Forest terbaik, menyimpan model,
membuat model.tar.gz, lalu mengupload ke Amazon S3.
"""

import os
import tarfile
import boto3
import joblib
from sklearn.model_selection import train_test_split

from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from train_aws import Trainer


BUCKET_NAME = "mduasanw07"
REGION = "us-east-1"
S3_PREFIX = "credit-score"


BASE_DIR = os.path.dirname(__file__)

MODEL_DIR = os.path.join(BASE_DIR, "aws_models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "data_D.csv")
INFERENCE_PATH = os.path.join(BASE_DIR, "inference.py")
PREPROCESSING_PATH = os.path.join(BASE_DIR, "preprocessing.py")
TAR_PATH = os.path.join(BASE_DIR, "model.tar.gz")


def run():

    print("Step 1: Data Ingestion...")

    ingestion = DataIngestion(path=DATA_PATH)
    df = ingestion.load_data()

    print("Berhasil!")

    print("Step 2: Preprocessing...")

    preprocessor = Preprocessor()

    df = preprocessor.clean(df)
    df = preprocessor.feature_engineering(df, fit=True)

    X = df.drop(columns=preprocessor.TARGET_COL)
    y = df[preprocessor.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_test = preprocessor.encode(
        X_train,
        X_test,
        fit=True
    )

    X_train, X_test = preprocessor.scale(
        X_train,
        X_test,
        fit=True
    )

    print("Step 3: Training Random Forest...")

    trainer = Trainer()

    trainer.set_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_model = trainer.train_random_forest()

    print("Training selesai!")

    print("Step 4: Evaluation...")

    metrics = trainer.results["random_forest"]

    print("-" * 40)
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"Weighted F1   : {metrics['weighted_f1']:.4f}")
    print("-" * 40)

    print("Step 5: Saving artifacts...")

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "best_model.pkl")
    )

    joblib.dump(
        preprocessor,
        os.path.join(MODEL_DIR, "preprocessor.pkl")
    )

    print("Step 6: Creating model.tar.gz...")

    with tarfile.open(TAR_PATH, "w:gz") as tar:

        tar.add(
            os.path.join(MODEL_DIR, "best_model.pkl"),
            arcname="best_model.pkl"
        )

        tar.add(
            os.path.join(MODEL_DIR, "preprocessor.pkl"),
            arcname="preprocessor.pkl"
        )

        tar.add(
            INFERENCE_PATH,
            arcname="inference.py"
        )

        tar.add(
            PREPROCESSING_PATH,
            arcname="preprocessing.py"
        )

    print("model.tar.gz berhasil dibuat.")

    print("Step 7: Upload ke Amazon S3...")

    s3 = boto3.client(
        "s3",
        region_name=REGION
    )

    s3_key = f"{S3_PREFIX}/model.tar.gz"

    s3.upload_file(
        TAR_PATH,
        BUCKET_NAME,
        s3_key
    )

    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    print("Upload berhasil!")
    print(s3_uri)

    print("\nPipeline selesai.")

    return s3_uri


if __name__ == "__main__":
    run()