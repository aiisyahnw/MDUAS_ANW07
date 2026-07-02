"""
AWS_pipeline.py
Training pipeline untuk Credit Score
"""

import os
import sys
import tarfile
import boto3
import joblib
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from train import Trainer

BUCKET_NAME = "MDUAS_ANW07" 
REGION = "us-east-1"             
S3_PREFIX = "credit-score"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def run():
    print("Step 1: Data Ingestion...")
    ingestion = DataIngestion(
        path=os.path.join(os.path.dirname(__file__), "..", "data", "data_D.csv")
    )
    df = ingestion.load_data()
    print("Berhasil!")

    print("Step 2: Preprocessing...")
    preprocessor = Preprocessor()
    df = preprocessor.clean(df)
    df = preprocessor.feature_engineering(df, fit=True)

    X = df.drop(columns=preprocessor.TARGET_COL)
    y = df[preprocessor.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = preprocessor.encode(X_train, X_test, fit=True)
    X_train, X_test = preprocessor.scale(X_train, X_test, fit=True)

    print("Step 3: Training...")
    trainer = Trainer()
    trainer.set_data(X_train, X_test, y_train, y_test)
    best_model = trainer.train_random_forest()
    print("Training selesai!")

    print("Step 4: Evaluation...")
    print("-" * 40)
    metrics = trainer.results["random_forest"]
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"Weighted F1   : {metrics['weighted_f1']:.4f}  <-- main metric")
    print("-" * 40)

    print("Step 5: Saving artifacts...")
    aws_model_dir = os.path.join(os.path.dirname(__file__), "..", "aws_models")
    os.makedirs(aws_model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(aws_model_dir, "best_model.pkl"))
    joblib.dump(preprocessor, os.path.join(aws_model_dir, "preprocessor.pkl"))

    # Bundle inference.py + pkl files ke model.tar.gz
    inference_src = os.path.join(os.path.dirname(__file__), "inference.py")
    tar_path = os.path.join(os.path.dirname(__file__), "..", "model.tar.gz")

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(os.path.join(aws_model_dir, "best_model.pkl"), arcname="best_model.pkl")
        tar.add(os.path.join(aws_model_dir, "preprocessor.pkl"), arcname="preprocessor.pkl")
        tar.add(inference_src, arcname="inference.py")

    print(f"model.tar.gz siap: {tar_path}")

    print("Step 6: Upload ke S3...")
    s3 = boto3.client("s3", region_name=REGION)
    s3_key = f"{S3_PREFIX}/model.tar.gz"
    s3.upload_file(tar_path, BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"
    print(f"Upload berhasil: {s3_uri}")

    print("\nPipeline selesai!")
    print(f"S3 URI: {s3_uri}")
    return s3_uri


if __name__ == "__main__":
    run()