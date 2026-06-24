"""
endpoint.py
Deploy SageMaker endpoint dari model.tar.gz.
"""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


BUCKET_NAME = "MDUAS_ANW07"
REGION = "us-east-1"
S3_PREFIX = "credit-score"
ENDPOINT_NAME = "credit-score-endpoint"
SKLEARN_VERSION = "1.2-1"

S3_MODEL_URI = f"s3://{BUCKET_NAME}/{S3_PREFIX}/model.tar.gz"


def deploy():
    sm_client = boto3.client("sagemaker", region_name=REGION)

    try:
        sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint '{ENDPOINT_NAME}' sudah ada, skip deploy.")
        return
    except:
        pass

    print("Sedang mendeploy endpoint, mohon tunggu sekitar 3-5 menit...")

    session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
    role = sagemaker.get_execution_role()

    model = SKLearnModel(
        model_data=S3_MODEL_URI,
        role=role,
        entry_point="inference.py",
        framework_version=SKLEARN_VERSION,
        sagemaker_session=session
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        endpoint_name=ENDPOINT_NAME
    )

    print(f"\nEndpoint berhasil dideploy!")
    print(f"Endpoint name: {ENDPOINT_NAME}")
    return predictor


if __name__ == "__main__":
    deploy()