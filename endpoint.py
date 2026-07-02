"""
endpoint.py
Deploy SageMaker endpoint dari model.tar.gz.
"""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


BUCKET_NAME = "mduasanw07"
REGION = "us-east-1"
S3_PREFIX = "credit-score"
SKLEARN_VERSION = "1.4-2"
ENDPOINT_NAME = "credit-score-endpoint"

S3_MODEL_URI = f"s3://{BUCKET_NAME}/{S3_PREFIX}/model.tar.gz"


def get_lab_role_arn():
    iam = boto3.client("iam", region_name=REGION)
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def deploy():

    boto3.setup_default_session(region_name=REGION)

    session = sagemaker.Session()
    sm_client = boto3.client("sagemaker", region_name=REGION)

    try:
        sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint '{ENDPOINT_NAME}' sudah ada.")
        return
    except sm_client.exceptions.ClientError:
        pass

    role = get_lab_role_arn()

    print(f"Role      : {role}")
    print(f"Model URI : {S3_MODEL_URI}")

    model = SKLearnModel(
        model_data=S3_MODEL_URI,
        role=role,
        entry_point="inference.py",
        framework_version=SKLEARN_VERSION,
        sagemaker_session=session
    )

    print("\nDeploying endpoint...")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME
    )

    print("\nEndpoint deployed successfully!")
    print(f"Endpoint Name : {ENDPOINT_NAME}")

    return predictor


if __name__ == "__main__":
    deploy()