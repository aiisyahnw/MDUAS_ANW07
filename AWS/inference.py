"""
inference.py
SageMaker model handler — bundled inside model.tar.gz.
SageMaker calls these four functions automatically.
"""

import os
import json
import joblib
import pandas as pd


def model_fn(model_dir):

    model = joblib.load(
        os.path.join(model_dir, "best_model.pkl")
    )

    preprocessor = joblib.load(
        os.path.join(model_dir, "preprocessor.pkl")
    )

    return {
        "model": model,
        "preprocessor": preprocessor
    }


def input_fn(
    request_body,
    content_type="application/json"
):

    if content_type == "application/json":

        data = json.loads(request_body)

        return pd.DataFrame([data])

    raise ValueError(
        f"Unsupported content type: {content_type}"
    )


def predict_fn(
    input_data,
    model_artifacts
):

    preprocessor = model_artifacts["preprocessor"]

    model = model_artifacts["model"]

    X = preprocessor.transform_for_inference(
        input_data
    )

    prediction = model.predict(X)[0]

    return prediction


def output_fn(
    prediction,
    accept="application/json"
):

    return json.dumps(
        {
            "prediction": str(prediction)
        }
    ), accept