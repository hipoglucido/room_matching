import subprocess
import time

import requests
from loguru import logger

from rooms.common import FlaskConfig, LOCALHOST
from rooms.model_creation import create_and_deploy_model, get_dummy_prediction_from_mlflow

import pytest

@pytest.fixture(scope="module")  # Use module scope to deploy once per module
def deployed_model():
    create_and_deploy_model()

@pytest.fixture(scope="module")  # Use module scope to deploy once per module
def deployed_flask_app():
    subprocess.Popen(
        ["python", FlaskConfig.APP_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)


def test_mlflow_deployment_and_prediction(deployed_model, deployed_flask_app):


    # Sample input data
    data = {
        "referenceCatalog": [

            "Big room with balcony",
            "small suite",

        ],
        "inputCatalog": [
            "Huge room along with a balcony",
            "big room with a balcony",

            "very big room with a nice balcony",
            "luxury suite",

        ],
    }

    # Send a POST request to the Flask API
    url = f"{LOCALHOST}:{FlaskConfig.PORT}/{FlaskConfig.ROUTE}"
    logger.info(f"{url=}")
    response = requests.post(url, json=data)
    actual = response.json()

    expected = []
    assert actual == expected
