import pytest
import requests
from loguru import logger
from rooms.common import FlaskConfig, LOCALHOST
from rooms.model_creation import create_and_deploy_model

@pytest.fixture(scope="module")
def url():
    url = f"{LOCALHOST}:{FlaskConfig.PORT}/{FlaskConfig.ROUTE}"
    logger.info(f"{url=}")
    return url

def test_mlflow_deployment_and_prediction(url):
    # NB! before running this test the FLASK APP need to be running
    # How to run it: python rooms/app.py
    create_and_deploy_model()
    # Sample input data
    data = {
        "referenceCatalog": [
            "room with a veranda",
            "huge room",
            " "
        ],
        "inputCatalog": [
            "big room with balcony",
            "huge room with a BALCONY",
            "small room",
            "    "
        ]
    }

    # Send a POST request to the Flask API

    response = requests.post(url, json=data)
    actual = response.json()
    logger.info(f"{actual=}")
    expected = {'positional_mapping': {'0': [0, 1], '1': [0, 1]}}
    assert actual == expected
