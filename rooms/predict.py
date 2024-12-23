from typing import List, Dict

import requests
from loguru import logger
from rooms.common import FlaskConfig, LOCALHOST
from rooms.data_processing import prepare_match_candidate_pairs, remove_invalid_rooms
from rooms.model_creation import load_model


if __name__ == "__main__":

    # Sample input data
    data = {
        "referenceCatalog": [
            None,
            "Big room with balcony",
            "",
            "small suite",
            "Economy Studio with open space",
            "",
        ],
        "inputCatalog": [
            "Huge room along with a balcony",
            None,
            "very big room with a nice balcony",
            "luxury suite",
            "",
            None,
            "A",
            "economy studio with spacious layout",
        ],
    }

    # Send a POST request to the Flask API
    url = f"{LOCALHOST}:{FlaskConfig.PORT}/{FlaskConfig.ROUTE}"
    logger.info(f"{url=}")
    response = requests.post(url, json=data)

    # Check the response status code
    if response.status_code == 200:
        print("Predictions:")
        print(response.json())  # Print the predictions
    else:
        print(f"Error: {response.status_code}")
        print(response.text)  # Print the error message


def get_mapping_from_reference_to_supplier_catalog(
    reference_catalog: List[str], supplier_catalog: List[str]
) -> Dict[int, List[int]]:
    """
    Get the mapping of reference to supplier catalog based on the model predictions.

    Args:
        reference_catalog: List of reference room names.
        supplier_catalog: List of supplier room names.

    Returns:
        Dict[int, List[int]]: Mapping of reference room indices to supplier room indices.
    """
    df = prepare_match_candidate_pairs(reference_catalog, supplier_catalog)
    df = remove_invalid_rooms(df)
    model = load_model()

    df["match"] = model.predict(df)
    logger.info(f"Predictions:\n{df=}")
    mapping = df[df["match"]].groupby("A_pos")["B_pos"].apply(list).to_dict()
    logger.info(f"{mapping=}")
    return mapping
