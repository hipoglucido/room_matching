from rooms.model_creation import create_and_deploy_model
from rooms.predict import get_mapping_from_reference_to_supplier_catalog

if __name__ == "__main__":
    create_and_deploy_model()
    referenceCatalog = [
        "Big room with balcony",
        "small suite",
    ]
    inputCatalog = [
        "Huge room along with a balcony",
        "big room with a balcony",
        "very big room with a nice balcony",
        "luxury suite",
    ]

    mapping = get_mapping_from_reference_to_supplier_catalog(
        reference_catalog=referenceCatalog,
        supplier_catalog=inputCatalog,
    )
    print(mapping)
