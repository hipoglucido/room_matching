from flask import Flask, request, jsonify

from rooms.common import FlaskConfig
from rooms.predict import get_mapping_from_reference_to_supplier_catalog

app = Flask(__name__)


@app.route(f"/{FlaskConfig.ROUTE}", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        mapping = get_mapping_from_reference_to_supplier_catalog(
            reference_catalog=data["referenceCatalog"],
            supplier_catalog=data["inputCatalog"],
        )
        result = jsonify({"positional_mapping": mapping})
        return result

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=FlaskConfig.PORT)
