from flask import Flask, request, jsonify
from flask_cors import CORS
from models import CharRNNModel, LSTMModel


models = {
    "char_rnn": CharRNNModel(
        "code_completion_salamadra_wo_filters_big.h5", "all_files.txt"
    ),
    "lstm": LSTMModel("code_completion_bigger.h5", "all_files.txt"),
}


app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "GET":
        return jsonify({"error": "Please send a POST request"})
    try:
        code = request.json["code"]
        model_type = request.json["model_type"]
        predictions = models[model_type].predict(code, n_chars=20)
    except:
        return jsonify({"error": "Error in prediction"})
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(debug=True)
