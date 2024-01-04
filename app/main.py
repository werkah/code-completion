from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import os
import numpy as np

with open(os.path.join("all_files.txt"), "r") as f:
    text = f.read()

tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, filters=None)
tokenizer.fit_on_texts(text)

[encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1

max_id = len(tokenizer.word_index)

model = tf.keras.models.load_model("code_completion_salamadra_wo_filters_big.h5")


def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)


def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new, verbose=0)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "GET":
        return jsonify({"error": "Please send a POST request"})
    try:
        code = request.json["code"]
        model_type = request.json["model_type"]
        predictions = complete_text(code, n_chars=20, temperature=0.21)
    except:
        return jsonify({"error": "Error in prediction"})
    return jsonify({"predictions": predictions.split(" ")})


if __name__ == "__main__":
    app.run(debug=True)
