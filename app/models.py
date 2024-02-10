import tensorflow as tf
import numpy as np
import os
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def predict(self, code, n_chars):
        pass


class CharRNNModel(BaseModel):
    def __init__(self, model_path, corpus_path):
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            char_level=True, filters=None
        )
        with open(os.path.join(corpus_path), "r") as f:
            text = f.read()
        self.tokenizer.fit_on_texts(text)
        self.max_id = len(self.tokenizer.word_index)

    def _preprocess(self, texts):
        X = np.array(self.tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, self.max_id)

    def _next_char(self, text, temperature=1):
        X_new = self._preprocess([text])
        y_proba = self.model.predict(X_new, verbose=0)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return self.tokenizer.sequences_to_texts(char_id.numpy())[0]

    def _complete_text(self, text, n_chars=50, temperature=1):
        for _ in range(n_chars):
            text += self._next_char(text, temperature)
        return text

    def predict(self, code, n_chars=20, temperature=0.21):
        return self._complete_text(
            code, n_chars=n_chars, temperature=temperature
        ).split(" ")


class LSTMModel(BaseModel):
    def __init__(self, model_path, corpus_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(os.path.join(corpus_path), "r") as f:
            corpus = f.read()
        self.chars = sorted(list(set(corpus)))
        self.char_to_int = {char: i for i, char in enumerate(self.chars)}
        self.int_to_char = {i: char for i, char in enumerate(self.chars)}
        self.seq_length = 50

    def _complete_text(self, text, n_chars=20):
        if len(text) < self.seq_length:
            text = " " * (self.seq_length - len(text)) + text

        generated_code = text

        for _ in range(n_chars):
            x = np.reshape(
                [self.char_to_int[char] for char in text],
                (1, len(text), 1),
            )
            x = x / float(len(self.chars))

            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = self.int_to_char[index]

            generated_code += result
            text = text[1:] + result

        return generated_code.lstrip()

    def predict(self, code, n_chars=20):
        return self._complete_text(code, n_chars=n_chars).split(" ")
