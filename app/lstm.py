import os

with open(os.path.join("all_files.txt"), "r") as f:
    corpus = f.read()

import tensorflow as tf
import numpy as np

chars = sorted(list(set(corpus)))
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

encoded_corpus = [char_to_int[char] for char in corpus]

seq_length = 50
sequences = []
targets = []

for i in range(0, len(encoded_corpus) - seq_length, 1):
    seq_in = encoded_corpus[i : i + seq_length]
    seq_out = encoded_corpus[i + seq_length]
    sequences.append(seq_in)
    targets.append(seq_out)

X = np.reshape(sequences, (len(sequences), seq_length, 1))
X = X / float(len(chars)) 

y = tf.keras.utils.to_categorical(targets)

model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(
            256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y.shape[1], activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam")

model.fit(X, y, epochs=10, batch_size=128)

model.save("code_completion_bigger.h5")


def generate_code(model, seed_sequence, num_chars):
    # Pad the seed sequence if its length is less than 50
    if len(seed_sequence) < seq_length:
        seed_sequence = " " * (seq_length - len(seed_sequence)) + seed_sequence

    generated_code = seed_sequence

    for _ in range(num_chars):
        x = np.reshape(
            [char_to_int[char] for char in seed_sequence], (1, len(seed_sequence), 1)
        )
        x = x / float(len(chars))

        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]

        generated_code += result

        seed_sequence = seed_sequence[1:] + result

    return generated_code.lstrip()


seed_sequence = "model = tf.keras.Sequential(["
generated_code = generate_code(model, seed_sequence, num_chars=100)
print(generated_code)
