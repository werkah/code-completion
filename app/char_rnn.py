import tensorflow as tf
import numpy as np
import os

with open(os.path.join("all_files.txt"), "r") as f:
    text = f.read()

tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, filters=None)
tokenizer.fit_on_texts(text)

[encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1

max_id = len(tokenizer.word_index)
dataset_size = tokenizer.document_count

train_size = dataset_size * 80 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_length = n_steps + 1
dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)
)
dataset = dataset.prefetch(1)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.GRU(
            128,
            return_sequences=True,
            input_shape=[None, max_id],
            dropout=0.2,
            recurrent_dropout=0.3,
        ),
        tf.keras.layers.GRU(
            128, return_sequences=True, dropout=0.2, recurrent_dropout=0.3
        ),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(max_id, activation="softmax")
        ),
    ]
)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

early_stopping = tf.keras.callbacks.EarlyStopping(patience=50)
history = model.fit(
    dataset, epochs=10, batch_size=batch_size, callbacks=[early_stopping]
)

model.save("code_completion_salamadra_wo_filters_big.h5")


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


seed_text = "def sum(a, b):\n    ret"
completed_text = complete_text(seed_text, temperature=0.21)

print(completed_text)
