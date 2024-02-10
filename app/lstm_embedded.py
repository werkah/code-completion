import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np
import pickle


with open(os.path.join("all_files.txt"), "r") as file:
    data = file.read().replace('\n', ' ').replace('\r', ' ')

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts([data])
pickle.dump(tokenizer, open('token.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]

sequences = [sequence_data[i - 5:i + 1] for i in range(5, len(sequence_data))]
sequences = np.array(sequences)

X, y = sequences[:, :-1], sequences[:, -1]
vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=vocab_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=30)

model = Sequential([
    Embedding(vocab_size, 10, input_length=5),
    LSTM(512, return_sequences=True),
    Dropout(0.2),
    LSTM(512),
    Dropout(0.2),
    Dense(512, activation="relu"),
    Dense(vocab_size, activation="softmax")
])

model.summary()
plot_model(model, to_file='plot.png', show_layer_names=True)

checkpoint = ModelCheckpoint("next_word.h5", monitor='loss', verbose=1, save_best_only=True)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=50, callbacks=[checkpoint])

model.save('nm.h5')

model = load_model('nm.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))
