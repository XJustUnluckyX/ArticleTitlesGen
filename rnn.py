import numpy as np
import keras as ks
import nltk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical  # Correction here
from processing import most_common_words, preprocess_text, analyze_named_entities, analyze_topics


def main():
    path = "C:/Users/Xzeni/Downloads/dataset1.csv"
    df = pd.read_csv(path, sep=',', quotechar='"')
    df = df.drop(df.columns[0], axis=1)
    df = df.head(20)
    abstracts = df[df.columns[1]].to_numpy()
    titles = df[df.columns[0]].to_numpy()

    X = []

    for title, abstract in zip(titles, abstracts):
        combined_text = f"{title} {abstract}"
        X.append(combined_text)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    y_sequences = tokenizer.texts_to_sequences(abstracts.copy())

    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    y_padded_sequences = pad_sequences(y_sequences, maxlen=max_length, padding='post')
    Y = to_categorical(y_padded_sequences, num_classes=len(tokenizer.word_index) + 1)

    model = ks.Sequential([
        ks.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length),
        ks.layers.LSTM(units=128, return_sequences=True),
        ks.layers.LSTM(units=128, return_sequences=True),
        ks.layers.TimeDistributed(ks.layers.Dense(units=len(tokenizer.word_index) + 1, activation='softmax'))
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(padded_sequences, Y, epochs=50, batch_size=16, validation_split=0.2)


if __name__ == "__main__":
    main()
