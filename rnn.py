import numpy as np
import keras as ks
import nltk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.src.utils import to_categorical

from processing import most_common_words, preprocess_text, analyze_named_entities, analyze_topics


def main():
    path= "C:/Users/Utente/Desktop/IA/datasets/dataset1.csv"
    df = pd.read_csv(path, sep = ',', quotechar='"')
    df = df.drop(df.columns[0], axis=1)
    df = df.head(20)
    abstracts = df[df.columns[1]].to_numpy()
    titles = df[df.columns[0]].to_numpy()
    all_texts= np.concatenate([abstracts, titles])

    # Estrazione features
    common_words = most_common_words(all_texts)
    dominant_topics = analyze_topics(all_texts)
    named_entities = analyze_named_entities(all_texts)

    # Concatenazione Features
    all_features = [f"{preprocess_text(text)} {' '.join([str(word) for word, _ in common_words])} "
                    f"{' '.join([str(topic[0]) for topic in dominant_topics])} "
                    f"{' '.join(str(named_entities))}" for text in all_texts]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_features)
    sequences = tokenizer.texts_to_sequences(all_features)

    # Padding delle sequenze per uniformarle in lunghezza
    max_length = max([len(seq) for seq in sequences])
    print("max_length")
    print(max_length)
    print("len(tokenizer.word_index)+1")
    print(len(tokenizer.word_index)+1)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    texts = to_categorical(padded_sequences, num_classes=len(tokenizer.word_index) + 1)
    abstracts_input = texts[:abstracts.size]
    titles_input = texts[-titles.size:]

    # RNN con LSTM
    model = ks.Sequential([
        ks.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=max_length),
        ks.layers.LSTM(units=128, return_sequences=True),
        ks.layers.LSTM(units=128),
        ks.layers.Dense(units=len(tokenizer.word_index)+1, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(abstracts_input, titles_input, epochs=50, batch_size= 16, validation_split=0.2)

if __name__ == "__main__":
    main()