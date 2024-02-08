import numpy as np
import keras as ks
import nltk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from processing import most_common_words, preprocess_text, analyze_named_entities, analyze_topics


def main():
    path= "C:/Users/Utente/Desktop/IA/datasets/dataset1.csv"
    df = pd.read_csv(path, sep = ',', quotechar='"')
    df = df.drop(df.columns[0], axis=1)
    print(df[df.columns[0]].to_numpy())

    abstracts = df[df.columns[1]].to_numpy()
    titles = df[df.columns[0]].to_numpy()


    # Estrazione features
    common_words = most_common_words(abstracts)
    dominant_topics = analyze_topics(abstracts)
    named_entities = analyze_named_entities(abstracts)

    # Concatenazione Features
    all_features = [f"{preprocess_text(abstract)} {' '.join([word for word, _ in common_words])} " 
                    f"{' '.join([str(topic[0]) for topic in dominant_topics])} " 
                    f"{' '.join(named_entities)}" for abstract in abstracts]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_features)
    sequences = tokenizer.texts_to_sequences(all_features)

    # Padding delle sequenze per uniformarle in lunghezza
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    # RNN con LSTM
    model = ks.Sequential([
        ks.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=max_length),
        ks.layers.LSTM(units=128, return_sequences=True),
        ks.layers.LSTM(units=128),
        ks.layers.Dense(units=len(tokenizer.word_index)+1, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(padded_sequences, np.array(titles), epochs=50, batch_size=16, validation_split=0.2)

if __name__ == "__main__":
    main()