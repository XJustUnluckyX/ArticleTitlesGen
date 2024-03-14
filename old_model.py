import numpy as np
import pandas as pd
from keras import Input, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Attention, Dense
import keras as ks
from processing import preprocess_text


def build_encoder_model(vocab_size, max_length, embedding_dim):
    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, embedding_dim)(input_layer)
    encoder = LSTM(256, return_state=True)
    _, state_h, state_c = encoder(embedding_layer)
    encoder_states = [state_h, state_c]
    encoder_model = Model(inputs=input_layer, outputs=encoder_states)
    return encoder_model


def build_decoder_model(vocab_size, latent_dim, encoder_states_shape):
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(vocab_size, latent_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=encoder_states_shape)

    attention_layer = Attention()
    context_vector = attention_layer([decoder_outputs, encoder_states_shape[0]])
    decoder_combined_context = ks.layers.concatenate([context_vector, decoder_outputs])
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_combined_context)

    decoder_model = Model(inputs=[decoder_inputs] + encoder_states_shape, outputs=[decoder_outputs, state_h, state_c])
    return decoder_model


def generate_title(encoder_model, decoder_model, input_sequence, title_tokenizer, max_title_length):
    states_value = encoder_model.predict(input_sequence[:, 0, :])

    target_seq = np.zeros((1, 1))

    target_seq[0, 0] = title_tokenizer.word_index['<start>']

    decoded_title = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in title_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        if sampled_word is None:
            continue

        decoded_title += ' ' + sampled_word

        if sampled_word == '<end>' or len(decoded_title.split()) > max_title_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_title.strip()


def main():
    path = "C:/Users/Xzeni/Downloads/dataset1.csv"
    df = pd.read_csv(path, sep=',', quotechar='"')
    df = df.drop(df.columns[0], axis=1)
    df = df.head(10000)
    abstracts = df["abstracts"].values
    titles = df["titles"].values

    abstracts = [preprocess_text(a) for a in abstracts]
    titles = [preprocess_text(t) for t in titles]

    abstract_tokenizer = Tokenizer()
    abstract_tokenizer.fit_on_texts(abstracts)

    abstract_sequences = abstract_tokenizer.texts_to_sequences(abstracts)

    title_tokenizer = Tokenizer(oov_token=None)
    title_tokenizer.fit_on_texts(titles)

    title_tokenizer.word_index['<start>'] = len(title_tokenizer.word_index) + 1
    title_tokenizer.word_index['<end>'] = len(title_tokenizer.word_index) + 2

    title_sequences = title_tokenizer.texts_to_sequences(titles)

    max_abstract_length = max([len(seq) for seq in abstract_sequences])
    max_title_length = max([len(seq) for seq in title_sequences])

    abstract_vocab_size = len(abstract_tokenizer.word_index) + 1
    title_vocab_size = len(title_tokenizer.word_index) + 1

    padded_abstract_sequences = []

    for a in abstract_sequences:
        padded_abstract_sequences.append(pad_sequences([a], maxlen=max_abstract_length, padding='post'))

    encoder_model = build_encoder_model(abstract_vocab_size, max_abstract_length, 100)
    decoder_model = build_decoder_model(title_vocab_size, 256, [Input(shape=(256,)), Input(shape=(256,))])

    # Addestramento

    input_seq_index = 1
    input_sequence = np.array([padded_abstract_sequences[input_seq_index]])
    generated_title = generate_title(encoder_model, decoder_model, input_sequence, title_tokenizer, max_title_length)
    print("Generated Title:", generated_title)
    print("Original Title:", titles[1])


if __name__ == "__main__":
    main()