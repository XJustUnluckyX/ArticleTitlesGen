import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.chunk import ne_chunk
from collections import Counter
import gensim
import numpy as np
import nltk

# nltk.download('stopwords')   #Da installare per far funzionare il progetto
# nltk.download('wordnet')     #Da installare per far funzionare il progetto
# nltk.download('averaged_perceptron_tagger') #Da installare per far funzionare il progetto
# nltk.download('maxent_ne_chunker') #Da installare per far funzionare il progetto
# nltk.download('words') #Da installare per far funzionare il progetto


SIZE = 20


def most_common_words(text):
    title_words_counter = Counter()

    words = re.findall(r'\b\w+\b', text)
    title_words_counter.update(words)

    most_common_title_words = [word for word, _ in title_words_counter.most_common(SIZE)]
    return most_common_title_words


def preprocess_text(text):
    # Rimozione caratteri non alfabetici e conversione in minuscolo
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Tokenizzazione del testo
    tokens = word_tokenize(text)

    # Rimozione delle stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatizzazione delle parole
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text


def analyze_topics(text):
    preprocessed_text = preprocess_text(text)
    tokens = preprocessed_text.split()

    # Costruzione di un corpus per l'analisi dei topic
    dictionary = gensim.corpora.Dictionary([tokens])

    bow_text = [dictionary.doc2bow(tokens)]

    # LDA (Latent Dirichlet Allocation)
    lda_model = gensim.models.LdaMulticore(bow_text, num_topics=3, id2word=dictionary, passes=10)

    topic_words = {}
    for i, topic in lda_model.show_topics(formatted=False):
        topic_words[i] = [word for word, _ in topic]

    dominant_topics = []
    for text_bow in bow_text:
        topics = lda_model.get_document_topics(text_bow)
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        dominant_topics.append(topic_words[dominant_topic])

    return [word for sublist in dominant_topics for word in sublist]


def analyze_named_entities(text):
    named_entities = []

    # Tokenizzazione e tagging POS (Part of Speech)
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    # Estrazione delle Named Entities
    chunked_tokens = ne_chunk(tagged_tokens)

    for subtree in chunked_tokens:
        if hasattr(subtree, 'label'):
            entity = " ".join([word for word, pos in subtree.leaves()])
            named_entities.append(entity)

    return named_entities
