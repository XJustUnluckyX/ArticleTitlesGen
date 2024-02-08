import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.chunk import ne_chunk
from collections import Counter
import gensim
import nltk

# nltk.download('stopwords')   #Da installare per far funzionare il progetto
# nltk.download('wordnet')     #Da installare per far funzionare il progetto
# nltk.download('averaged_perceptron_tagger') #Da installare per far funzionare il progetto
# nltk.download('maxent_ne_chunker') #Da installare per far funzionare il progetto
# nltk.download('words') #Da installare per far funzionare il progetto


SIZE = 20


def most_common_words(abstracts):
    title_words_counter = Counter()

    for abstract in abstracts:
        words = re.findall(r'\b\w+\b', abstract)
        title_words_counter.update(words)

    most_common_title_words = title_words_counter.most_common(SIZE)
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

    # Ricostruzione
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text


def analyze_topics(abstracts):
    preprocessed_abstracts = [preprocess_text(abstract).split(sep=' ') for abstract in abstracts]

    # Costruzione di un corpus per l'analisi dei topic
    dictionary = gensim.corpora.Dictionary(preprocessed_abstracts)

    bow_corpus = []

    for abstract in preprocessed_abstracts:
        bow_corpus.append(dictionary.doc2bow(abstract))

    # LDA (Latent Dirichlet Allocation)
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=10)

    # Topic principali
    dominant_topics = []
    for abstract_bow in bow_corpus:
        topics = lda_model.get_document_topics(abstract_bow)
        dominant_topic = max(topics, key=lambda x: x[1])
        dominant_topics.append(dominant_topic)

    return dominant_topics


def analyze_named_entities(abstracts):
    named_entities = []
    for abstract in abstracts:
        # Tokenizzazione e tagging POS (Part of Speech)
        tokens = word_tokenize(abstract)
        tagged_tokens = pos_tag(tokens)

        # Estrazione delle Named Entities
        chunked_tokens = ne_chunk(tagged_tokens)

        for subtree in chunked_tokens:
            if hasattr(subtree, 'label'):
                entity = " ".join([word for word, pos in subtree.leaves()])
                named_entities.append(entity)

    return named_entities
