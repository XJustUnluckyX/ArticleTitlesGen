# ArticleTitlesGen

## Introduzione
Questo progetto è stato sviluppato come parte del corso di Intelligenza Artificiale presso l'Università degli Studi di Salerno nell'anno accademico 2023/2024. 

L'obiettivo del progetto è implementare un modello in grado di generare titoli coerenti per Articoli di Ricerca. Sono stati testati diversi approcci:

1. Realizzazione manuale Encoder-Decoder
2. Finetuning di BART
3. Finetuning di GPT-2
4. Finetuning di T5

## Autori 
Nicolò Delogu: [https://github.com/XJustUnluckyX]
Davide La Gamba: [https://github.com/davide-lagamba]

## Librerie Principali
Le principali librerie utilizzate per lo sviluppo del progetto sono:
- transformers
- nltk

## Dataset Utilizzato
Il dataset utilizzato per il finetuning è disponibile su Kaggle al seguente link: [Kaggle - ArXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts)

## Riferimenti
Durante lo sviluppo del progetto, abbiamo fatto riferimento alle seguenti guide:
- [Fine-tuning GPT-2](https://github.com/lizatukhtina/fine-tune-gpt2-for-meetiing-summarization/blob/main/Fine-tuning%20GPT-2.ipynb)
- [NLP with HuggingFace Tutorial](https://github.com/laxmimerit/NLP-Tutorials-with-HuggingFace/blob/main/4%20Summarization%20%7C%20NLP%20with%20HuggingFace%20Tutorial.ipynb)
- [HuggingFace Documentation for BART](https://huggingface.co/docs/transformers/model_doc/bart)
- [Develop an Encoder-Decoder Model for Sequence-to-Sequence Prediction in Keras](https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/)

---
**Nota:** Questo progetto è stato realizzato per scopi accademici e di ricerca. 
