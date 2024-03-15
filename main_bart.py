import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from model_bart import CustomDatasetBart
from processing import analyze_topics, analyze_named_entities

SUMMARY_LEN = 20


def calculate_intersection_percentage(abstract_keywords, title_keywords):
    intersection = set(abstract_keywords) & set(title_keywords)
    percentage = (len(intersection) / len(abstract_keywords)) * 100
    return percentage


def test_best_model():
    # C:/Users/Utente/Desktop/IA/datasets/dataset1.csv
    path = "C:/Users/Xzeni/Downloads/dataset1.csv"
    df = pd.read_csv(path, sep=',', quotechar='"')
    df = df.drop(df.columns[0], axis=1)
    df_test = df.tail(50)

    df_test = df_test.reset_index()

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    bart_shared = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_shared.config.decoder_start_token_id = tokenizer.bos_token_id
    bart_shared.config.eos_token_id = tokenizer.eos_token_id
    bart_shared.config.pad_token_id = tokenizer.pad_token_id

    generation_params = {
        'max_length': SUMMARY_LEN,
        'min_length': 15,
        'early_stopping': True
    }

    test_dataset = CustomDatasetBart(abstracts=df_test["abstracts"], titles=df_test["titles"], tokenizer=tokenizer,
                                     max_length=512)

    test_loader = DataLoader(test_dataset, batch_size=1)

    bart_shared.load_state_dict(torch.load("best_model.pt"))
    bart_shared.eval()

    abstracts = df_test["abstracts"].tolist()

    intersection_mean_original = []
    intersection_mean_decoded = []

    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            with torch.no_grad():
                outputs = bart_shared.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_params)
                decoded_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
                original_title = tokenizer.decode(labels[0], skip_special_tokens=True)

            print(f"Original Title: {original_title}")
            print(f"Original Topics: {analyze_topics(original_title)}")
            print(f"Original Named Entities: {analyze_named_entities(original_title)}")

            print(f"Decoded Title: {decoded_title}")
            print(f"Decoded Topics: {analyze_topics(decoded_title)}")
            print(f"Decoded Named Entities: {analyze_named_entities(decoded_title)}")

            print(f"Abstract Topics: {analyze_topics(abstracts[index])}")
            print(f"Abstract Named Entities: {analyze_named_entities(abstracts[index])}")

            original_title_keywords = analyze_topics(original_title) + analyze_named_entities(original_title)

            decoded_title_keywords = analyze_topics(decoded_title) + analyze_named_entities(decoded_title)

            abstract_keywords = analyze_topics(df_test["abstracts"][index]) + analyze_named_entities(
                df_test["abstracts"][index])

            intersection_original = calculate_intersection_percentage(abstract_keywords, original_title_keywords)
            intersection_mean_original.append(intersection_original)

            intersection_decoded = calculate_intersection_percentage(abstract_keywords, decoded_title_keywords)
            intersection_mean_decoded.append(intersection_decoded)

            print(f"Percentuale keywords titolo originale: {intersection_original}%")
            print(f"Percentuale keywords titolo generato: {intersection_decoded}%")

            print("-------------------------------------")

    sum_original = 0
    sum_decoded = 0

    for x in intersection_mean_original:
        sum_original += x

    for x in intersection_mean_decoded:
        sum_decoded += x

    print(f"Percentuale media topic titoli originali: {sum_original / len(intersection_mean_original)}%")
    print(f"Percentuale media topic titoli generati: {sum_decoded / len(intersection_mean_decoded)}%")


if __name__ == "__main__":
    test_best_model()
