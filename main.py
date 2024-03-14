import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from model import CustomDataset
from processing import analyze_topics, analyze_named_entities, most_common_words

SUMMARY_LEN = 20


def calculate_intersection_percentage(abstract_keywords, title_keywords):
    intersection = set(abstract_keywords) & set(title_keywords)
    percentage = (len(intersection) / len(abstract_keywords)) * 100
    return percentage


def test_best_model():
    path = "C:/Users/Xzeni/Downloads/dataset1.csv"
    df = pd.read_csv(path, sep=',', quotechar='"')
    df = df.drop(df.columns[0], axis=1)
    df_test = df.tail(10)

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

    test_dataset = CustomDataset(abstracts=df_test["abstracts"], titles=df_test["titles"], tokenizer=tokenizer,
                                 max_length=512)

    test_loader = DataLoader(test_dataset, batch_size=1)

    bart_shared.load_state_dict(torch.load("best_model.pt"))
    bart_shared.eval()

    abstracts = df_test["abstracts"].tolist()

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
            print(f"Original Most Common words: {most_common_words(original_title)}")
            print(f"Decoded Title: {decoded_title}")
            print(f"Decoded Topics: {analyze_topics(decoded_title)}")
            print(f"Decoded Named Entities: {analyze_named_entities(decoded_title)}")
            print(f"Decoded Most Common Words: {most_common_words(decoded_title)}")
            print(f"Abstract Topics: {analyze_topics(abstracts[index])}")
            print(f"Abstract Named Entities: {analyze_named_entities(abstracts[index])}")
            print(f"Abstract Most Common words: {most_common_words(abstracts[index])}")

            original_title_keywords = analyze_topics(original_title) + analyze_named_entities(original_title)

            decoded_title_keywords = analyze_topics(decoded_title) + analyze_named_entities(decoded_title)

            abstract_keywords = analyze_topics(df_test["abstracts"][index]) + analyze_named_entities(df_test["abstracts"][index])

            intersection_original = calculate_intersection_percentage(abstract_keywords, original_title_keywords)
            intersection_decoded = calculate_intersection_percentage(abstract_keywords, decoded_title_keywords)

            print(f"Intersection with Original Title: {intersection_original}%")
            print(f"Intersection with Decoded Title: {intersection_decoded}%")

            if intersection_decoded > intersection_original:
                print("The decoded title better fits the abstract.")
            elif intersection_decoded < intersection_original:
                print("The original title better fits the abstract.")
            else:
                print("Both titles have the same fit to the abstract.")

            print("-------------------------------------")


if __name__ == "__main__":
    test_best_model()


