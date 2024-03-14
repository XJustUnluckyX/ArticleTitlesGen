import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import processing
from torch.optim import Adam
from tqdm.auto import tqdm
from datasets import load_metric, load_dataset


# E' stato utilizzato come riferimento la seguente guida:
# https://github.com/lizatukhtina/fine-tune-gpt2-for-meetiing-summarization/blob/main/Fine-tuning%20GPT-2.ipynb

# Classe usata per formattare il Dataset e supportare l'utilizzo del modello
class CustomDatasetForGPT(Dataset):
    def __init__(self, path: str, tokenizer):
        self.data = pd.read_csv(path)
        self.abstracts = self.data['abstracts'].values
        self.titles = self.data['titles'].values

        self.text = []

        for abstract, title in zip(self.abstracts, self.titles):
            self.text.append("Abstract: " + abstract + "  Title for the previous abstract: " + title + " ")

        self.text_encoded = tokenizer(self.text, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.text_encoded['input_ids']
        self.attention_mask = self.text_encoded['attention_mask']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]


#Funzione di training
def train(dataloader, model, optim):
    epochs = 10 # Stop a 10 epoche
    losses = []

    for i in tqdm(range(epochs)):
        print(f'\nIniziata epoca {i + 1}')
        total_loss = 0
        num_batches = 0

        for input, a in tqdm(dataloader):
            optim.zero_grad()
            loss = model(input, attention_mask=a, labels=input).loss
            loss.backward()
            optim.step()

            total_loss += loss.item()
            num_batches += 1
            losses.append(loss.item())

        avg_loss = total_loss / num_batches
        print(f'Epoca: {i + 1}, Loss: {avg_loss:.4f}')

    model.save_pretrained(save_directory='finetuned_gpt2')
    return losses

def main():
    path = "C:/Users/Utente/Desktop/IA/traindatasets/traindataset.csv"
    path2 = "C:/Users/Utente/Desktop/IA/datasets/"

    df = load_dataset(path=path2)
    textA = df["train"]["abstracts"][15000:15050]
    textT = df["train"]["titles"][15000:15050]

    seed = 23
    torch.manual_seed(seed)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model_base = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.add_special_tokens({"pad_token": "",
                                  "bos_token": "",
                                  "eos_token": ""
                                  })

    model.resize_token_embeddings(len(tokenizer))

    dataset = CustomDatasetForGPT(path, tokenizer)

    dataloader = DataLoader(dataset, batch_size=8)

    optim = Adam(model.parameters(), lr=1e-3)

    losses_result = train(dataloader, model, optim)

    plt.figure(figsize=(15, 10))
    plt.plot(losses_result)
    plt.grid()
    plt.xlabel("Batches")
    plt.ylabel("Average loss")
    plt.show()

    pipe_gpt_finetuned = pipeline("text-generation", model="finetuned_gpt2", tokenizer=tokenizer,
                                  max_new_tokens=20)

    avg_percent_original = 0
    avg_percent_generated = 0


    for i in range(0, len(textA)-1):
        # print(textA[i])
        print("\n\nFine-tuned GPT-2:")
        text = pipe_gpt_finetuned("Abstract: " + textA[i] + " Title for the previous abstract: ",
                                  pad_token_id=tokenizer.eos_token_id)
        generated_title = text[0]["generated_text"].rsplit(" Title for the previous abstract: ", 1)[-1]
        print(generated_title)

        print("\n\nOriginal title:")
        print(textT[i])

        topics_abstract = processing.analyze_topics(textA[i])
        entities_abstract = processing.analyze_named_entities(textA[i])
        topics_original_title = processing.analyze_topics(textT[i])
        entities_original_title = processing.analyze_named_entities(textT[i])
        topics_generated_title = processing.analyze_topics(generated_title)
        entities_generated_title = processing.analyze_named_entities(generated_title)

        keywords_abstract = topics_abstract + entities_abstract
        keywords_original_title = topics_original_title + entities_original_title
        keywords_generated_title = topics_generated_title + entities_generated_title

        percent_keywords_generated = len(
            (set(keywords_abstract) & set(keywords_generated_title))) \
                                     / len(keywords_abstract) * 100

        percent_keywords_original = len((set(keywords_abstract) & set(keywords_original_title))) \
                                    / len(keywords_abstract) * 100

        print("Percentuale keywords titolo originale: ", percent_keywords_original)
        print("Percentuale keywords titolo generato: ", percent_keywords_generated)

        avg_percent_original = avg_percent_original + percent_keywords_original
        avg_percent_generated = avg_percent_generated + percent_keywords_generated


    avg_percent_original = avg_percent_original / len(textA)
    avg_percent_generated = avg_percent_generated / len(textA)
    print("Percentuale media topic titoli originali: ", avg_percent_original)
    print("Percentuale media topic titoli generati: ", avg_percent_generated)


if __name__ == "__main__":
    main()
