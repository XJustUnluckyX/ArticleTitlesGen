import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

TRAIN_BATCH_SIZE = 4
SUMMARY_LEN = 20


def plot_losses(train_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()


class CustomDatasetBart(Dataset):
    def __init__(self, abstracts, titles, tokenizer, max_length):
        self.abstracts = abstracts
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        print(f"Accessing index: {idx}")
        abstract = str(self.abstracts[idx])
        title = str(self.titles[idx])

        inputs = self.tokenizer.encode_plus(
            abstract,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = self.tokenizer.encode(
            title,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels.flatten()
        }


def main():
    # C:/Users/Utente/Desktop/IA/datasets/dataset1.csv
    path = "C:/Users/Xzeni/Downloads/dataset1.csv"
    df = pd.read_csv(path, sep=',', quotechar='"')
    df = df.drop(df.columns[0], axis=1)
    df1 = df.head(200)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    bart_shared = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    bart_shared.config.decoder_start_token_id = tokenizer.bos_token_id
    bart_shared.config.eos_token_id = tokenizer.eos_token_id
    bart_shared.config.pad_token_id = tokenizer.pad_token_id
    batch_size = TRAIN_BATCH_SIZE

    train_dataset = CustomDatasetBart(abstracts=df1["abstracts"], titles=df1["titles"], tokenizer=tokenizer,
                                  max_length=512)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    learning_rate = 3e-5
    num_epochs = 10
    optimizer = optim.AdamW(bart_shared.parameters(), lr=learning_rate)
    best_train_loss = np.inf
    early_stop_count = 0
    early_stop_patience = 2

    train_losses = []

    for epoch in range(num_epochs):
        bart_shared.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()

            outputs = bart_shared(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        bart_shared.eval()

        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            early_stop_count = 0
            torch.save(bart_shared.state_dict(), "best_model.pt")
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            print("Early stopping triggered.")
            break

    plot_losses(train_losses)


if __name__ == "__main__":
    main()
