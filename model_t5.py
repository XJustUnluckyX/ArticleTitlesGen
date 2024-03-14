from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq

import processing
import torch
from datasets import load_metric, load_dataset

# E' stato utilizzato come riferimento la seguente guida, usando T5 come modello:
# https://github.com/laxmimerit/NLP-Tutorials-with-HuggingFace/blob/main/4%20Summarization%20%7C%20NLP%20with%20HuggingFace%20Tutorial.ipynb

def main():

    #Funzione per restituire gli encoding in un formato di supporto al modello
    def get_feature_for_t5(batch):
        text_encodings = tokenizer_t5(batch['abstracts'], text_target=batch['titles'],
                                   max_length=1024, truncation=True)

        text_encodings = {'input_ids': text_encodings['input_ids'],
                     'attention_mask': text_encodings['attention_mask'],
                     'labels': text_encodings['labels']}

        return text_encodings

    path = "C:/Users/Utente/Desktop/IA/datasets"
    df_for_t5 = load_dataset(path=path)
    textA = df_for_t5["train"]["abstracts"][15000:15050]
    textT = df_for_t5["train"]["titles"][15000:15050]
    df_for_t5["train"], _ = df_for_t5["train"].train_test_split(test_size=0.995).values()
    df_for_t5["train"], df_for_t5["test"] = df_for_t5["train"].train_test_split(test_size=0.2).values()

    seed = 23
    torch.manual_seed(seed)

    tokenizer_t5 = AutoTokenizer.from_pretrained("google-t5/t5-base", model_max_length=1024)

    model_t5 = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    # model_base = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    df_for_t5 = df_for_t5.map(get_feature_for_t5, batched=True)

    columns_t5 = ['input_ids', 'labels', 'attention_mask']
    df_for_t5.set_format(type='torch', columns=columns_t5)

    data_collator_t5 = DataCollatorForSeq2Seq(tokenizer_t5, model=model_t5)
    training_args_t5 = TrainingArguments("training_args_t5", evaluation_strategy="epoch", num_train_epochs=10)

    trainer_t5 = Trainer(
        model=model_t5, args=training_args_t5, train_dataset=df_for_t5["train"],
        eval_dataset=df_for_t5["test"],
        tokenizer=tokenizer_t5,
        data_collator=data_collator_t5,
    )

    trainer_t5.train()
    print(trainer_t5.state.log_history)
    trainer_t5.save_model("finetuned_T5")

    pipe_t5 = pipeline("text2text-generation", model="fine_tuned_T5", tokenizer=tokenizer_t5)
    # pipe_base = pipeline("text2text-generation", model=model_base, tokenizer=tokenizer_t5)
    gen_parameters_t5 = {"length_penalty": 0.8, "num_beams": 8, "max_length": 20}

    avg_percent_original = 0
    avg_percent_generated = 0
    for i in range(1, len(textA)):
        # print(textA[i])
        print("\n\n\nT5:")
        generated_title = pipe_t5(textA[i], **gen_parameters_t5)[0]["generated_text"]
        print(generated_title)

        # print("\n\n\nBASE T5:")
        # generated_title = pipe_base(textA[i], **gen_parameters_t5)[0]["generated_text"]
        # print(generated_title)

        print("\n\nOriginal title:\n", textT[i])

        topics_abstract = processing.analyze_topics(textA[i])
        entities_abstract = processing.analyze_named_entities(textA[i])
        topics_original_title = processing.analyze_topics(textT[i])
        entities_original_title = processing.analyze_named_entities(textT[i])
        topics_generated_title = processing.analyze_topics(generated_title)
        entities_generated_title = processing.analyze_named_entities(generated_title)

        keywords_abstract = topics_abstract + entities_abstract
        keywords_original_title = topics_original_title + entities_original_title
        keywords_generated_title = topics_generated_title + entities_generated_title

        percent_keywords_generated = len((set(keywords_abstract) & set(keywords_generated_title)))\
                                     /len(keywords_abstract) * 100

        percent_keywords_original = len((set(keywords_abstract) & set(keywords_original_title)))\
                                     /len(keywords_abstract) * 100

        print("Percentuale keywords titolo originale: ", percent_keywords_original)
        print("Percentuale keywords titolo generato: ", percent_keywords_generated)

        avg_percent_original = avg_percent_original + percent_keywords_original
        avg_percent_generated = avg_percent_generated + percent_keywords_generated

    avg_percent_original = avg_percent_original / len(textA)
    avg_percent_generated = avg_percent_generated / len(textA)
    print("Percentuale media keywords titoli originali: ", avg_percent_original)
    print("Percentuale media keywords titoli generati: ", avg_percent_generated)


if __name__ == "__main__":
    main()
