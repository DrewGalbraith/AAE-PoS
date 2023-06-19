# %% [markdown]
# # AAVE PoS-tagging Tutorial
# *Disclaimer: This code relies HEAVILY on this [Hugging Face](https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt) tutorial.

# %%
# Ensure we're working in the right directory

import os
target_dir = r"C:\Users\drews\PycharmProjects\POS_AAVE_495R"
os.chdir(target_dir)

# %%
# Load in data

from datasets import load_dataset as ld
data_files:dict = {"Train": "train_tweets.json", "Validate": "validate_tweets.json", "Test": "test_tweets.json"}
raw_data = ld("json", data_files= data_files)
raw_data.column_names

# %%
# Sanity check
this = raw_data["Train"][0]["Words"]
that = raw_data["Train"][0]["Tags"]
print(list(zip(this, that)))

# %%
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # This is called 'caching the tokenizer.'
tokenizer.is_fast  # verify this is a HF-backed tokenizer

# %%
inputs = tokenizer(raw_data["Train"][0]["Words"], is_split_into_words=True)  # don't forget to inform your tokenizer of the pre-split words
wids, tks = inputs.word_ids(), inputs.tokens()

list(zip(wids, tks))

# %%
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # # If the label is B-XXX we change it to I-XXX
            # if label % 2 == 1:
            #     label += 1
            new_labels.append(label)

    return new_labels

# %%
labels = raw_data["Train"][0]["Tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))

# raw_data["Train"]['Tags']

# %%
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["Words"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["Tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# %%
tokenized_datasets = raw_data.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_data["Train"].column_names,
)

# When the inputs to the tokenizer are list of lists of words, the word_ids() function needs to get the index of the example we want the word IDs of 

# %%
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# %%
batch = data_collator([tokenized_datasets["Train"][i] for i in range(2)])
batch["labels"]

# %%
for i in range(2):
    print(tokenized_datasets["Train"][i]["labels"])

# %%
!pip install seqeval
!pip install evaluate


# %%
import evaluate

metric = evaluate.load("seqeval")

# %%
tag_list = ["ADJ",
"ADP",
".",
"ADV",
"AUX",
"SYM",
"INTJ",
"CONJ",
"X",
"NOUN",
"DET",
"PROPN",
"NUM",
"VERB",
"PRT",
"PRON",
"SCONJ"]
labels = raw_data["Train"][0]["Tags"]
labels = [tag_list[i] for i in labels]
# labels

# %%
predictions = labels.copy()
predictions[6] = "9"
metric.compute(predictions=[predictions], references=[labels])

# %%
import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[tag_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [tag_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

# %%
id2label = {i: label for i, label in enumerate(tag_list)}
label2id = {v: k for k, v in id2label.items()}

# %%
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id
)

# %%
model.config.num_labels


# %%
pip install ipywidgets

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
!pip uninstall -y transformers accelerate
!pip install transformers accelerate

# %%
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["Train"],
    eval_dataset=tokenized_datasets["Validate"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()

trainer.push_to_hub(commit_message="Training complete")



