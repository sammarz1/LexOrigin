import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from huggingface_hub import login



# Connect to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 


# replace with you token to loggin into hugginface
HUGGINGFACE_TOKEN = "YOUR TOKEN"
login(token=HUGGINGFACE_TOKEN)


#read your dataset
df = pd.read_csv('data.csv')



#remove all emtpy rows from you text column
df["TEXTO"] = df["TEXTO"].dropna().astype(str) #


#Make labels for each country
unique_labels = df["PAÍS"].unique().tolist()
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
df["labels"] = df["PAÍS"].map(label2id)


dataset = Dataset.from_pandas(df)


# Use model of you choice
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)




def tokenize_function(examples):
    texts = examples["TEXTO"] 
    if isinstance(texts, str):  
        texts = [texts] 
    elif isinstance(texts, list):
        texts = [str(t) for t in texts] 
    else:
        texts = list(map(str, texts)) 
    
    return tokenizer(texts, padding="longest", truncation=True)



#Tokenize the text column
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["TEXTO"])



#Train-test split
split = dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
test_dataset = split["test"]

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
).to(device)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# function to calculate and save metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)


    with open("classification_report.txt", "w") as f:
        f.write(classification_report(labels, predictions, target_names=unique_labels))

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    avg_confidence = {label: np.mean(probs[:, i]) for label, i in label2id.items()}

    print("Accuracy:", acc)
    print("Confidence per label:", avg_confidence)
    
    return {"accuracy": acc}


#training args, change as desired
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    num_train_epochs=2,  
    weight_decay=0.01,
    fp16=True,  
    push_to_hub=False, 
    gradient_accumulation_steps=2, 
)


optimizer = AdamW(model.parameters(), lr=5e-5, fused=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)  
)


trainer.train()s


trainer.evaluate()

#Save model once trained
model.save_pretrained("./modelo_finetuned")
tokenizer.save_pretrained("./modelo_finetuned")

trainer.push_to_hub("nombre_del_modelo_en_huggingface")

