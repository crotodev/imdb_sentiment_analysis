#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset

dataset = load_dataset("imdb")


# In[2]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


# In[3]:


from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_steps=10,
    output_dir="./results",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)


# In[ ]:


trainer.train() 


# In[ ]:


model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

