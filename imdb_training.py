#!/usr/bin/env python
# coding: utf-8

# In[10]:


from datasets import load_dataset

dataset = load_dataset("imdb")["train"].train_test_split(test_size=0.225, train_size=0.45)

# In[11]:


dataset["test"].to_pandas()

# In[12]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize(data):
    return tokenizer(data["text"], padding=True, truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize, batched=True)

# In[13]:


import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[14]:


from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=750,
    logging_dir="./imdb_logs",
    logging_steps=10,
    save_steps=1000,
    output_dir="./imdb_results",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# In[15]:


trainer.train()

# In[7]:


model.save_pretrained("./imdb-sentiment")
tokenizer.save_pretrained("./imdb-sentiment_pos_neg")


# In[ ]:
