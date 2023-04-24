#!/usr/bin/env python
# coding: utf-8

# # Finetuning All Layers

# <img src="figures/3_finetune-all.png" width=500>

# In[ ]:


# pip install transformers


# In[ ]:


# pip install datasets


# In[ ]:


# pip install lightning


# In[ ]:


# get_ipython().run_line_magic('load_ext', 'watermark')
# get_ipython().run_line_magic('watermark', '--conda -p torch,transformers,datasets,lightning')


# # 1 Loading the dataset into DataFrames

# In[ ]:


# pip install datasets

import shutil

from datasets import load_dataset

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import CountVectorizer

from local_dataset_utilities import download_dataset, load_dataset_into_to_dataframe, partition_dataset
from local_dataset_utilities import IMDBDataset


# In[ ]:


download_dataset()

df = load_dataset_into_to_dataframe()
partition_dataset(df)


# In[ ]:


df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")


# # 2 Tokenization and Numericalization

# **Load the dataset via `load_dataset`**

# In[ ]:


imdb_dataset = load_dataset(
    "csv",
    data_files={
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv",
    },
)

print(imdb_dataset)


# **Tokenize the dataset**

# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer input max length:", tokenizer.model_max_length)
print("Tokenizer vocabulary size:", tokenizer.vocab_size)


# In[ ]:


def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)


# In[ ]:


imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)


# In[ ]:


del imdb_dataset


# In[ ]:


imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# In[ ]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # 3 Set Up DataLoaders

# In[ ]:


from torch.utils.data import DataLoader, Dataset


class IMDBDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


# In[ ]:


train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=12,
    shuffle=True, 
    num_workers=4
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=12,
    num_workers=4
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=12,
    num_workers=4
)


# # 4 Initializing Modules

# **Wrap in LightningModule for Training**

# In[ ]:


import lightning as L
import torch
import torchmetrics


class CustomLightningModule(L.LightningModule):
    def __init__(self, model, learning_rate=5e-5):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
        
    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"])        
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"])        
        self.log("val_loss", outputs["loss"], prog_bar=True)
        
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.val_acc(predicted_labels, batch["label"])
        self.log("val_acc", self.val_acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"])        
        
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.test_acc(predicted_labels, batch["label"])
        self.log("accuracy", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# In[ ]:


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


callbacks = [
    ModelCheckpoint(
        save_top_k=1, mode="max", monitor="val_acc"
    )  # save top 1 model
]
logger = CSVLogger(save_dir="logs/", name="my-model")


# # 5 Finetuning

# ## All layers

# In[ ]:


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

lightning_model = CustomLightningModule(model)


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


import time
start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")


# ## 1 -- Last Layer

# In[ ]:

print("1 -- Last Layer")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

lightning_model = CustomLightningModule(model)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False
    
for param in model.classifier.parameters():
    param.requires_grad = True


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")


# ## 2 -- Last 2 Layers

# In[ ]:

print("2 -- Last 2 Layers")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

lightning_model = CustomLightningModule(model)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False
    
for param in model.pre_classifier.parameters():
    param.requires_grad = True
    
for param in model.classifier.parameters():
    param.requires_grad = True


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")


# ## 3 -- Last 2 Layers + Last Tranformer Block

print("3 -- Last 2 Layers + Last Tranformer Block")

# In[ ]:



model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

lightning_model = CustomLightningModule(model)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False
    
for param in model.pre_classifier.parameters():
    param.requires_grad = True
    
for param in model.classifier.parameters():
    param.requires_grad = True

for param in model.distilbert.transformer.layer[5].parameters():
    param.requires_grad = True


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")


# ## 4 -- Last 2 Layers + Last 2 Transformer Blocks

# In[ ]:

print("4 -- Last 2 Layers + Last 2 Transformer Blocks")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

lightning_model = CustomLightningModule(model)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False
    
for param in model.pre_classifier.parameters():
    param.requires_grad = True
    
for param in model.classifier.parameters():
    param.requires_grad = True

for param in model.distilbert.transformer.layer[5].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[4].parameters():
    param.requires_grad = True


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")


# ## 5 -- Last 2 Layers + Last 3 Transformer Blocks

# In[ ]:

print("5 -- Last 2 Layers + Last 3 Transformer Blocks")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

lightning_model = CustomLightningModule(model)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False
    
for param in model.pre_classifier.parameters():
    param.requires_grad = True
    
for param in model.classifier.parameters():
    param.requires_grad = True

for param in model.distilbert.transformer.layer[5].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[4].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[3].parameters():
    param.requires_grad = True


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")



## 6 -- Last 2 Layers + Last 4 Transformer Blocks

print("6 -- Last 2 Layers + Last 4 Transformer Blocks")

for param in model.parameters():
    param.requires_grad = False
    
for param in model.pre_classifier.parameters():
    param.requires_grad = True
    
for param in model.classifier.parameters():
    param.requires_grad = True

for param in model.distilbert.transformer.layer[5].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[4].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[3].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[2].parameters():
    param.requires_grad = True


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")


# ## 7 -- Last 2 Layers + Last 5 Transformer Blocks

# In[ ]:

print("## 7 -- Last 2 Layers + Last 5 Transformer Blocks")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

lightning_model = CustomLightningModule(model)


# In[ ]:


for param in model.distilbert.transformer.layer[5].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[4].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[3].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[2].parameters():
    param.requires_grad = True

for param in model.distilbert.transformer.layer[1].parameters():
    param.requires_grad = True


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")


# ## 8 -- Last 2 Layers + Last 6 Transformer Blocks

# In[ ]:

print("8 -- Last 2 Layers + Last 6 Transformer Blocks")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

lightning_model = CustomLightningModule(model)


# In[ ]:


for param in model.distilbert.transformer.layer[5].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[4].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[3].parameters():
    param.requires_grad = True
    
for param in model.distilbert.transformer.layer[2].parameters():
    param.requires_grad = True

for param in model.distilbert.transformer.layer[1].parameters():
    param.requires_grad = True

for param in model.distilbert.transformer.layer[0].parameters():
    param.requires_grad = True


# In[ ]:


trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
)


# In[ ]:


start = time.time()

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")


# In[ ]:


trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")


# In[ ]:


trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
shutil.rmtree("logs")
logger = CSVLogger(save_dir="logs/", name="my-model")

