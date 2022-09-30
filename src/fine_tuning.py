#%%
import os
import json
from transformers import AutoModelForSequenceClassification
#from transformers import AutoTokenizer
from utils import datasets_exist, train_transformer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
from pathlib import Path
# %%
""" TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 2e-5 
#WEIGHT_DECAY = 0.01
#SEED_TRAIN = 0
#data_path_save = "data/processed"
#data_path_save = os.path.join("../",data_path_save)
#base_model = "xlm-roberta-base"
basenet_tokenizer = 'roberta'
model_mlm = "models/mlm/xmlr_mlm/"
model_mlm = os.path.join("../",model_mlm)
task=1
schedule='linear'
epochs=1
model_path_save = '../models/fine-tuned/bert_test.pt'
language = None """

#%%
config_file = open('config.json')
config = json.load(config_file)

# %%
TRAIN_BATCH_SIZE = config["fine_tuning"]["TRAIN_BATCH_SIZE"]
LEARNING_RATE = config["fine_tuning"]["LEARNING_RATE"]
basenet_tokenizer = config["fine_tuning"]["basenet_tokenizer"]
model_mlm = config["mlm_training"]["MODEL_PATH_SAVE"]
task=config["fine_tuning"]["TASK"]
schedule=config["fine_tuning"]["schedule"]
epochs=config["fine_tuning"]["EPOCHS"]
model_path_save = config["fine_tuning"]["MODEL_PATH_SAVE"]
language = config["fine_tuning"]["language"]
Path(os.path.split(model_path_save)[0]).mkdir(parents=True, exist_ok=True) 

#%% 
print("-------------- MLM Training INPUTS: --------------")
print("Path model: ", model_path_save)
#%% 
print("Loading data...")
train_loader = DataLoader(
    dataset=datasets_exist.exist_2021(
        config["fine_tuning"]["train_dataset"], 
    sample = config["fine_tuning"]["sample"], basenet = basenet_tokenizer, 
    concat_metwo=False, text_cleaner=False, balance_metwo=False, language = language), 
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)

validation_loader = DataLoader(
    dataset=datasets_exist.exist_2021(
        config["fine_tuning"]["validation_dataset"], 
    sample = config["fine_tuning"]["sample"], basenet = basenet_tokenizer, text_cleaner=False, language = language), 
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)

#%%
if task==1:
    model = AutoModelForSequenceClassification.from_pretrained(model_mlm)
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_mlm, num_labels=6)

#%%
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = epochs * len(train_loader)
num_warmup_steps = num_training_steps // 10
if schedule == "linear":
    print("Using linear scheduler with warmup")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
elif schedule == "constant":
    print("Using constant scheduler with warmup")
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
    )

#%%
print("Train starting...")
train_transformer.train(
    model, optimizer, train_loader, validation_loader, epochs, model_path_save, 
    scheduler=scheduler, task=task)

# %%
