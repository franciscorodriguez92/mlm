#%%
import os
import json
from transformers import AutoModelForSequenceClassification
#from transformers import AutoTokenizer
from utils import datasets_exist, train_transformer
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
from pathlib import Path
from utils.architectures import Transformer_lstm_model,transformer_sbert_lstm_model,transformer_sbert_lstm_model_v1, transformer_sbert_lstm_model_
from transformers.trainer_utils import set_seed
#set_seed(1)
#set_seed(2023)
#set_seed(0)
set_seed(123)
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
cascade_task2 = config["fine_tuning"]["cascade_task2"]
cascade_task1 = config["fine_tuning"]["cascade_task1"]
Path(os.path.split(model_path_save)[0]).mkdir(parents=True, exist_ok=True) 
semi_supervised = False

#%% 
print("-------------- MLM Training INPUTS: --------------")
print("Path model: ", model_path_save)
#%% 
print("Loading data...")
train_dataset = datasets_exist.exist_2021(
        config["fine_tuning"]["train_dataset"], 
    sample = config["fine_tuning"]["sample"], basenet = basenet_tokenizer, 
    concat_metwo=False, text_cleaner=True, balance_metwo=False, language = language, cascade_task2=cascade_task2)

if semi_supervised:
    semi_supervised_dataset = datasets_exist.exist_2021(
        "/data/frodriguez/data_mlm/input/semi_supervised_label_23022023_10k.tsv", 
        #"/data/frodriguez/data_mlm/input/semi_supervised_label_exist2022_24022023_10k.tsv",     
    sample = config["fine_tuning"]["sample"], basenet = basenet_tokenizer, 
    concat_metwo=False, text_cleaner=True, balance_metwo=False, language = language, cascade_task2=cascade_task2)
    train_loader = DataLoader(dataset=ConcatDataset([train_dataset, semi_supervised_dataset]),
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)
else:
    train_loader = DataLoader(dataset=train_dataset, 
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)

validation_loader = DataLoader(
    dataset=datasets_exist.exist_2021(
        config["fine_tuning"]["validation_dataset"], 
    sample = config["fine_tuning"]["sample_validation"], basenet = basenet_tokenizer, text_cleaner=True, language = language, cascade_task2=cascade_task2), 
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)

#%%
if task==1 and cascade_task1:
    #model = transformer_sbert_lstm_model_('xlm-roberta-base', 'sentence-transformers/stsb-xlm-r-multilingual', num_labels=2, sbert_freeze=True)
    model = transformer_sbert_lstm_model_(model_mlm, 'sentence-transformers/stsb-xlm-r-multilingual', num_labels=2, sbert_freeze=True)
    #model = transformer_sbert_lstm_model_(model_mlm, '/data/frodriguez/data_mlm/models/sbert/sentence-transformers/stsb-xlm-r-multilingual-2023-02-26', num_labels=2, sbert_freeze=True)
    #model = Transformer_lstm_model(model_mlm, num_labels=2)
elif task==2 and cascade_task2:
    #model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=5)
    #model = transformer_sbert_lstm_model_('xlm-roberta-base', 'sentence-transformers/stsb-xlm-r-multilingual', num_labels=5, focal_loss=True, sbert_freeze=True)
    #model = AutoModelForSequenceClassification.from_pretrained(model_mlm, num_labels=5)
    model = transformer_sbert_lstm_model_(model_mlm, 'sentence-transformers/stsb-xlm-r-multilingual', num_labels=5, focal_loss=True, sbert_freeze=True)
    #model = Transformer_lstm_model(model_mlm, num_labels=5, focal_loss=True)
elif task==1:
    model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base')
    #model = AutoModelForSequenceClassification.from_pretrained(model_mlm)
elif task==2:
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
    scheduler=scheduler, task=task, early_stopping_tolerance=20)

# %%
