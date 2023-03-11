#%%
import os
import json
from transformers import AutoModelForSequenceClassification
#from transformers import AutoTokenizer
from utils import datasets_exist, train_transformer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
from pathlib import Path
#from utils.architectures import Transformer_lstm_model, transformer_sbert_lstm_model
import pandas as pd
from utils.datasets_exist import TextCleaner

#%%
config_file = open('config.json')
config = json.load(config_file)

# %%
TRAIN_BATCH_SIZE = config["fine_tuning"]["TRAIN_BATCH_SIZE"]
LEARNING_RATE = config["fine_tuning"]["LEARNING_RATE"]
basenet_tokenizer = config["fine_tuning"]["basenet_tokenizer"]
task=1
schedule=config["fine_tuning"]["schedule"]
epochs=config["fine_tuning"]["EPOCHS"]
#######################################################
model_path_save = config["fine_tuning"]["MODEL_PATH_SAVE"]
#######################################################
language = config["fine_tuning"]["language"]
cascade_task2 = config["fine_tuning"]["cascade_task2"]
cascade_task1 = config["fine_tuning"]["cascade_task1"]
Path(os.path.split(model_path_save)[0]).mkdir(parents=True, exist_ok=True) 

#%% 
print("-------------- MLM Training INPUTS: --------------")
print("Path model: ", model_path_save)

#%%
#ideological-inequality       
#non-sexist                   
#objectification              
#misogyny-non-sexual-violence 
#stereotyping-dominance       
#sexual-violence    
train_class = "ideological-inequality"
test_class = "objectification"
train_path = "../data/input/crossed_training_train.tsv"
test_path = "../data/input/crossed_training_test.tsv"

#%% 
train_file = pd.read_table(train_path, sep="\t")
train_file=train_file[(train_file['task2']=="non-sexist")|(train_file['task2']==train_class)]
test_file = pd.read_table(test_path, sep="\t")
test_file=test_file[(test_file['task2']=="non-sexist")|(test_file['task2']==test_class)]

train_file['task1']=train_file['task1'].map({'non-sexist' : 0, 'sexist': 1})
train_file['task2']=train_file['task1']
test_file['task1']=test_file['task1'].map({'non-sexist' : 0, 'sexist': 1})
test_file['task2']=test_file['task1']
if config["fine_tuning"]["sample"]:
    train_file=train_file.sample(frac=0.01, random_state=1)
    test_file=test_file.sample(frac=0.1, random_state=1)

#%%
preprocessor = TextCleaner(filter_users=True, filter_hashtags=False, 
                           filter_urls=True, convert_hastags=False, lowercase=False, 
                           replace_exclamation=False, replace_interrogation=False, 
                           remove_accents=False, remove_punctuation=False)
train_file['text'] = train_file['text'].apply(lambda row: preprocessor(row))
test_file['text'] = test_file['text'].apply(lambda row: preprocessor(row))

#%% 
dataset_train=datasets_exist.exist_2021(
        config["fine_tuning"]["train_dataset"], 
    sample = config["fine_tuning"]["sample"], basenet = basenet_tokenizer, 
    concat_metwo=False, text_cleaner=False, balance_metwo=False, language = language, cascade_task2=cascade_task2)
dataset_train.data = train_file
print("Loading data...")
train_loader = DataLoader(
    dataset=dataset_train, 
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)

dataset_test = datasets_exist.exist_2021(
        config["fine_tuning"]["validation_dataset"], 
    sample = config["fine_tuning"]["sample_validation"], basenet = basenet_tokenizer, text_cleaner=False, language = language, cascade_task2=cascade_task2)
dataset_test.data = test_file
validation_loader = DataLoader(
    dataset=dataset_test, 
    batch_size=TRAIN_BATCH_SIZE, shuffle=True)

#%%
#model = AutoModelForSequenceClassification.from_pretrained(model_mlm)
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")
#%%
#model = transformer_sbert_lstm_model('xlm-roberta-base', 'sentence-transformers/stsb-xlm-r-multilingual', sbert_freeze=True)
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
