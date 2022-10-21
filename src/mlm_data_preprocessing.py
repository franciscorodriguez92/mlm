#%% 
import os
from pathlib import Path
import datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import multiprocessing
import json
#%% inputs
""" disable_caching_dataset = True
sample = 100
MAX_SEQ_LEN = 128
MODEL = "xlm-roberta-base"
language = 'en'
mlm_probability = 0.15
data_path = "../data/metwo-unlabeled-data/corpus/"
data_path_save = "data/processed"
data_path_save=os.path.join("../",data_path_save, language) """

#%%
config_file = open('config.json')
config = json.load(config_file)

#%% inputs
disable_caching_dataset = True
sample = config["preprocessing"]["sample"]
MAX_SEQ_LEN = config["preprocessing"]["max_seq_len"]
MODEL = config["preprocessing"]["tokenizer"]
language = config["preprocessing"]["language"]
mlm_probability = config["preprocessing"]["mlm_probability"]
data_path = config["preprocessing"]["data_path"]
data_path_save = config["preprocessing"]["data_processing_save"]


#%% 
print("-------------- MLM Preprocessing INPUTS: --------------")
print("Data path: " , data_path)
print("Sample raw data: ", sample)
print("Model: ", MODEL)
print("Max sequence lenght: ", MAX_SEQ_LEN)
print("Path processed data: ", data_path_save)
print("MLM probability: ", mlm_probability)

#%%
if disable_caching_dataset:
    from datasets import disable_caching
    disable_caching()

#%%
en_files = [str(x) for x in Path(data_path+'/en/').glob("*.csv")]
es_files = [str(x) for x in Path(data_path+'/es/').glob("*.csv")]
if language == "es":
    all_files = es_files
    data_path_save=os.path.join(data_path_save, language)
elif language == "en":
    all_files = en_files
    data_path_save=os.path.join(data_path_save, language)
else:
    all_files = en_files+es_files

#%%
print("Loading all data from ", data_path)
unlabeled_data = datasets.load_dataset(
    'csv',
    data_files=all_files,
    keep_in_memory =False,
)

#%%
columns_to_remove = unlabeled_data.column_names['train']
columns_to_remove.remove('text')
unlabeled_data = unlabeled_data.remove_columns(columns_to_remove)

#%%
tokenizer = AutoTokenizer.from_pretrained(MODEL)

#%%
if sample != False:
    print('Sampling ', sample,' rows from dataset')
    unlabeled_data = unlabeled_data["train"].train_test_split(
        train_size=sample, test_size=1,  seed=42
    )

#%%
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
)

def tokenize_function(row):
    return tokenizer(
        row['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_special_tokens_mask=True)

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

def tokenize_and_mask(batch):
    batch_tokenized = tokenize_function(batch)
    return insert_random_mask(batch_tokenized)

#%%
print("Tokenizing and masking raw data, this will take a while...")
train_dataset = unlabeled_data.map(
    tokenize_and_mask,
    batched=True,
    #num_proc=multiprocessing.cpu_count(),
    remove_columns=unlabeled_data['train'].column_names,
)

train_dataset = train_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
print("Tonenization finished")
#%%
print("Saving processed dataset, , this will take a while...")
Path(data_path_save).mkdir(parents=True, exist_ok=True) 
train_dataset.save_to_disk(data_path_save)
print("Processed finished!")
# %%
