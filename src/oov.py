#%%
import pandas as pd
from transformers import AutoTokenizer
import json
#%%
MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

#%%
#%%
config_file = open('config.json')
config = json.load(config_file)
#%%
test_path_labeled = config["inference"]["gold_standard_exist_2021"]
train_path_labeled = "../data/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2021_training.tsv"
test_path_labeled_2022 = config["inference"]["gold_standard_exist_2022"]
files = [test_path_labeled, train_path_labeled, test_path_labeled_2022]
#files = [test_path_labeled_2022]
gold_standard = pd.concat(
    (pd.read_table(f, sep="\t", dtype=str) for f in files), ignore_index=True)

#%%
def oov_words(tokenizer, text):
  """
  Returns: number of tokens OOV, number of tokens, number of words
  """
  num_words = len(text.split())
  ids = tokenizer.encode(text, add_special_tokens = False)
  num_tokens = len(ids)
  num_oov = ids.count(tokenizer.convert_tokens_to_ids(tokenizer._unk_token))
  return pd.Series([num_oov, num_tokens, num_words])

#%%
gold_standard[['num_oov', 'num_tokens', 'num_words']] = gold_standard.apply(
    lambda row: oov_words(tokenizer, row['text']), axis = 1)
#%%
gold_standard['num_oov'].sum()
#%%
gold_standard['num_tokens'].sum()
#%% 
gold_standard['num_words'].sum()