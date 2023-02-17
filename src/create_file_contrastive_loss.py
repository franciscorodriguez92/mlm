#%%
import pandas as pd
from itertools import combinations
from utils.datasets_exist import TextCleaner
# %%
#filename = "/data/frodriguez/data_mlm/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2021_training_split.tsv"
filename = "../data/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2021_training_split.tsv"
df_train = pd.read_table(filename, sep="\t")
preprocessor = TextCleaner(filter_users=True, filter_hashtags=True, 
                           filter_urls=True, convert_hastags=False, lowercase=False, 
                           replace_exclamation=False, replace_interrogation=False, 
                           remove_accents=False, remove_punctuation=False)
df_train['text'] = df_train['text'].apply(lambda row: preprocessor(row))
print("Transforming the file...")

# %%
#mask_seeds = "/data/frodriguez/data_mlm/input/seed_terms.txt"
mask_seeds = '../data/input/seed_terms.txt'
#mask_seeds = False
if mask_seeds:
    seeds = pd.read_table(mask_seeds, sep="\t")
    seeds = seeds.drop_duplicates()
    seeds = seeds['terms'].tolist()
    seeds_pattern = '|'.join(seeds)
    df_train['text_masked'] = df_train['text'].str.replace(
        seeds_pattern, '<mask>')


#%%

""" list_contrastive = []
for index, row in df_train.iterrows():
    list_contrastive.append([row['text'], row['text_masked']]+list(str(1)))
for index, index2 in zip(combinations(df_train['text'],2), combinations(df_train['task1'],2)):
    list_contrastive.append(list(index)+list(str(0))) """

# %%
def all_same(items):
    return str(1) if all(x == items[0] for x in items) else str(0)

if mask_seeds:
    list_contrastive_masked = []
    for index, row in df_train.iterrows():
        list_contrastive_masked.append([row['text'], row['text_masked']]+list(str(1)))

list_contrastive = []
for index, index2 in zip(combinations(df_train['text'],2), combinations(df_train['task1'],2)):
    list_contrastive.append(list(index)+list(all_same(index2)))
    #print(list(index)+list(all_same(index2)))
    #print(all_same(index2))
    #print(index[0], index[1])
#%%
print("Creating df...")

df_train_cl=pd.DataFrame.from_records(list_contrastive, columns=['sentence_1', 'sentence_2', 'label'])
if mask_seeds:
    df_train_cl = df_train_cl.sample(frac=1, random_state=123)
    df_train_cl = df_train_cl.drop_duplicates(subset=['sentence_1'])
    df_train_cl_masked=pd.DataFrame.from_records(list_contrastive_masked, columns=['sentence_1', 'sentence_2', 'label'])
    df_train_cl = pd.concat([df_train_cl_masked, df_train_cl])
    df_train_cl = df_train_cl.sample(frac=1, random_state=123)
print("Storing the file...")
# %%
#df_train_cl = df_train_cl.rename(columns={'sentence_1': 'sentence_1', 'sentence_2': 'sentence_2', 'sentence_3': 'label'})
df_train_cl.to_csv('EXIST2021_training_split_cl_masked.csv', index=False)
#df_train_cl.to_csv('/data/frodriguez/data_mlm/input/sbert/EXIST2021_task1_preprocess_training_split_cl.csv', index=False)
# %%
print("Finished!")


# %%
