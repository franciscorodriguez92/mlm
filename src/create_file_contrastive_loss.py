#%%
import pandas as pd
from itertools import combinations
# %%
filename = "../data/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2021_training_split.tsv"
filename = "../data/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2021_training_split.tsv"
df_train = pd.read_table(filename, sep="\t")
# %%
def all_same(items):
    return str(1) if all(x == items[0] for x in items) else str(0)
list_contrastive = []
for index, index2 in zip(combinations(df_train['text'],2), combinations(df_train['task1'],2)):
    list_contrastive.append(list(index)+list(all_same(index2)))
    #print(list(index)+list(all_same(index2)))
    #print(all_same(index2))
    #print(index[0], index[1])
#%%
df_train_cl=pd.DataFrame.from_records(list_contrastive, columns=['sentence_1', 'sentence_2', 'label'])

# %%
#df_train_cl = df_train_cl.rename(columns={'sentence_1': 'sentence_1', 'sentence_2': 'sentence_2', 'sentence_3': 'label'})
df_train_cl.to_csv('EXIST2021_training_split_cl.csv', index=False)
# %%

