
#%%
#import os
from pathlib import Path
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import utils.datasets_exist as datasets
from utils.datasets_exist import TextCleaner

#%%
config_file = open('config.json')
config = json.load(config_file)
#%%
data_path = config["preprocessing"]["data_path"]
basenet = config["inference"]["basenet_tokenizer"]
test_path = config["inference"]["test_path"]
output_path = config["inference"]["output_path"]
batch_size = config["inference"]["BATCH_SIZE"]
sample = config["inference"]["sample"]
model_path = config["inference"]["MODEL_PATH_SAVE"]
language = config["inference"]["language"]
threshold = 0.183
model_path_task2=model_path
output_file = "semi_supervised_label.tsv"

#%%
en_files = [str(x) for x in Path(data_path+'/en/').glob("*.csv")]
es_files = [str(x) for x in Path(data_path+'/es/').glob("*.csv")]
#%%
all_files = en_files+es_files
all_files = en_files[:5]
#%%
li = []
for filename in all_files:
    df = pd.read_csv(filename, dtype='str')
    df['status_id'] = df['status_id'].str.replace('x', '')
    df['status_id'] = df['status_id'].astype(np.int64)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
# %%
frame = frame[['status_id', 'text', 'lang']]
frame.rename(columns={'status_id': 'id', 
                      'text': 'text',
                      'lang': 'language'}, inplace=True)
frame = frame.sample(frac=0.0001, replace=True, random_state=1)
preprocessor = TextCleaner(filter_users=True, filter_hashtags=False, 
               filter_urls=True, convert_hastags=False, lowercase=False, 
               replace_exclamation=False, replace_interrogation=False, 
               remove_accents=False, remove_punctuation=False)
frame['text'] = frame['text'].apply(lambda row: preprocessor(row))

#%%
#use_cuda = not args_cuda and torch.cuda.is_available()
args_seed = 123
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args_seed)
if(torch.cuda.is_available()):
	torch.cuda.manual_seed(args_seed)
	torch.backends.cudnn.benchmark = True
np.random.seed(args_seed)

print("\nDevice: " + str(device) +"; Seed: "+str(args_seed))


#%%
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    return model

#%%
def get_result_test(model, dataloader, device):
    model.eval()
    logits, true_labels, predictions, probas = [], [], [], []
    with torch.no_grad():
      for batch in dataloader:
          data = batch[0]
          b_input_ids = data[0].squeeze()
          b_input_mask = data[1].squeeze()  
          b_labels = batch[1].squeeze() 
          b_input_ids = b_input_ids.to(device, dtype=torch.long)
          b_input_mask = b_input_mask.to(device, dtype=torch.long)
          b_labels = b_labels.to(device, dtype=torch.long)
          logits_ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
          probas.append(torch.max(torch.softmax(logits_, dim=-1), 1)[0].detach().cpu().tolist())
          #probas.append(torch.max(torch.softmax(logits_, dim=-1)).detach().cpu())
          logits_ = logits_.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()
          logits.append(logits_)
          true_labels.append(label_ids)
    for i in range(len(true_labels)):
        pred_=np.argmax(logits[i], axis=1)
        predictions.append(pred_)
    ids = np.concatenate(true_labels).ravel()
    predictions = np.concatenate(predictions).ravel()
    probas = np.concatenate(probas).ravel()
    return [ids, predictions, probas]
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))

#%%
dataset = datasets.exist_2021(test_path, sample = sample, basenet = basenet, is_test = True, language = language)
dataset.data=frame

#%%
test_data_loader = DataLoader(
        dataset=dataset,
        #dataset=torch.utils.data.ConcatDataset([dataset, dataset]),
        shuffle=False,
        batch_size=batch_size)
model=load_model(model_path, device)
model_task2=load_model(model_path_task2, device)
ids, predictions, probas = get_result_test(model, test_data_loader, device)
ids_task2, predictions_task2, probas_task2 = get_result_test(model_task2, test_data_loader, device)

# %%
frame['task1']=predictions
frame['probas']=probas
frame['task1']=frame['task1'].map({0: 'non-sexist', 1: 'ideological-inequality', 2: 'stereotyping-dominance', 3: 'objectification', 4: 'sexual-violence', 5: 'misogyny-non-sexual-violence'})
# %%
frame['task2']=predictions_task2
frame['probas_task2']=probas_task2
frame['task2']=frame['task2'].map({0: 'non-sexist', 1: 'ideological-inequality', 2: 'stereotyping-dominance', 3: 'objectification', 4: 'sexual-violence', 5: 'misogyny-non-sexual-violence'})

# %%
output=frame[(frame['probas']>= threshold) & (frame['probas_task2']>= threshold) ]
#%%
output['test_case'] = 'semi-supervised'
output['source'] = 'twitter'
output = output[['test_case', 'id', 'source', 'language', 'text', 'task1', 'task2']]
#test_case	id	source	language	text	task1	task2

# %%
output.to_csv(output_file, sep="\t", index=False)
# %%
