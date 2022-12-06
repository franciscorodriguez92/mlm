#%%
import numpy as np
from statistics import mode
import utils.datasets_exist as datasets
from torch.utils.data import DataLoader
import torch
from utils.utils import load_model
import json
from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import classification_report
#%%
def soft_voting(predicted_probas : list) -> np.array:

    sv_predicted_proba = np.mean(predicted_probas, axis=0)
    #sv_predicted_proba[:,-1] = 1 - np.sum(sv_predicted_proba[:,:-1], axis=1)    

    return sv_predicted_proba.argmax(axis=1)

def maximum_voting(predicted_probas : list) -> np.array:
    sv_predicted_proba = np.max(predicted_probas, axis=0)
    #sv_predicted_proba[:,-1] = 1 - np.sum(sv_predicted_proba[:,:-1], axis=1)    

    return sv_predicted_proba.argmax(axis=1)

def hard_voting(predictions : list) -> list:
    return [mode(v) for v in np.array(predictions).T]

#%% 
def get_result_test_ensemble(model, dataloader, device, task):
    model=load_model(model, device)
    model.eval()
    probas, true_labels, predictions = [], [], []
    if task == 'multitask':
        probas_task2, true_labels_task2, predictions_task2 = [], [], []
    with torch.no_grad():
      for batch in dataloader:
          data = batch[0]
          b_input_ids = data[0].squeeze()
          b_input_mask = data[1].squeeze()  
          b_labels = batch[1].squeeze() 
          b_input_ids = b_input_ids.to(device, dtype=torch.long)
          b_input_mask = b_input_mask.to(device, dtype=torch.long)
          b_labels = b_labels.to(device, dtype=torch.long)
          if task != 'multitask':
              logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
          else:
              logits_task1, logits_task2 = model(input_id=b_input_ids, token_type_id=None, mask_id=b_input_mask)
          if task != "multitask":
            #logits = logits.detach().cpu().numpy()
            logits = logits.softmax(-1).detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            probas.append(logits)
            true_labels.append(label_ids)
          else:
            logits_task1 = logits_task1.detach().cpu().numpy()
            logits_task2 = logits_task2.detach().cpu().numpy()
            labels_ids = b_labels.to('cpu').numpy()
            probas.append(logits_task1)
            probas_task2.append(logits_task2)
            true_labels.append(labels_ids)
    if task == 'multitask':
        #Task1
        for i in range(len(true_labels)):
            pred_=np.argmax(probas[i], axis=1)
            predictions.append(pred_)
        ids = np.concatenate(true_labels).ravel()
        predictions = np.concatenate(predictions).ravel()
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))
        #Task2
        for i in range(len(true_labels)):
            pred_=np.argmax(probas_task2[i], axis=1)
            predictions_task2.append(pred_)
        predictions_task2 = np.concatenate(predictions_task2).ravel()
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))
        return [ids, predictions, predictions_task2]
    else:
        for i in range(len(true_labels)):
            pred_=np.argmax(probas[i], axis=1)
            predictions.append(pred_)
        ids = np.concatenate(true_labels).ravel()
        predictions = np.concatenate(predictions).ravel()
        return [ids, predictions, probas]
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))


#%%
config_file = open('config.json')
config = json.load(config_file)

#%%
args_seed = 123
models = config["ensemble"]["models"]
tokenizers = config["ensemble"]["tokenizers"]
test_path = config["ensemble"]["test_path"]
output_path = config["ensemble"]["output_path"]
batch_size = config["ensemble"]["BATCH_SIZE"]
sample = config["ensemble"]["sample"]
task = config["ensemble"]["TASK"]
test_case = config["ensemble"]["test_case"]
ensemble_type = config["ensemble"]["ensemble_type"]
Path(os.path.split(output_path)[0]).mkdir(parents=True, exist_ok=True) 
#%% 
print("-------------- MLM Preprocessing INPUTS: --------------")
print("Data training path: " , output_path)
print("Models: ", models)

#%%
#use_cuda = not args_cuda and torch.cuda.is_available()
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args_seed)
if(torch.cuda.is_available()):
	torch.cuda.manual_seed(args_seed)
	torch.backends.cudnn.benchmark = True
np.random.seed(args_seed)

print("\nDevice: " + str(device) +"; Seed: "+str(args_seed))


#%% 
probas_ = []
predictions_ = []
for tokenizer, model in zip(tokenizers, models):
    print("Tonekizer: ", tokenizer)
    print("Model: ", model)
    dataset = datasets.exist_2021(test_path, sample = sample, basenet = tokenizer, 
        is_test = True)
    test_data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size)
    ids, preds, probas = get_result_test_ensemble(
        model, test_data_loader, device, task)
    probas_.append(np.concatenate(probas))
    predictions_.append(preds)

#%%
if ensemble_type == "hard":
    predictions = hard_voting(predictions_)
elif ensemble_type == "mean":
    predictions = soft_voting(np.asarray(probas_))
else:
    predictions = maximum_voting(np.asarray(probas_))

#%%
df_pred = pd.read_table(test_path, sep="\t", dtype={'id': 'str'})
if sample:
    df_pred=df_pred.sample(frac=0.01, random_state=123)

df_pred['id_'] = ids
df_pred['predictions'] = predictions

#%%
if task==1:
    df_pred['category']=df_pred['predictions'].map({ 0: 'non-sexist', 1: 'sexist'})
elif task==2:
    df_pred['category']=df_pred['predictions'].map({0: 'non-sexist', 1: 'ideological-inequality', 
    2: 'stereotyping-dominance', 3: 'objectification', 4: 'sexual-violence', 
    5: 'misogyny-non-sexual-violence'})
df_pred=df_pred[['id', 'test_case', 'category']]
df_pred.to_csv(output_path, sep="\t", index=False)
#%% Evaluation 
if test_case == 'EXIST2021':
    test_path_labeled = config["ensemble"]["gold_standard_exist_2021"]
else:
    test_path_labeled = config["ensemble"]["gold_standard_exist_2022"]
gold_standard = pd.read_table(test_path_labeled, sep="\t", dtype=str)
gold_standard_merge = gold_standard.merge(df_pred)
# %%
print("Spanish report::")
gold_standard_merge_es = gold_standard_merge[gold_standard_merge['language']=='es']
print(classification_report(gold_standard_merge_es['task' + str(task)], 
    gold_standard_merge_es['category'], digits=4))

print("English report::")
gold_standard_merge_en = gold_standard_merge[gold_standard_merge['language']=='en']
print(classification_report(gold_standard_merge_en['task' + str(task)], 
    gold_standard_merge_en['category'], digits=4))

print("Global report::")
print(classification_report(gold_standard_merge['task' + str(task)], 
    gold_standard_merge['category'], digits=4))

# %%
