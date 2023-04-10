#%%
import os
from utils.utils import generate_submission
import torch
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import classification_report
import json
from pathlib import Path
#%%
""" parser = argparse.ArgumentParser(description = 'Generate submissions EXIST 2021')

parser.add_argument('--basenet', type = str, default = 'bert', help = 'basenet')
parser.add_argument('--model_path', type = str, default = '../models/bert_test.pt', help = 'path to save trained model (e.g. ../models/bert_test.pt)')
parser.add_argument('--test_path', type = str, default = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test.tsv', help = 'train_path')
parser.add_argument('--output_path', type = str, default = '../submissions/submission.tsv', help = 'output path for submission file')
parser.add_argument('--task', type = str, default = '1', help = 'task (1, 2 or multitask)')
parser.add_argument('--batch_size', type = int, default = 16, help = 'batch-size (default: 16)')
parser.add_argument('--sample', action = 'store_true', default = False, help = 'get a sample of 1 percent')
parser.add_argument('--no_cuda', action = 'store_true',   default = False,        help = 'disables CUDA training')
parser.add_argument('--seed', type = int, default = 123, help = 'random seed (default: 123)')

args = parser.parse_args() """


#%%
""" model_path = args.model_path
basenet = args.basenet
test_path = args.test_path
output_path = args.output_path
batch_size = args.batch_size
sample = args.sample
args_cuda = args.no_cuda
args_seed = args.seed

if args.task=='1':
    task=1
elif args.task=='2':
    task=2
else:
    task='multitask' """

#%%
""" model_path_save = '../models/fine-tuned/bert_test.pt'
basenet = 'roberta'
test_path = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test.tsv'
output_path = '../submissions/submission.tsv'
batch_size = 16
sample = True
args_cuda = False
args_seed = 123
task = 1
test_case = 'EXIST2021'
language=None """

#%%
config_file = open('config.json')
config = json.load(config_file)

#%%
args_seed = 123
model_path_save = config["inference"]["MODEL_PATH_SAVE"]
basenet = config["inference"]["basenet_tokenizer"]
test_path = config["inference"]["test_path"]
output_path = config["inference"]["output_path"]
batch_size = config["inference"]["BATCH_SIZE"]
sample = config["inference"]["sample"]
task = config["inference"]["TASK"]
test_case = config["inference"]["test_case"]
language = config["inference"]["language"]
cascade_system = config["inference"]["cascade_system"]
Path(os.path.split(output_path)[0]).mkdir(parents=True, exist_ok=True) 
#%% 
print("-------------- MLM Preprocessing INPUTS: --------------")
print("Data training path: " , output_path)
print("Model: ", model_path_save)

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
if cascade_system and language=='monolingual':
    basenet_es = config["inference"]["basenet_tokenizer_es"]
    model_path_save_es = config["inference"]["MODEL_PATH_SAVE_ES"]
    df_pred_task2 = generate_submission(
        model_path_save_es, basenet_es, device, test_path, output_path, 2, batch_size, sample, cascade_system=cascade_system)    
    df_pred_task2.rename(columns={"category": "category_task2"}, inplace=True)
    df_pred_task1 = generate_submission(
        model_path_save, basenet, device, test_path, output_path, 1, batch_size, sample)    
    df_pred_task1.rename(columns={"category": "category_task1"}, inplace=True)
    df_pred = pd.merge(df_pred_task1, df_pred_task2)
    mask = (df_pred['category_task1'] == 'non-sexist')
    df_pred['category_task2'][mask] = df_pred['category_task1']
elif language=='monolingual':
    basenet_es = config["inference"]["basenet_tokenizer_es"]
    model_path_save_es = config["inference"]["MODEL_PATH_SAVE_ES"]
    df_pred_es = generate_submission(
        model_path_save_es, basenet_es, device, test_path, output_path, task, batch_size, sample, language='es')    
    df_pred_en = generate_submission(
        model_path_save, basenet, device, test_path, output_path, task, batch_size, sample, language='en')    
    df_pred = pd.concat([df_pred_es, df_pred_en])
else:
    df_pred = generate_submission(
        model_path_save, basenet, device, test_path, output_path, task, batch_size, sample, language)    

#%% Evaluation 
if test_case == 'EXIST2021':
    test_path_labeled = config["inference"]["gold_standard_exist_2021"]
else:
    test_path_labeled = config["inference"]["gold_standard_exist_2022"]
gold_standard = pd.read_table(test_path_labeled, sep="\t", dtype=str)
gold_standard_merge = gold_standard.merge(df_pred)
gold_standard_merge.to_csv(output_path, sep="\t", index=False)
# %%

if cascade_system and language=='monolingual':
    print("Global report task1::")
    print(classification_report(gold_standard_merge['task1'], gold_standard_merge['category_task1'], digits=4))
    print("Global report task2::")
    print(classification_report(gold_standard_merge['task2'], gold_standard_merge['category_task2'], digits=4))
elif language=='monolingual' and task==2:
    print("Spanish report::")
    gold_standard_merge_es = gold_standard_merge[gold_standard_merge['language']=='es']
    print(classification_report(gold_standard_merge_es['task' + str(task)], gold_standard_merge_es['category'], digits=4))

    print("English report::")
    gold_standard_merge_en = gold_standard_merge[gold_standard_merge['language']=='en']
    print(classification_report(gold_standard_merge_en['task' + str(task)], gold_standard_merge_en['category'], digits=4))

    print("Global report::")
    print(classification_report(gold_standard_merge['task' + str(task)], gold_standard_merge['category'], digits=4))
    print("Global report task 1::")
    gold_standard_merge['category']=gold_standard_merge['category'].map({'non-sexist':'non-sexist','ideological-inequality': 'sexist', 'stereotyping-dominance': 'sexist', 'objectification': 'sexist', 'sexual-violence': 'sexist', 'misogyny-non-sexual-violence': 'sexist'})
    print(classification_report(gold_standard_merge['task1'], gold_standard_merge['category'], digits=4))

else:
    print("Spanish report::")
    gold_standard_merge_es = gold_standard_merge[gold_standard_merge['language']=='es']
    print(classification_report(gold_standard_merge_es['task' + str(task)], gold_standard_merge_es['category'], digits=4))

    print("English report::")
    gold_standard_merge_en = gold_standard_merge[gold_standard_merge['language']=='en']
    print(classification_report(gold_standard_merge_en['task' + str(task)], gold_standard_merge_en['category'], digits=4))

    print("Global report::")
    print(classification_report(gold_standard_merge['task' + str(task)], gold_standard_merge['category'], digits=4))

# %%
