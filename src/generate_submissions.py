#%%
from utils.utils import generate_submission
import torch
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import classification_report

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

args = parser.parse_args()


#%%
model_path = args.model_path
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
model_path_save = '../models/fine-tuned/bert_test.pt'
basenet = 'roberta'
test_path = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test.tsv'
output_path = '../submissions/submission.tsv'
batch_size = 16
sample = True
args_cuda = False
args_seed = 123

task = 1
test_case = 'EXIST2021'
language=None

#%% 
print("-------------- MLM Preprocessing INPUTS: --------------")
print("Data training path: " , output_path)
print("Model: ", model_path_save)

#%%
use_cuda = not args_cuda and torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args_seed)
if(use_cuda):
	torch.cuda.manual_seed(args_seed)
	torch.backends.cudnn.benchmark = True
np.random.seed(args_seed)

print("\nDevice: " + str(device) +"; Seed: "+str(args_seed))

#%%
df_pred = generate_submission(
    model_path_save, basenet, device, test_path, output_path, task, batch_size, sample, language)    

#%% Evaluation 
if test_case == 'EXIST2021':
    test_path_labeled = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test_labeled.tsv'
else:
    test_path_labeled = '../data/input/EXIST 2022 Dataset/test/test_EXIST2022_labeled.tsv'
gold_standard = pd.read_table(test_path_labeled, sep="\t", dtype=str)

gold_standard_merge = gold_standard.merge(df_pred)
# %%
if task==1:
    print(classification_report(gold_standard_merge['task1'], gold_standard_merge['category']))
else:
    print(classification_report(gold_standard_merge['task2'], gold_standard_merge['category']))
