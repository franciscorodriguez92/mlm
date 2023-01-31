#%%
from sentence_transformers import SentenceTransformer, losses, InputExample
#import os
from torch.utils.data import DataLoader
#import csv
import math
from datetime import datetime
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers.trainer_utils import set_seed
set_seed(0)
#%%
# Read the dataset
model_name = 'sentence-transformers/stsb-xlm-r-multilingual'
train_batch_size = 16
num_epochs = 4
#model_save_path = "models/sbert/"
#model_save_path = os.path.join("../",model_save_path)
sample=0.000001
filename='../data/input/sbert/EXIST2021_training_split_cl.csv'
model_save_path = '../models/sbert/'+model_name+'-'+datetime.now().strftime("%Y-%m-%d")
# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
print("Read train dataset")

#%%
import pandas as pd
df = pd.read_csv(filename)
df = df.reset_index()
df = df.sample(frac=sample, random_state=123)
train_samples = []
for index, row in df.iterrows():
    #print(row['c1'], row['c2'])
    inp_example = InputExample(texts=[row['sentence_1'], row['sentence_2']], label=float(row['label']))
    train_samples.append(inp_example)
#%%
""" train_samples = []
with open(filename, 'r', encoding="utf8") as file:
  csvreader = csv.DictReader(file)
  for id, row in enumerate(csvreader):
    #print(row)
    #print(row['sentence_1'])
    #print(id)
    inp_example = InputExample(texts=[row['sentence_1'], row['sentence_2']], label=float(row['label']))
    train_samples.append(inp_example)
    if id>=sample: break """


#%%
""" train_samples = []
dev_samples = []
test_samples = []
reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
for row in reader:
    score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
    inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

    if row['split'] == 'dev':
        dev_samples.append(inp_example)
    elif row['split'] == 'test':
        test_samples.append(inp_example)
    else:
        train_samples.append(inp_example) """


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
#train_loss = losses.CosineSimilarityLoss(model=model)
train_loss = losses.OnlineContrastiveLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
#print("Read STSbenchmark dev dataset")
evaluator = BinaryClassificationEvaluator.from_input_examples(train_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
print("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=2000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

# %%
""" model_loaded = SentenceTransformer(model_save_path)
sentences = ["This is an example sentence", "Each sentence is converted"]

#model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
embeddings = model_loaded.encode(sentences)
print(embeddings)

# %%
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
model = AutoModel.from_pretrained(model_save_path)

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings) """
# %%
