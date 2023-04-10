# %%
import os
from torch import nn
import torch.nn.functional as F
from enum import Enum
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
set_seed(0)
#%%
class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)

class ContrastiveLoss(nn.Module):
    """
    Modified from https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5, size_average:bool = True):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def forward(self, rep_anchor, rep_other, labels):
        #reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        #assert len(reps) == 2
        #rep_anchor, rep_other = reps
        #print(rep_anchor)
        #print(rep_other)
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()
# %%
class sbert_training(torch.nn.Module):
    def __init__(self, model_mlm2):
          super(sbert_training, self).__init__()
          self.model_mlm2 = model_mlm2
          self.encoder2 = AutoModel.from_pretrained(model_mlm2)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
          
    def forward(self, input_ids=None, attention_mask=None, input_ids2=None, attention_mask2=None, labels=None):
          outputs_encoder2 = self.encoder2(
               input_ids, 
               attention_mask=attention_mask)
          sentence_embeddings = self.mean_pooling(outputs_encoder2, attention_mask)
        #   outputs_encoder2 = self.encoder2(
        #        input_ids2, 
        #        attention_mask=attention_mask2)
        #   sentence_embeddings2 = self.mean_pooling(outputs_encoder2, attention_mask2)
          #print("sequence_output", outputs[1].shape)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          loss = None
          loss_fct = ContrastiveLoss()
          if labels is not None:
              outputs_encoder2 = self.encoder2(input_ids2, 
                                        attention_mask=attention_mask2)
              sentence_embeddings2 = self.mean_pooling(outputs_encoder2, attention_mask2)
              #loss = loss_fct(sentence_embeddings.view(-1), sentence_embeddings2.view(-1), labels.view(-1))
              loss = loss_fct(sentence_embeddings, sentence_embeddings2, labels)
              #loss2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels[:,1].view(-1))
              #loss = loss1 + loss2
              #rint("LOSS::::::::::::::", loss.item())
              #return (loss,)
              return (loss,)
          else:
              return (sentence_embeddings,)

#%%
class contrastive_loss_dataset(torch.utils.data.Dataset):
	'''
	Hahackathon dataset
	filename: train/val/test file to be read
	basenet : bert/ernie/roberta/deberta
	is_test : if the input file does not have groundtruth labels
	        : (for evaluation on the leaderboard)
	'''

	def __init__(self, filename, basenet= 'roberta', max_length= 128,  is_test= False, sample=False):
		super(contrastive_loss_dataset, self).__init__()
		self.is_test = is_test
		self.data    = self.read_file(filename, sample)
		if basenet == 'roberta':
			print("Tokenizer: xlm-roberta-base\n")
			self.token = AutoTokenizer.from_pretrained('xlm-roberta-base')
		else:
			print("Tokenizer: ", basenet)
			self.token = AutoTokenizer.from_pretrained(basenet)
		self.max_length = max_length

		
	def read_file(self, filename, sample):
		#df = pd.read_table(filename, sep="\t", dtype={'id': 'str'})
		df = pd.read_csv(filename)
		if sample:
			if sample is True:
				df=df.sample(frac=0.01, random_state=123)
			else:
				df=df.sample(frac=sample, random_state=123)

		print(df.shape)
		print("Sampled input from the file: {}".format(filename))
		print(df.head())

		return df

	def get_tokenized_text(self, text):		
		# marked_text = "[CLS] " + text + " [SEP]"
		encoded = self.token(text= text,  					# the sentence to be encoded
							 add_special_tokens= True,  	# add [CLS] and [SEP]
							 max_length= self.max_length,  	# maximum length of a sentence
							 padding= 'max_length',  		# add [PAD]s
							 return_attention_mask= True,  	# generate the attention mask
							 return_tensors = 'pt',  		# return PyTorch tensors
							 truncation= True
							) 

		input_id = encoded['input_ids']
		mask_id  = encoded['attention_mask']

		return input_id, mask_id

		
	def __len__(self):
		return len(self.data)
	

	def __getitem__(self, idx):
		
		sentence1  = self.data.iloc[idx]['sentence_1']
		sentence2  = self.data.iloc[idx]['sentence_2']

		label = []

		if not self.is_test:
			label.append(self.data.iloc[idx]['label'])


		label = torch.tensor(label)

		input_id, mask_id  = self.get_tokenized_text(sentence1)
		input_id2, mask_id2  = self.get_tokenized_text(sentence2)
		
		return [input_id, mask_id, input_id2, mask_id2], label


#%%
def train_one_epoch(model, dataloader, optimizer, device, scheduler=None):
    model = model.to(device)
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(dataloader):
        #b_input_ids, b_input_mask, b_labels = batch
        data = batch[0]
        b_input_ids = data[0].squeeze()
        b_input_mask = data[1].squeeze()
        b_input_ids2 = data[2].squeeze()
        b_input_mask2 = data[3].squeeze()
        b_labels = batch[1].squeeze()
        b_input_ids = b_input_ids.to(device, dtype=torch.long)
        b_input_mask = b_input_mask.to(device, dtype=torch.long)
        b_labels = b_labels.to(device, dtype=torch.long)
        b_input_ids2 = b_input_ids2.to(device, dtype=torch.long)
        b_input_mask2 = b_input_mask2.to(device, dtype=torch.long)
        optimizer.zero_grad()
        loss = model(input_ids=b_input_ids, attention_mask=b_input_mask,input_ids2=b_input_ids2, attention_mask2=b_input_mask2,labels=b_labels)[0]
        #print(loss)
        #break
        #print(b_input_ids[:,:maxlength].shape)
        #print(b_input_ids.shape)
        #print(b_input_ids[:,maxlength:].shape)
        #print("IDS::::::: ", b_input_ids.shape)
        #print("MASKS::::::: ", b_input_mask[1][0])
        #print("LABELS:::::: ", b_labels.shape)
        #print("LABELS:::::: ", b_labels)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1
        if scheduler:
            scheduler.step()
    print("\n\nTrain loss: {}".format(tr_loss/nb_tr_steps))
    return tr_loss/nb_tr_steps


#%%
TRAIN_BATCH_SIZE=16
model_path = "/data/frodriguez/data_mlm/models/sbert/sbert_2.pt"
#model_path = os.path.join("../",model_path)
epochs=4
filename="/data/frodriguez/data_mlm/input/sbert/EXIST2021_training_split_cl.csv"
encoder = 'sentence-transformers/stsb-xlm-r-multilingual'
sample=0.000001
lr=2e-5

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = contrastive_loss_dataset(filename=filename, basenet= encoder, sample=sample)
dataset_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
model = sbert_training(encoder)
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = epochs * len(dataset_loader)
num_warmup_steps = num_training_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps)

# %%
for _ in trange(epochs, desc="Epoch"):
    train_one_epoch(model, dataset_loader, optimizer, device, scheduler=None)
    print(f"\n\nSaving model at {model_path}")
    #torch.save(model, model_path)
    torch.save(model.state_dict(), 'model_weights.pth')
print("Training SBERT finished!")
# %%
""" from utils.utils import load_model
trained_model = load_model(model_path, device)

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = trained_model(encoded_input['input_ids'], encoded_input['attention_mask'])
model_output[0][1] """
# %%
