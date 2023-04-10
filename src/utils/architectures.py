import torch
from transformers import AutoModel
from torch.nn import CrossEntropyLoss

class FocalLoss(torch.nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, 
            reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

class Transformer_lstm_model(torch.nn.Module):
    def __init__(self, model_mlm, num_labels=2, focal_loss=False):
          super(Transformer_lstm_model, self).__init__()
          self.focal_loss = focal_loss
          self.num_labels = num_labels
          self.model_mlm = model_mlm
          self.encoder = AutoModel.from_pretrained(self.model_mlm)
          self.lstm = torch.nn.LSTM(768, 256, batch_first=True,bidirectional=True)
          self.dropout = torch.nn.Dropout(0.1)
          self.linear = torch.nn.Linear(256*2, self.num_labels)

          

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
          sequence_output = self.encoder(
               input_ids, 
               attention_mask=attention_mask,
               token_type_ids= token_type_ids) ['last_hidden_state']
          #print("sequence_output", sequence_output)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          lstm_output, (h,c) = self.lstm(sequence_output) # extract the 1st token's embeddings
          hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
          hidden = self.dropout(hidden)
          linear_output = self.linear(hidden.view(-1,256*2)) # assuming that you are only using the output of the last LSTM cell to perform classification
          if labels is not None:
              if self.focal_loss:
                  loss_fct = FocalLoss(ignore_index=-1)
              else:
                  loss_fct = CrossEntropyLoss(ignore_index=-1)
              #print("labels::", self.num_labels)
              #print("labels::", labels)

              loss = loss_fct(linear_output.view(-1, self.num_labels), labels.view(-1))
              #loss2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels[:,1].view(-1))
              #loss = loss1 + loss2
              #print("LOSS::::::::::::::", loss.item())
              return (loss,)
          else:
              return (linear_output,)

class transformer_sbert_lstm_model(torch.nn.Module):
    def __init__(self, model_mlm, model_mlm2, num_labels=2, focal_loss=False, sbert_freeze=False):
          super(transformer_sbert_lstm_model, self).__init__()
          self.focal_loss = focal_loss
          self.num_labels = num_labels
          self.model_mlm = model_mlm
          self.model_mlm2 = model_mlm2
          self.encoder = AutoModel.from_pretrained(self.model_mlm)
          self.encoder2 = AutoModel.from_pretrained(self.model_mlm2)
          self.lstm = torch.nn.LSTM(768, 256, batch_first=True,bidirectional=True)
          self.dropout = torch.nn.Dropout(0.1)
          #self.linear = torch.nn.Linear(256*2, self.num_labels)
          self.linear_sbert = torch.nn.Linear(768, 512)
          #self.linear = torch.nn.Linear((256*2) + 768, (256*2))
          self.linear = torch.nn.Linear((256*2) + 512, (256*2)+512)
          self.linear2 = torch.nn.Linear((256*2)+512, self.num_labels)
          if sbert_freeze:
            print("Freezing encoder 2!")
            for name, param in self.encoder2.named_parameters():                
                if param.requires_grad is not None:
                    param.requires_grad = False
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
          
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
          outputs = self.encoder(
               input_ids, 
               attention_mask=attention_mask,
               token_type_ids= token_type_ids,
               output_attentions=False,
               output_hidden_states=False)
          sequence_output = outputs ['last_hidden_state']
          outputs_encoder2 = self.encoder2(
               input_ids, 
               attention_mask=attention_mask)
          sentence_embeddings = self.mean_pooling(outputs_encoder2, attention_mask)
          #print("sequence_output", outputs[1].shape)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          lstm_output, (h,c) = self.lstm(sequence_output) # extract the 1st token's embeddings
          hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
          #hidden = self.dropout(hidden)
          #linear_output = self.linear(hidden.view(-1,256*2))
          sentence_embeddings = self.linear_sbert(sentence_embeddings)
          linear_output = self.linear(torch.cat((hidden.view(-1,256*2), sentence_embeddings), 1))
          linear_output = self.dropout(linear_output)
          linear_output = self.linear2(linear_output) 
          loss = None
          if labels is not None:
              if self.focal_loss:
                  loss_fct = FocalLoss(ignore_index=-1)
              else:
                  loss_fct = CrossEntropyLoss(ignore_index=-1)
              #print("labels::", self.num_labels)
              #print("labels::", labels)

              loss = loss_fct(linear_output.view(-1, self.num_labels), labels.view(-1))
              #loss2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels[:,1].view(-1))
              #loss = loss1 + loss2
              #rint("LOSS::::::::::::::", loss.item())
              #return (loss,)
              return (loss,)
          else:
              return (linear_output,)


class transformer_sbert_lstm_model_(torch.nn.Module):
    def __init__(self, model_mlm, model_mlm2, num_labels=2, focal_loss=False, sbert_freeze=False):
          super(transformer_sbert_lstm_model_, self).__init__()
          self.focal_loss = focal_loss
          self.num_labels = num_labels
          self.model_mlm = model_mlm
          self.model_mlm2 = model_mlm2
          self.encoder = AutoModel.from_pretrained(self.model_mlm)
          self.encoder2 = AutoModel.from_pretrained(self.model_mlm2)
          self.lstm = torch.nn.LSTM(768, 256, batch_first=True,bidirectional=True)
          self.dropout = torch.nn.Dropout(0.1)
          #self.linear = torch.nn.Linear(256*2, self.num_labels)
          #self.linear_sbert = torch.nn.Linear(768, 512)
          #self.linear = torch.nn.Linear((256*2) + 768, (256*2))
          self.linear = torch.nn.Linear((256*2) + 768, (256*2)+768)
          self.linear2 = torch.nn.Linear((256*2)+768, self.num_labels)
          if sbert_freeze:
            print("Freezing encoder 2!")
            for name, param in self.encoder2.named_parameters():                
                if param.requires_grad is not None:
                    param.requires_grad = False
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
          
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
          outputs = self.encoder(
               input_ids, 
               attention_mask=attention_mask,
               token_type_ids= token_type_ids,
               output_attentions=False,
               output_hidden_states=False)
          sequence_output = outputs ['last_hidden_state']
          outputs_encoder2 = self.encoder2(
               input_ids, 
               attention_mask=attention_mask)
          sentence_embeddings = self.mean_pooling(outputs_encoder2, attention_mask)
          #print("sentence_embeddings:::: ", sentence_embeddings)
          #print("sequence_output:::: ", sequence_output)
          #print("sequence_output", outputs[1].shape)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          lstm_output, (h,c) = self.lstm(outputs_encoder2['last_hidden_state']) # extract the 1st token's embeddings
          hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
          #hidden = self.dropout(hidden)
          #linear_output = self.linear(hidden.view(-1,256*2))
          #sentence_embeddings = self.linear_sbert(sentence_embeddings)
          linear_output = self.linear(torch.cat((hidden.view(-1,256*2), outputs[1]), 1))
          linear_output = self.dropout(linear_output)
          linear_output = self.linear2(linear_output) 
          loss = None
          if labels is not None:
              if self.focal_loss:
                  loss_fct = FocalLoss(ignore_index=-1)
              else:
                  loss_fct = CrossEntropyLoss(ignore_index=-1)
              #print("labels::", self.num_labels)
              #print("labels::", labels)

              loss = loss_fct(linear_output.view(-1, self.num_labels), labels.view(-1))
              #loss2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels[:,1].view(-1))
              #loss = loss1 + loss2
              #rint("LOSS::::::::::::::", loss.item())
              #return (loss,)
              return (loss,)
          else:
              return (linear_output,)


class transformer_sbert_lstm_model_v1(torch.nn.Module):
    def __init__(self, model_mlm, model_mlm2, num_labels=2, focal_loss=False, sbert_freeze=False):
          super(transformer_sbert_lstm_model_v1, self).__init__()
          self.focal_loss = focal_loss
          self.num_labels = num_labels
          self.model_mlm = model_mlm
          self.model_mlm2 = model_mlm2
          self.encoder = AutoModel.from_pretrained(self.model_mlm)
          self.encoder2 = AutoModel.from_pretrained(self.model_mlm2)
          self.lstm = torch.nn.LSTM(768, 256, batch_first=True,bidirectional=True)
          self.dropout = torch.nn.Dropout(0.1)
          #self.linear = torch.nn.Linear(256*2, self.num_labels)
          self.linear = torch.nn.Linear((256*2) + 768, (256*2))
          self.linear2 = torch.nn.Linear((256*2), self.num_labels)
          if sbert_freeze:
            print("Freezing encoder 2!")
            for name, param in self.encoder2.named_parameters():                
                if param.requires_grad is not None:
                    param.requires_grad = False
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
          
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
          outputs = self.encoder(
               input_ids, 
               attention_mask=attention_mask,
               token_type_ids= token_type_ids,
               output_attentions=False,
               output_hidden_states=False)
          sequence_output = outputs ['last_hidden_state']
          outputs_encoder2 = self.encoder2(
               input_ids, 
               attention_mask=attention_mask)
          sentence_embeddings = self.mean_pooling(outputs_encoder2, attention_mask)
          #print("sequence_output", outputs[1].shape)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          lstm_output, (h,c) = self.lstm(sequence_output) # extract the 1st token's embeddings
          hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
          hidden = self.dropout(hidden)
          #linear_output = self.linear(hidden.view(-1,256*2))
          linear_output = self.linear(torch.cat((hidden.view(-1,256*2), sentence_embeddings), 1))
          linear_output = self.linear2(linear_output) 
          loss = None
          if labels is not None:
              if self.focal_loss:
                  loss_fct = FocalLoss(ignore_index=-1)
              else:
                  loss_fct = CrossEntropyLoss(ignore_index=-1)
              #print("labels::", self.num_labels)
              #print("labels::", labels)

              loss = loss_fct(linear_output.view(-1, self.num_labels), labels.view(-1))
              #loss2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels[:,1].view(-1))
              #loss = loss1 + loss2
              #rint("LOSS::::::::::::::", loss.item())
              #return (loss,)
              return (loss,)
          else:
              return (linear_output,)