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