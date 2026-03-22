import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

# Define the BERT Model
class BertModel(nn.Module):
    def __init__(self, model_name, num_labels_mwe):
        super(BertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier_mwe = nn.Linear(self.bert.config.hidden_size, num_labels_mwe)

    def forward(self, input_ids, attention_mask, labels_mwe=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits_mwe = self.classifier_mwe(sequence_output)

        if labels_mwe is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits_mwe.view(-1, logits_mwe.shape[-1]), labels_mwe.view(-1))
            return loss
        else:
            return logits_mwe
        
# Define the BERT-CRF Model
class BertCRFModel(nn.Module):
    def __init__(self, model_name, num_labels_mwe):
        super(BertCRFModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier_mwe = nn.Linear(self.bert.config.hidden_size, num_labels_mwe)
        self.crf = CRF(num_labels_mwe, batch_first=True)

    def forward(self, input_ids, attention_mask, labels_mwe=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits_mwe = self.classifier_mwe(sequence_output)
        mask = attention_mask.bool()

        if labels_mwe is not None:
            labels_mwe = labels_mwe.clone()
            labels_mwe[labels_mwe == -100] = 0
            loss = -self.crf(logits_mwe, labels_mwe, mask=mask)
            return loss
        else:
            predictions = self.crf.decode(logits_mwe, mask=mask)
            return predictions

# Define the RoBERTa-CRF Model
class RoBertaCRFModel(nn.Module):
    def __init__(self, model_name, num_labels_mwe):
        super(RoBertaCRFModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier_mwe = nn.Linear(self.roberta.config.hidden_size, num_labels_mwe)
        self.crf = CRF(num_labels_mwe, batch_first=True)

    def forward(self, input_ids, attention_mask, labels_mwe=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits_mwe = self.classifier_mwe(sequence_output)

        mask = attention_mask.bool()
        if labels_mwe is not None:
            labels_mwe = labels_mwe.clone()
            labels_mwe[labels_mwe == -100] = 0
            loss = -self.crf(logits_mwe, labels_mwe, mask=mask)
            return loss
        else:
            predictions = self.crf.decode(logits_mwe, mask=mask)
            return predictions

# Define the LSTM-CRF Model
class LSTMCRFModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, pad_idx):
        super(LSTMCRFModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels_mwe=None):
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out)

        if labels_mwe is not None:
            mask = input_ids != pad_idx  # Mask non-padding tokens
            loss = -self.crf(logits, labels_mwe, mask=mask)
            return loss
        else:
            predictions = self.crf.decode(logits)
            return predictions
