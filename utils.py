import string
import torch
import streamlit as st
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import nltk

# Import models from the models.py file 
from models import BertModel, BertCRFModel, RoBertaCRFModel, LSTMCRFModel

# Ensure NLTK data is downloaded
nltk.download('punkt_tab', quiet=True)

# Map tag indices to labels
mwe_label_to_id = {'B-MWE': 0, 'I-MWE': 1, 'O': 2}
idx2tag = {v: k for k, v in mwe_label_to_id.items()}

def is_punctuation(token):
    return all(char in string.punctuation for char in token)

@st.cache_resource
def load_model_and_tokenizer(model_name, repo_id, token, filename, lstm_vocab=None, pad_idx=None):
    if model_name == "LSTM-CRF":
        vocab = lstm_vocab
        tokenizer = None
        model = LSTMCRFModel(
            vocab_size=len(vocab),
            embedding_dim=100,
            hidden_dim=256,
            num_labels=len(mwe_label_to_id),
            pad_idx=pad_idx
        )
        # Download weights
        model_weights_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))

        # Handle missing CRF keys
        missing_keys =["crf.start_transitions", "crf.end_transitions", "crf.transitions"]
        for key in missing_keys:
            if key not in state_dict:
                num_labels = model.crf.num_tags
                if "start_transitions" in key or "end_transitions" in key:
                    state_dict[key] = torch.zeros(num_labels)
                elif "transitions" in key:
                    state_dict[key] = torch.zeros(num_labels, num_labels)

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, tokenizer, vocab

    elif model_name == "BERT":
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
        model = BertModel('bert-base-uncased', num_labels_mwe=len(mwe_label_to_id))
    elif model_name == "BERT-CRF":
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
        model = BertCRFModel('bert-base-uncased', num_labels_mwe=len(mwe_label_to_id))
    elif model_name == "RoBERTa-CRF":
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
        model = RoBertaCRFModel('roberta-base', num_labels_mwe=len(mwe_label_to_id))
    else:
        raise ValueError("Unsupported model name!")

    # Load model weights for BERT-based models
    model_weights_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()

    return model, tokenizer

def perform_mwe_detection(sentence, model, tokenizer=None, model_name=None, vocab=None, pad_idx=None):
    if model_name == "LSTM-CRF":
        from nltk.tokenize import word_tokenize
        tokenized_sentence = word_tokenize(sentence.lower())
        indexed_sentence =[vocab.get(word, vocab.get("<UNK>", 0)) for word in tokenized_sentence]

        if len(indexed_sentence) < 128:
            indexed_sentence += [pad_idx] * (128 - len(indexed_sentence))
        else:
            indexed_sentence = indexed_sentence[:128]

        input_ids = torch.tensor([indexed_sentence])
        attention_mask = None
        tokens = tokenized_sentence
    else:
        encoded = tokenizer(
            sentence.lower(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        if model_name in["BERT-CRF", "RoBERTa-CRF", "LSTM-CRF"]:
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            crf_predictions = predictions[0]
        elif model_name == "BERT":
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = logits.squeeze(0)
            crf_predictions = torch.argmax(logits, dim=-1).tolist()
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    current_mwe_tokens =[]
    detected_mwes = []
    table_data =[]

    for token, pred_idx in zip(tokens, crf_predictions):
        if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']:
            continue

        clean_token = token.lstrip("Ġ")
        pred_tag = idx2tag.get(pred_idx, 'O')

        if is_punctuation(clean_token):
            pred_tag = 'O'

        table_data.append({"Token": clean_token, "Prediction": pred_tag})

        if pred_tag in['B-MWE', 'I-MWE']:
            current_mwe_tokens.append(clean_token)
        elif current_mwe_tokens:
            mwe_text = ' '.join(current_mwe_tokens)
            detected_mwes.append(mwe_text)
            current_mwe_tokens =[]

    if current_mwe_tokens:
        mwe_text = ' '.join(current_mwe_tokens)
        detected_mwes.append(mwe_text)

    return table_data, detected_mwes
