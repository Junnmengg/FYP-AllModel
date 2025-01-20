import pandas
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from torch import nn
import string
from huggingface_hub import hf_hub_download
import nltk
import json
import io
import openpyxl
import xlsxwriter
import os

Username1 = os.getenv()

# Download tokenizer data
nltk.download('punkt_tab')

st.set_page_config(
    page_title="MWEs Prediction App",
    page_icon="🚀"
)

# Function to check if a token is punctuation
def is_punctuation(token):
    return all(char in string.punctuation for char in token)

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

# Load tokenizer and model from Hugging Face
@st.cache_resource
def load_model_and_tokenizer(model_name, repo_id, token, filename, lstm_vocab=None, pad_idx=None):
    if model_name == "LSTM-CRF":
        # LSTM-CRF Model
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
        model_weights_path = hf_hub_download(
            repo_id="Junmengg/FYP-LSTM-CRF", 
            filename=filename, 
            use_auth_token=token
        )
        state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))

        # Handle missing CRF keys
        missing_keys = ["crf.start_transitions", "crf.end_transitions", "crf.transitions"]
        for key in missing_keys:
            if key not in state_dict:
                print(f"Initializing missing key: {key}")
                num_labels = model.crf.num_tags
                if "start_transitions" in key or "end_transitions" in key:
                    state_dict[key] = torch.zeros(num_labels)
                elif "transitions" in key:
                    state_dict[key] = torch.zeros(num_labels, num_labels)

        # Load updated state_dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, tokenizer, vocab

    elif model_name == "BERT":
        # BERT Model
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=token)
        model = BertModel('bert-base-uncased', num_labels_mwe=len(mwe_label_to_id))
    elif model_name == "BERT-CRF" or model_name == "RoBERTa-CRF":
        # CRF-enhanced models (BERT-CRF or RoBERTa-CRF)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=token)
        if model_name == "BERT-CRF":
            model = BertCRFModel('bert-base-uncased', num_labels_mwe=len(mwe_label_to_id))
        elif model_name == "RoBERTa-CRF":
            model = RoBertaCRFModel('roberta-base', num_labels_mwe=len(mwe_label_to_id))
    else:
        raise ValueError("Unsupported model name!")

    # Load model weights
    model_weights_path = hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=token)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()

    # Return appropriate outputs based on model type
    if model_name == "LSTM-CRF":
        return model, tokenizer, vocab
    else:
        return model, tokenizer

# Map tag indices to labels
mwe_label_to_id = {'B-MWE': 0, 'I-MWE': 1, 'O': 2}
idx2tag = {v: k for k, v in mwe_label_to_id.items()}

# Function for MWE Detection
def perform_mwe_detection(sentence, model, tokenizer=None, model_name=None, vocab=None, pad_idx=None):
    """
    Perform MWE detection based on the selected model.

    Args:
        sentence (str): Input sentence to analyze.
        model (nn.Module): The loaded model (BERT, BERT-CRF, RoBERTa-CRF, or LSTM-CRF).
        tokenizer (AutoTokenizer): Tokenizer for BERT-based models (None for LSTM-CRF).
        model_name (str): Name of the model (e.g., "BERT", "BERT-CRF", "RoBERTa-CRF", "LSTM-CRF").
        vocab (dict): Vocabulary for LSTM-CRF (None for other models).
        pad_idx (int): Padding index for LSTM-CRF (None for other models).

    Returns:
        table_data (list): Token-level predictions with tags.
        detected_mwes (list): Detected MWEs (multi-word expressions).
    """
    # Handle tokenization differently for LSTM-CRF
    if model_name == "LSTM-CRF":
        from nltk.tokenize import word_tokenize
        
        # Tokenize and map tokens to vocab indices
        tokenized_sentence = word_tokenize(sentence.lower())
        indexed_sentence = [vocab.get(word, vocab["<UNK>"]) for word in tokenized_sentence]

        # Pad or truncate to fixed length
        if len(indexed_sentence) < 128:
            indexed_sentence += [pad_idx] * (128 - len(indexed_sentence))
        else:
            indexed_sentence = indexed_sentence[:128]

        # Convert to tensor
        input_ids = torch.tensor([indexed_sentence])
        attention_mask = None  # Not used in LSTM-CRF
        tokens = tokenized_sentence  # Use tokenized_sentence directly
    else:
        # Tokenization for BERT, BERT-CRF, and RoBERTa-CRF
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

    # Make predictions
    with torch.no_grad():
        if model_name in ["BERT-CRF", "RoBERTa-CRF", "LSTM-CRF"]:
            # CRF-based models return decoded sequences (list of predictions)
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            crf_predictions = predictions[0]  # First batch (list)
        elif model_name == "BERT":
            # BERT (non-CRF) returns logits as tensor
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = logits.squeeze(0)  # Remove batch dimension
            crf_predictions = torch.argmax(logits, dim=-1).tolist()  # Convert to list for compatibility
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    # Decode tokens and predictions
    current_mwe_tokens = []
    detected_mwes = []
    table_data = []  # For token-level predictions

    for token, pred_idx in zip(tokens, crf_predictions):
        # Skip special tokens for BERT-based models
        if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]']:
            continue

        clean_token = token.lstrip("Ġ")  # Remove special token prefixes (e.g., for RoBERTa)
        pred_tag = idx2tag.get(pred_idx, 'O')  # Get tag from prediction index

        # If punctuation, force label to "O"
        if is_punctuation(clean_token):
            pred_tag = 'O'

        # Add to table data
        table_data.append({"Token": clean_token, "Prediction": pred_tag})

        # Collect MWEs
        if pred_tag in ['B-MWE', 'I-MWE']:
            current_mwe_tokens.append(clean_token)
        elif current_mwe_tokens:  # End of MWE
            mwe_text = ' '.join(current_mwe_tokens)
            detected_mwes.append(mwe_text)
            current_mwe_tokens = []

    # Capture remaining MWE tokens
    if current_mwe_tokens:
        mwe_text = ' '.join(current_mwe_tokens)
        detected_mwes.append(mwe_text)

    # Return both token-level table data and detected MWEs
    return table_data, detected_mwes

# Sidebar for page navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Sentence Prediction"

with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            color: #1E90FF;
            margin-bottom: 20px;
        }
        .sidebar-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
        }
        .sidebar-button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">Navigation Bar</div>', unsafe_allow_html=True)
    
    # Manage navigation state
    if st.button("✍️ Sentence Prediction"):
        st.session_state["page"] = "Sentence Prediction"
    if st.button("📄 Excel File Prediction"):
        st.session_state["page"] = "Excel File Prediction"

# Navigate based on session state
if st.session_state["page"] == "Sentence Prediction":
    st.title("✍️ MWE Detection with Different Deep Learning Models")
    st.write(
        """
        This app detects **Multi-Word Expressions (MWEs)** in a sentence using different fine-tuned 
        **Deep Learning Model**.
        """
    )

    # Add a selectbox to choose the model
    selected_model = st.selectbox("Choose the model:", ["BERT", "BERT-CRF", "RoBERTa-CRF", "LSTM-CRF"])

    if selected_model == "BERT":
        with st.spinner("Loading BERT model..."):
            model, tokenizer = load_model_and_tokenizer(
                "BERT", "Junmengg/FYP-BERT", "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", "bert_model_weights.pth"
            )
    elif selected_model == "BERT-CRF":
        with st.spinner("Loading BERT-CRF model..."):
            model, tokenizer = load_model_and_tokenizer(
                "BERT-CRF", "Junmengg/FYP-BERT-CRF", "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", "bert_crf_model_weights.pth"
            )
    elif selected_model == "RoBERTa-CRF":
        with st.spinner("Loading RoBERTa-CRF model..."):
            model, tokenizer = load_model_and_tokenizer(
                "RoBERTa-CRF", "Junmengg/FYP-RoBERTa-CRF", "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", "roberta_crf_model_weights.pth"
            )
    elif selected_model == "LSTM-CRF":
        with st.spinner("Loading LSTM-CRF model..."):
            # Download the vocabulary file from Hugging Face
            vocab_path = hf_hub_download(repo_id="Junmengg/FYP-LSTM-CRF", filename="lstm_crf_vocab.json", use_auth_token="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")
            with open(vocab_path, "r") as f:
                vocab = json.load(f)

            pad_idx = vocab["<PAD>"]  # Ensure <PAD> token exists in vocab
            model, tokenizer, vocab = load_model_and_tokenizer(
                "LSTM-CRF", None, "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", "lstm_crf_model_weights.pth", lstm_vocab=vocab, pad_idx=pad_idx
            )

    # Define test set performance metrics for each model
    model_performance = {
        "BERT": {
            "accuracy": 0.8610,
            "classification_report": {
                "Metric": ["Precision", "Recall", "F1-Score", "Support"],
                "MWE": [0.51, 0.58, 0.54, 1415],
                "Micro Avg": [0.51, 0.58, 0.54, 1415],
                "Macro Avg": [0.51, 0.58, 0.54, 1415],
                "Weighted Avg": [0.51, 0.58, 0.54, 1415],
            },
        },
        "BERT-CRF": {
            "accuracy": 0.8499,
            "classification_report": {
                "Metric": ["Precision", "Recall", "F1-Score", "Support"],
                "MWE": [0.49, 0.57, 0.53, 1443],
                "Micro Avg": [0.49, 0.57, 0.53, 1443],
                "Macro Avg": [0.49, 0.57, 0.53, 1443],
                "Weighted Avg": [0.49, 0.57, 0.53, 1443],
            },
        },
        "RoBERTa-CRF": {
            "accuracy": 0.8871,
            "classification_report": {
                "Metric": ["Precision", "Recall", "F1-Score", "Support"],
                "MWE": [0.62, 0.70, 0.66, 1396],
                "Micro Avg": [0.62, 0.70, 0.66, 1396],
                "Macro Avg": [0.62, 0.70, 0.66, 1396],
                "Weighted Avg": [0.62, 0.70, 0.66, 1396],
            },
        },
        "LSTM-CRF": {
            "accuracy": 0.8577,
            "classification_report": {
                "Metric": ["Precision", "Recall", "F1-Score", "Support"],
                "MWE": [0.49, 0.48, 0.49, 1137],
                "Micro Avg": [0.49, 0.48, 0.49, 1137],
                "Macro Avg": [0.49, 0.48, 0.49, 1137],
                "Weighted Avg": [0.49, 0.48, 0.49, 1137],
            },
        },
    }    

    # Display the selected model's performance metrics
    st.markdown(f'<h3 style="color:skyblue;">Model Performance on the Test Set ({selected_model})</h3>', unsafe_allow_html=True)
    selected_performance = model_performance[selected_model]
    st.write(f"**Accuracy**: {selected_performance['accuracy']:.4f}")

    classification_df = pandas.DataFrame.from_dict(
        selected_performance["classification_report"], orient="index"
    )
    classification_df.columns = classification_df.iloc[0]  # Set the first row as column headers
    classification_df = classification_df[1:]  # Remove the first row
    st.table(classification_df)

    # Input box for user sentence
    user_sentence = st.text_input("Type your sentence here:")

    if user_sentence:
        # Check if the selected model is LSTM-CRF and pass additional parameters
        if selected_model == "LSTM-CRF":
            table_data, detected_mwes = perform_mwe_detection(
                user_sentence, model, tokenizer, selected_model, vocab, pad_idx
            )
        else:
            # For other models, pass only the model name and tokenizer
            table_data, detected_mwes = perform_mwe_detection(
                user_sentence, model, tokenizer, selected_model
            )

        # Display token-level predictions
        st.markdown('<h3 style="color:skyblue;">Token-Level Predictions</h3>', unsafe_allow_html=True)
        st.table(table_data)

        # Highlight detected MWEs
        st.markdown('<h3 style="color:skyblue;">Detected MWEs</h3>', unsafe_allow_html=True)
        if detected_mwes:
            for mwe in detected_mwes:
                st.markdown(f'<span style="color:green; font-weight:bold;">- [{mwe}]</span>', unsafe_allow_html=True)
        else:
            st.info("No MWEs detected in the sentence.")

        # Add some spacing for aesthetics
        st.markdown("---")

elif st.session_state["page"] == "Excel File Prediction":
    st.title("📄 MWE Detection on Excel File")
    st.write(
        """
        This app detects **Multi-Word Expressions (MWEs)** in an excel file using different fine-tuned 
        **Deep Learning Model**.
        """
    )

    # Add a selectbox to choose the model
    selected_model = st.selectbox("Choose the model:", ["BERT", "BERT-CRF", "RoBERTa-CRF", "LSTM-CRF"])

    # Load the selected model
    if selected_model == "BERT":
        with st.spinner("Loading BERT model..."):
            model, tokenizer = load_model_and_tokenizer(
                "BERT", "Junmengg/FYP-BERT", "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", "bert_model_weights.pth"
            )
    elif selected_model == "BERT-CRF":
        with st.spinner("Loading BERT-CRF model..."):
            model, tokenizer = load_model_and_tokenizer(
                "BERT-CRF", "Junmengg/FYP-BERT-CRF", "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", "bert_crf_model_weights.pth"
            )
    elif selected_model == "RoBERTa-CRF":
        with st.spinner("Loading RoBERTa-CRF model..."):
            model, tokenizer = load_model_and_tokenizer(
                "RoBERTa-CRF", "Junmengg/FYP-RoBERTa-CRF", "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", "roberta_crf_model_weights.pth"
            )
    elif selected_model == "LSTM-CRF":
        with st.spinner("Loading LSTM-CRF model..."):
            vocab_path = hf_hub_download(
                repo_id="Junmengg/FYP-LSTM-CRF", filename="lstm_crf_vocab.json", use_auth_token="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"
            )
            with open(vocab_path, "r") as f:
                vocab = json.load(f)
            pad_idx = vocab["<PAD>"]
            model, tokenizer, vocab = load_model_and_tokenizer(
                "LSTM-CRF", None, "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", "lstm_crf_model_weights.pth", lstm_vocab=vocab, pad_idx=pad_idx
            )

    # Upload excel file
    uploaded_file = st.file_uploader("Upload your excel file:", type=["xlsx"])

    if uploaded_file:
        # Read the uploaded Excel file
        df = pandas.read_excel(uploaded_file)

        # Check if 'Sentence' column exists
        if "Sentence" not in df.columns:
            st.error("The uploaded file must contain a 'Sentence' column.")
        else:
            # Perform MWE detection on each sentence
            detected_mwes = []
            
            # Perform MWE detection for each sentence in the uploaded Excel file
            for sentence in df["Sentence"]:
                if selected_model == "LSTM-CRF":
                    table_data, mwes = perform_mwe_detection(sentence, model, tokenizer, selected_model, vocab, pad_idx)
                else:
                    table_data, mwes = perform_mwe_detection(sentence, model, tokenizer, selected_model)

                # Process MWEs into desired format
                if not mwes:
                    detected_mwes.append("")  # No MWE detected -> Blank
                elif len(mwes) == 1:
                    detected_mwes.append(mwes[0])  # Single MWE -> No trailing comma
                else:
                    detected_mwes.append(", ".join(mwes))  # Multiple MWEs -> Comma-separated

            # Add Detected MWEs column to the DataFrame
            df["Detected MWEs"] = detected_mwes

            # Display the updated DataFrame
            st.markdown('<h3 style="color:skyblue;">Updated Excel File</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(10))

            # Convert the DataFrame to an Excel file for download
            @st.cache_data
            def convert_df_to_excel(dataframe):
                output = io.BytesIO()
                with pandas.ExcelWriter(output, engine="xlsxwriter") as writer:
                    dataframe.to_excel(writer, index=False, sheet_name="Sheet1")
                processed_file = output.getvalue()
                return processed_file

            # Create and download the Excel file
            excel_file = convert_df_to_excel(df)
            st.download_button(
                label="📥 Download Updated Excel File",
                data=excel_file,
                file_name="Updated_Dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
