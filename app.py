import streamlit as st
import pandas as pd
import json
import io
from huggingface_hub import hf_hub_download

# Import helper functions from utils.py
from utils import load_model_and_tokenizer, perform_mwe_detection

st.set_page_config(page_title="MWEs Prediction App", page_icon="🚀")

# Load Streamlit Secrets
Username1 = st.secrets["USERNAME1"]
Username2 = st.secrets["USERNAME2"]
Username3 = st.secrets["USERNAME3"]
Username4 = st.secrets["USERNAME4"]
Token = st.secrets["TOKEN"]

# Sidebar navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Sentence Prediction"

with st.sidebar:
    st.markdown("""
        <style>
        .sidebar-title { font-size: 20px; font-weight: bold; color: #1E90FF; margin-bottom: 20px; }
        .sidebar-button { display: block; width: 100%; padding: 10px; margin: 5px 0; text-align: center; font-size: 16px; font-weight: bold; color: white; background-color: #4CAF50; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; }
        .sidebar-button:hover { background-color: #45a049; }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Navigation Bar</div>', unsafe_allow_html=True)
    
    if st.button("✍️ Sentence Prediction"):
        st.session_state["page"] = "Sentence Prediction"
    if st.button("📄 Excel File Prediction"):
        st.session_state["page"] = "Excel File Prediction"

# Page 1: Sentence Prediction
if st.session_state["page"] == "Sentence Prediction":
    st.title("✍️ MWEs Detection with Different Deep Learning Models")
    st.write("This app detects **Multi-Word Expressions (MWEs)** in a sentence using different fine-tuned **Deep Learning Models**.")

    selected_model = st.selectbox("Choose the model:",["BERT", "BERT-CRF", "RoBERTa-CRF", "LSTM-CRF"])

    if selected_model == "BERT":
        with st.spinner("Loading BERT model..."):
            model, tokenizer = load_model_and_tokenizer("BERT", Username1, Token, "bert_model_weights.pth")
    elif selected_model == "BERT-CRF":
        with st.spinner("Loading BERT-CRF model..."):
            model, tokenizer = load_model_and_tokenizer("BERT-CRF", Username2, Token, "bert_crf_model_weights.pth")
    elif selected_model == "RoBERTa-CRF":
        with st.spinner("Loading RoBERTa-CRF model..."):
            model, tokenizer = load_model_and_tokenizer("RoBERTa-CRF", Username3, Token, "roberta_crf_model_weights.pth")
    elif selected_model == "LSTM-CRF":
        with st.spinner("Loading LSTM-CRF model..."):
            vocab_path = hf_hub_download(repo_id=Username4, filename="lstm_crf_vocab.json", token=Token)
            with open(vocab_path, "r") as f:
                vocab = json.load(f)
            pad_idx = vocab.get("<PAD>", 0)
            model, tokenizer, vocab = load_model_and_tokenizer("LSTM-CRF", Username4, Token, "lstm_crf_model_weights.pth", lstm_vocab=vocab, pad_idx=pad_idx)

    # Dictionary storing test metrics
    model_performance = {
        "BERT": { "accuracy": 0.8610, "classification_report": { "Metric": ["Precision", "Recall", "F1-Score", "Support"], "MWE":[0.51, 0.58, 0.54, 1415], "Micro Avg":[0.51, 0.58, 0.54, 1415], "Macro Avg":[0.51, 0.58, 0.54, 1415], "Weighted Avg":[0.51, 0.58, 0.54, 1415] } },
        "BERT-CRF": { "accuracy": 0.8499, "classification_report": { "Metric":["Precision", "Recall", "F1-Score", "Support"], "MWE":[0.49, 0.57, 0.53, 1443], "Micro Avg":[0.49, 0.57, 0.53, 1443], "Macro Avg":[0.49, 0.57, 0.53, 1443], "Weighted Avg":[0.49, 0.57, 0.53, 1443] } },
        "RoBERTa-CRF": { "accuracy": 0.8871, "classification_report": { "Metric": ["Precision", "Recall", "F1-Score", "Support"], "MWE":[0.62, 0.70, 0.66, 1396], "Micro Avg":[0.62, 0.70, 0.66, 1396], "Macro Avg":[0.62, 0.70, 0.66, 1396], "Weighted Avg":[0.62, 0.70, 0.66, 1396] } },
        "LSTM-CRF": { "accuracy": 0.8577, "classification_report": { "Metric":["Precision", "Recall", "F1-Score", "Support"], "MWE":[0.49, 0.48, 0.49, 1137], "Micro Avg":[0.49, 0.48, 0.49, 1137], "Macro Avg":[0.49, 0.48, 0.49, 1137], "Weighted Avg":[0.49, 0.48, 0.49, 1137] } },
    }    

    st.markdown(f'<h3 style="color:skyblue;">Model Performance on the Test Set ({selected_model})</h3>', unsafe_allow_html=True)
    selected_performance = model_performance[selected_model]
    st.write(f"**Accuracy**: {selected_performance['accuracy']:.4f}")

    classification_df = pd.DataFrame.from_dict(selected_performance["classification_report"], orient="index")
    classification_df.columns = classification_df.iloc[0]
    classification_df = classification_df[1:]
    st.table(classification_df)

    user_sentence = st.text_input("Type your sentence here:")

    if user_sentence:
        if selected_model == "LSTM-CRF":
            table_data, detected_mwes = perform_mwe_detection(user_sentence, model, tokenizer, selected_model, vocab, pad_idx)
        else:
            table_data, detected_mwes = perform_mwe_detection(user_sentence, model, tokenizer, selected_model)

        st.markdown('<h3 style="color:skyblue;">Token-Level Predictions</h3>', unsafe_allow_html=True)
        st.table(table_data)

        st.markdown('<h3 style="color:skyblue;">Detected MWEs</h3>', unsafe_allow_html=True)
        if detected_mwes:
            for mwe in detected_mwes:
                st.markdown(f'<span style="color:green; font-weight:bold;">-[{mwe}]</span>', unsafe_allow_html=True)
        else:
            st.info("No MWEs detected in the sentence.")
        st.markdown("---")


# Page 2: Excel File Prediction
elif st.session_state["page"] == "Excel File Prediction":
    st.title("📄 MWEs Detection on Excel File")
    st.write("This app detects **Multi-Word Expressions (MWEs)** in an excel file using different fine-tuned **Deep Learning Models**.")

    selected_model = st.selectbox("Choose the model:",["BERT", "BERT-CRF", "RoBERTa-CRF", "LSTM-CRF"])

    if selected_model == "BERT":
        with st.spinner("Loading BERT model..."):
            model, tokenizer = load_model_and_tokenizer("BERT", Username1, Token, "bert_model_weights.pth")
    elif selected_model == "BERT-CRF":
        with st.spinner("Loading BERT-CRF model..."):
            model, tokenizer = load_model_and_tokenizer("BERT-CRF", Username2, Token, "bert_crf_model_weights.pth")
    elif selected_model == "RoBERTa-CRF":
        with st.spinner("Loading RoBERTa-CRF model..."):
            model, tokenizer = load_model_and_tokenizer("RoBERTa-CRF", Username3, Token, "roberta_crf_model_weights.pth")
    elif selected_model == "LSTM-CRF":
        with st.spinner("Loading LSTM-CRF model..."):
            vocab_path = hf_hub_download(repo_id=Username4, filename="lstm_crf_vocab.json", token=Token)
            with open(vocab_path, "r") as f:
                vocab = json.load(f)
            pad_idx = vocab.get("<PAD>", 0)
            model, tokenizer, vocab = load_model_and_tokenizer("LSTM-CRF", Username4, Token, "lstm_crf_model_weights.pth", lstm_vocab=vocab, pad_idx=pad_idx)

    uploaded_file = st.file_uploader("Upload your excel file:", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        if "Sentence" not in df.columns:
            st.error("The uploaded file must contain a 'Sentence' column.")
        else:
            detected_mwes =[]
            predicted_labels = []
            
            for sentence in df["Sentence"]:
                if selected_model == "LSTM-CRF":
                    table_data, mwes = perform_mwe_detection(sentence, model, tokenizer, selected_model, vocab, pad_idx)
                else:
                    table_data, mwes = perform_mwe_detection(sentence, model, tokenizer, selected_model)

                if not mwes:
                    detected_mwes.append("")
                elif len(mwes) == 1:
                    detected_mwes.append(mwes[0])
                else:
                    detected_mwes.append(", ".join(mwes))

                labels = [entry["Prediction"] for entry in table_data]
                predicted_labels.append(", ".join(labels))

            df["Predicted Labels"] = predicted_labels
            df["Detected MWEs"] = detected_mwes

            column_order =["Sentence", "Predicted Labels", "Detected MWEs"]
            df = df[column_order]

            st.markdown('<h3 style="color:skyblue;">Updated Excel File</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(10))

            @st.cache_data
            def convert_df_to_excel(dataframe):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    dataframe.to_excel(writer, index=False, sheet_name="Sheet1")
                return output.getvalue()

            excel_file = convert_df_to_excel(df)
            st.download_button(
                label="📥 Download Updated Excel File",
                data=excel_file,
                file_name="Updated_Dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
