# 🚀 Multi-Word Expressions (MWEs) Detection App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fyp-mwedetection.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the source code for my Final Year Project (FYP): a web-based application built with **Streamlit** and **PyTorch** to detect Multi-Word Expressions (MWEs) in text. 

The application allows users to test state-of-the-art NLP models on single sentences or perform batch predictions via Excel file uploads.

## ✨ Features
* **Multiple Deep Learning Architectures:** Compare predictions between BERT, BERT-CRF, RoBERTa-CRF, and LSTM-CRF models.
* **Real-time Sentence Prediction:** Type a sentence and instantly view token-level predictions and extracted MWEs.
* **Batch Processing (Excel):** Upload an Excel file containing multiple sentences, and download an updated file with predicted labels attached.
* **Interactive UI:** Built entirely in Python using Streamlit.

## 🧠 Models & Performance
The models were fine-tuned to classify tokens into `B-MWE`, `I-MWE`, and `O` tags. 

| Model | Accuracy | F1-Score (MWE) |
|---|---|---|
| **BERT** | 86.10% | 0.54 |
| **BERT-CRF** | 84.99% | 0.53 |
| **RoBERTa-CRF** | 88.71% | 0.66 |
| **LSTM-CRF** | 85.77% | 0.49 |

## 🛠️ Installation & Setup (Local)
To run this application locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/FYP-AllModel.git
   cd FYP-AllModel
