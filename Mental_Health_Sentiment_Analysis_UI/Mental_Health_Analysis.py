import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os
import zipfile

# Extract model if needed
MODEL_DIR = "saved_mental_status_bert"
ZIP_FILE = "saved_mental_status_bert.zip"
LABEL_ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# Download stopwords if not available
nltk.download('stopwords')

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Function to clean input text
def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)  # Remove special characters
    statement = re.sub(r'\d+', '', statement)  # Remove numbers
    words = statement.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Function to predict mental health status
def detect_anxiety(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]

# Streamlit App UI
st.title("🌿 Mental Health Status Detection")
st.write("This tool analyzes mental health-related text and predicts its category (e.g., normal, anxiety, depression, etc.).")

st.subheader("📝 Enter your mental health statement:")
input_text = st.text_area("Write your thoughts here...", height=150)

if st.button("🔍 Detect Mental State"):
    if input_text.strip():
        predicted_class = detect_anxiety(input_text)
        st.success(f"The Predicted Mental Health State is: *{predicted_class}*")
    else:
        st.error("⚠️ Please enter a valid statement.")

# Run the app using: streamlit run Mental_Health_Analysis.py
