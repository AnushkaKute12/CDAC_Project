# CDAC_Project

# Big Data Project link : https://github.com/AnushkaKute12/Big_Data_Project.git

## Mental Health Sentiment Analysis
[[View Here]](http://3.95.171.236:8501/)

This project is an AI-driven mental health detection system that uses BERT (Bidirectional Encoder Representations from Transformers) to classify text into different mental health categories such as anxiety, depression, bipolar disorder, or normal state. The model is fine-tuned using Hugging Face's Transformers library and deployed using Streamlit for a simple and interactive web interface.

### 🚀 Project Overview

Preprocessing: Text is cleaned by removing special characters, numbers, and stopwords.

Model Training: BERT-based model fine-tuned on a labeled dataset for mental health classification.

Inference Pipeline: Tokenized text is passed through the model to predict the mental health category.

Web UI: Built using Streamlit for easy user interaction.

Model Deployment: The trained model and tokenizer are stored as a ZIP file and extracted dynamically during execution.

### 🛠 Features

✔ Fine-tuned BERT-based classification model.

✔ Text cleaning using NLP techniques.

✔ Interactive Streamlit web app for easy predictions.

✔ On-the-fly model extraction to save storage space.

### 🖥 How to Run Locally

1 Clone the repository

git clone https://github.com/your-repo-name.git

cd your-repo-name

2 Install dependencies

pip install -r requirements.txt

3 Run the Streamlit app

streamlit run Mental_Health_Analysis.py

### 📂 Dataset

The dataset contains mental health-related text labeled with categories like normal, anxiety, depression, bipolar, suicide, etc..

The labels are encoded using LabelEncoder.

### 📜 Model Training

Fine-tuned bert-base-uncased for multi-class classification.

Used Hugging Face's Trainer API with optimized hyperparameters.

### 📊 Evaluation Metrics

✔ Confusion Matrix for misclassification analysis.

✔ Classification Report (Precision, Recall, F1-Score).

✔ Accuracy, Precision, Recall, and F1-score calculation.

### 📦 Requirements

See requirements.txt for dependencies.

### 📊 Tableau Visualizations

Tableau dashboards provide insights into data distribution, model performance.

[[View Here]](https://public.tableau.com/app/profile/anushka.kute/viz/ML_Project_Charts1/FinalDashboard)

### 📌 Future Enhancements

🔹 Add more advanced NLP techniques (e.g., attention-based analysis).

🔹 Integrate with a database for real-time feedback and learning.

