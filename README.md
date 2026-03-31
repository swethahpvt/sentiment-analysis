Sentiment Analysis API

A machine learning API that predicts sentiment (Positive, Neutral, Negative) using SVM and DistilBERT models.

Live Demo
API is live at: https://sentiment-analysis-1-c0lp.onrender.com

Models Used
- SVM with TF-IDF Vectorizer (fast, lightweight)
- DistilBERT (deep learning, more accurate)

API Usage
Check if API is running
GET https://sentiment-analysis-1-c0lp.onrender.com/
Get Sentiment Prediction
POST https://sentiment-analysis-1-c0lp.onrender.com/predict

Request Body:
json
{
    "text": "This product is amazing!"
}

Response:
json
{
    "text": "This product is amazing!",
    "svm_prediction": "Positive",
    "bert_prediction": "Positive"
}


Tech Stack
- Python
- Flask
- Transformers (DistilBERT)
- Scikit-learn
- Deployed on Render
