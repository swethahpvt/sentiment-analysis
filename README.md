Sentiment Analysis API

This project is a simple and efficient API that analyzes text and predicts whether the sentiment is Positive, Neutral, or Negative.

It combines both a traditional machine learning model and a modern deep learning model to give reliable predictions. The goal was to build something that is not just accurate, but also fast and easy to use in real-world applications.

Live Demo
API is live at: https://sentiment-analysis-1-c0lp.onrender.com

Models Used
🔹 SVM + TF-IDF
  A lightweight and fast model
  Works well for quick predictions
  Good baseline for comparison
  
🔹 DistilBERT
  A transformer-based deep learning model
  Understands context better than traditional models
  Provides more accurate predictions

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

Run Locally

1. Clone the repo
git clone https://github.com/your-username/sentiment-analysis-api.git
cd sentiment-analysis-api
2. Install dependencies
pip install -r requirements.txt
3. Run the app
python app.py

Then open:
http://127.0.0.1:5000/

Where this can be used

Analyzing customer reviews
Monitoring social media sentiment
Feedback systems
Chatbots or support tools

Future Improvements

Add a simple frontend (like Streamlit)
Support more languages
Improve model performance with more data
