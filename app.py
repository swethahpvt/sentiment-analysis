from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
import os
import gdown
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = Flask(__name__)
CORS(app)
 
svm_model = None
vectorizer = None
tokenizer = None
bert_model = None
device = None
labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
 
def load_models():
    global svm_model, vectorizer, tokenizer, bert_model, device
    if svm_model is not None:
        return
    if not os.path.exists("bert_model/model.safetensors"):
        os.makedirs("bert_model", exist_ok=True)
        print("Downloading BERT model from Google Drive...", flush=True)
        url = "https://drive.google.com/uc?id=1YePgS3o6fKu_noegQwpkhn3PEGBMGb7A&confirm=t"
        gdown.download(url, "bert_model/model.safetensors", quiet=False)
        print("Download complete!", flush=True)
    print("Loading SVM model...", flush=True)
    svm_model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("Loading BERT model...", flush=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model")
    bert_model = DistilBertForSequenceClassification.from_pretrained("bert_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()
    print("All models loaded!", flush=True)
 
@app.route("/")
def home():
    return "Sentiment Analysis API is running!"
 
@app.route("/health")
def health():
    return jsonify({"status": "ok"})
 
@app.route("/predict", methods=["POST"])
def predict():
    load_models()
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "No text provided"}), 400
    text = data["text"]
    X = vectorizer.transform([text])
    svm_pred = svm_model.predict(X)[0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        bert_pred = torch.argmax(outputs.logits, dim=1).item()
    return jsonify({
        "text": text,
        "svm_prediction": labels[svm_pred],
        "bert_prediction": labels[bert_pred]
    })
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
