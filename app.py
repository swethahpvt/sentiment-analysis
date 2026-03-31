from flask import Flask, request, jsonify
import joblib
import torch
import os
import gdown
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = Flask(__name__)

# -----------------------------
# Download BERT model from Google Drive
# -----------------------------
if not os.path.exists("bert_model/model.safetensors"):
    os.makedirs("bert_model", exist_ok=True)
    print("Downloading BERT model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1YePgS3o6fKu_noegQwpkhn3PEGBMGb7A",
        "bert_model/model.safetensors",
        quiet=False
    )
    print("Download complete!")

# -----------------------------
# Load SVM
# -----------------------------
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Load BERT
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model")
bert_model = DistilBertForSequenceClassification.from_pretrained("bert_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# -----------------------------
# Label mapping
# -----------------------------
labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# -----------------------------
# Home route
# -----------------------------
@app.route("/")
def home():
    return "Sentiment Analysis API is running 🚀"

# -----------------------------
# Predict route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"})

    # -------- SVM Prediction --------
    X = vectorizer.transform([text])
    svm_pred = svm_model.predict(X)[0]

    # -------- BERT Prediction --------
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

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
```

---

## Step 2 — Update `requirements.txt`

Open `requirements.txt` and make sure it has exactly this:
```
flask
joblib
torch
transformers
scikit-learn
gdown