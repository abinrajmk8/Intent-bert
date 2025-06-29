from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load tokenizer and model from local directory
MODEL_DIR = "./intent-bert"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Label map used during training (must match training order)
label_map = {
    0: "Affiliation", 1: "AwardYear", 2: "Awarded", 3: "BirthDate", 4: "BirthPlace",
    5: "CauseOfDeath", 6: "Children", 7: "Citizenship", 8: "CoFounded", 9: "CollaboratedWith",
    10: "DeathDate", 11: "DeathPlace", 12: "Discovered", 13: "Employer", 14: "Field",
    15: "Founded", 16: "Influenced", 17: "Invention", 18: "KnownFor", 19: "Membership",
    20: "Mentor", 21: "Nationality", 22: "PublishedWork", 23: "Religion", 24: "Residence",
    25: "Spouse", 26: "Student", 27: "StudiedAt", 28: "Theory", 29: "WorkedAt"
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence = data.get("text", "")

    if not sentence:
        return jsonify({"error": "No input text provided"}), 400

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        intent = label_map.get(predicted_class, "unknown")

    return jsonify({"intent": intent})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)