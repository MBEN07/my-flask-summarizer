import os
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import requests
import pandas as pd

app = Flask(__name__)

# Dynamically download the model if not present
model_path = "fine_tuned_arabart"
if not os.path.exists(model_path):
    model_url = "https://your-google-drive-link/model.safetensors"
    print("Downloading model file...")
    os.makedirs(model_path, exist_ok=True)
    response = requests.get(model_url)
    with open(os.path.join(model_path, "model.safetensors"), "wb") as f:
        f.write(response.content)

# Load the fine-tuned AraBART model
arabart_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
arabart_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the test dataset (adjust path relative to project root)
test_data_path = os.path.join("Test", "test_subset.csv")
if os.path.exists(test_data_path):
    test_df = pd.read_csv(test_data_path).dropna().reset_index(drop=True)
else:
    test_df = pd.DataFrame(columns=["article", "summary"])  # Fallback if file is missing

# Summary generation function
def generate_summary(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Main Page
@app.route('/')
def index():
    return render_template('index.html')

# API: Generate Summary
@app.route('/generate_summary', methods=['POST'])
def generate_summary_route():
    user_input = request.form['article_text']
    arabart_summary = generate_summary(arabart_model, arabart_tokenizer, user_input)
    return jsonify({"arabart_summary": arabart_summary})

# API: Get Test Samples
@app.route('/get_test_samples', methods=['GET'])
def get_test_samples():
    if not test_df.empty:
        samples = test_df[['article', 'summary']].head(10).to_dict(orient='records')
    else:
        samples = [{"article": "No test data available", "summary": ""}]
    return jsonify(samples)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
