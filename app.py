import os
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd

app = Flask(__name__)

# -------- Load the fine-tuned AraBART model --------
arabart_model_path = r"fine_tuned_arabart"
arabart_model = AutoModelForSeq2SeqLM.from_pretrained(arabart_model_path)
arabart_tokenizer = AutoTokenizer.from_pretrained(arabart_model_path)

# -------- Load the test dataset --------
test_data_path = os.path.join("test", "test_subset.csv")
test_df = pd.read_csv(test_data_path).dropna().reset_index(drop=True)

# -------- Summary generation function --------
def generate_summary(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -------- Main Page --------
@app.route('/')
def index():
    return render_template('index.html')

# -------- API: Generate Summary --------
@app.route('/generate_summary', methods=['POST'])
def generate_summary_route():
    user_input = request.form['article_text']
    arabart_summary = generate_summary(arabart_model, arabart_tokenizer, user_input)

    return jsonify({
        "arabart_summary": arabart_summary
    })

# -------- API: Get Test Samples --------
@app.route('/get_test_samples', methods=['GET'])
def get_test_samples():
    samples = test_df[['article', 'summary']].head(10).to_dict(orient='records')
    return jsonify(samples)

# -------- Run the app --------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
