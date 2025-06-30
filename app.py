import os
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd

app = Flask(__name__)

# -------- Load the fine-tuned AraBART model --------
arabart_model_path = r"C:\Users\hp\Desktop\arabart_summarizer\fine_tuned_arabart"
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
from flask import Flask, request, jsonify, render_template
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import os

app = Flask(__name__)

# Load models (ensure paths are correct)
pretrained_model_name = "google/mt5-small"
pretrained_model = MT5ForConditionalGeneration.from_pretrained(pretrained_model_name)
pretrained_tokenizer = MT5Tokenizer.from_pretrained(pretrained_model_name)

fine_tuned_model_1_path = "./fine_tuned_model_10000"
fine_tuned_model_1 = MT5ForConditionalGeneration.from_pretrained(fine_tuned_model_1_path)
fine_tuned_tokenizer_1 = MT5Tokenizer.from_pretrained(fine_tuned_model_1_path)

fine_tuned_model_2_path = "./fine_tuned_model_49000_new"
fine_tuned_model_2 = MT5ForConditionalGeneration.from_pretrained(fine_tuned_model_2_path)
fine_tuned_tokenizer_2 = MT5Tokenizer.from_pretrained(fine_tuned_model_2_path)

fine_tuned_model_3_path = "./fine_tuned_model_arasum-xlsum_preprocessed"
fine_tuned_model_3 = MT5ForConditionalGeneration.from_pretrained(fine_tuned_model_3_path)
fine_tuned_tokenizer_3 = MT5Tokenizer.from_pretrained(fine_tuned_model_3_path)

# Helper function to generate summaries
def generate_summary(model, tokenizer, text):
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_summary', methods=['POST'])
def generate_summary_route():
    user_input = request.form['article_text']
    pretrained_summary = generate_summary(pretrained_model, pretrained_tokenizer, user_input)
    fine_tuned_summary_1 = generate_summary(fine_tuned_model_1, fine_tuned_tokenizer_1, user_input)
    fine_tuned_summary_2 = generate_summary(fine_tuned_model_2, fine_tuned_tokenizer_2, user_input)
    fine_tuned_summary_3 = generate_summary(fine_tuned_model_3, fine_tuned_tokenizer_3, user_input)
    return jsonify({
        "pretrained_summary": pretrained_summary,
        "fine_tuned_summary_1": fine_tuned_summary_1,
        "fine_tuned_summary_2": fine_tuned_summary_2,
        "fine_tuned_summary_3": fine_tuned_summary_3
    })

# Vercel serverless handler
def handler(request):
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    from werkzeug.serving import run_simple
    from io import StringIO
    import sys

    # Capture output for debugging
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    # Run the app
    response = run_simple('0.0.0.0', 3000, app, use_reloader=False)

    # Restore stdout and return response
    sys.stdout = old_stdout
    return response.get_data()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
