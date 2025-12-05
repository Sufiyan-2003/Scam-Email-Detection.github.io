from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging, traceback, os
from email_analyzer import EmailAnalyzer
from ml_model import EmailClassifier

app = Flask(__name__, template_folder="templates")
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

email_analyzer = EmailAnalyzer()
ml_model = EmailClassifier()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/status')
def status():
    model_status = "loaded" if getattr(ml_model, "is_trained", False) else "not_loaded"
    return jsonify({"status": model_status, "version": "3.0.0"})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        content = data.get("email_content", "").strip()
        parsed = email_analyzer.parse_email(content)
        feats = email_analyzer.extract_features(parsed)
        prediction = ml_model.predict(parsed)
        explanation = ml_model.get_prediction_explanation(parsed, feats)
        risk = calculate_risk(prediction, feats)
        label = "scam" if prediction['prediction'] == 'spam' else 'safe'
        
        return jsonify({
            "prediction": label,
            "confidence": prediction['confidence'],
            "risk_score": risk,
            "explanation": explanation
        })
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "trace": tb}), 500

def calculate_risk(pred, f):
    base = 5.0
    if pred['prediction'] == 'spam': base += pred['confidence'] * 6
    else: base -= pred['confidence'] * 5
    if f.get('scam_phrases_count', 0) > 0: base += min(3, f['scam_phrases_count'] * 0.7)
    if f.get('money_words_count', 0) > 0: base += min(2, f['money_words_count'] * 0.5)
    if f.get('urgent_words_count', 0) > 0: base += 1.5
    if f.get('suspicious_url_count', 0) > 0: base += 2
    if f.get('sender_suspicious_domain', False): base += 1.5
    return max(0.0, min(10.0, round(base, 1)))

if __name__ == '__main__':
    logger.info("🚀 Starting Hybrid Scam Detector API...")
    path = os.path.join("data", "hybrid_model.pkl")
    if os.path.exists(path):
        ml_model.load_model(path)
        logger.info("✅ Hybrid model loaded.")
    else:
        logger.warning("⚠️ Model not found, please train using train_model.py.")
    app.run(host='localhost', port=5000, debug=True)