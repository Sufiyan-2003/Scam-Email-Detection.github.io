import pickle
import os
import numpy as np
import pandas as pd
import logging
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_extractor import EmailFeatureExtractor

try:
    from minisom import MiniSom
except Exception:
    MiniSom = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailClassifier:
    """
    🌟 Dual-Stage Hybrid Email Scam Detector (K-Means + SOM + Random Forest)
    -----------------------------------------------------------------------
    1️⃣ K-Means learns global spam clusters
    2️⃣ SOM refines local, non-linear relationships
    3️⃣ Random Forest learns final high-accuracy decision boundary
    """

    def __init__(self, n_clusters=3, som_x=12, som_y=12,
                 som_sigma=0.9, som_learning_rate=0.4):
        self.scaler = StandardScaler()
        self.feature_extractor = EmailFeatureExtractor()
        self.feature_columns = None
        self.is_trained = False

        self.n_clusters = n_clusters
        self.som_x, self.som_y = som_x, som_y
        self.som_sigma, self.som_learning_rate = som_sigma, som_learning_rate

        self.kmeans_model = None
        self.som_model = None
        self.rf_model = None

    # ------------------ Data loading ------------------
    def preprocess_data(self, data_path):
        logger.info("📂 Loading and preprocessing data...")
        dataset_files = {
            'CEAS_08.csv': 0,
            'Enron.csv': 0,
            'Ling.csv': 0,
            'phishing_email.csv': 1,
            'SpamAssain.csv': 1,
            'Nigerian_Fraud.csv': 1,
            'Nazario.csv': 1
        }

        emails, labels = [], []
        for fname, label in dataset_files.items():
            fpath = os.path.join(data_path, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, encoding='utf-8', low_memory=False)
                processed = self._process_csv(df)
                emails.extend(processed)
                labels.extend([label] * len(processed))
                logger.info(f"✅ Loaded {len(processed)} from {fname}")
            else:
                logger.warning(f"⚠️ Missing file: {fpath}")
        logger.info(f"📊 Total Emails: {len(emails)} | Legit: {labels.count(0)} | Spam: {labels.count(1)}")
        return emails, labels

    def _process_csv(self, df):
        processed = []
        mapping = {
            'subject': ['Subject', 'subject', 'Subject Line'],
            'body': ['Body', 'body', 'Message', 'Content', 'Text'],
            'sender': ['From', 'from', 'Sender'],
            'reply_to': ['Reply-To', 'reply_to']
        }
        cols = {}
        for key, possible in mapping.items():
            for c in possible:
                if c in df.columns:
                    cols[key] = c
                    break
        for _, r in df.iterrows():
            email = {
                'subject': str(r.get(cols.get('subject', ''), '')),
                'body': str(r.get(cols.get('body', ''), '')),
                'sender': str(r.get(cols.get('sender', ''), '')),
                'reply_to': str(r.get(cols.get('reply_to', ''), '')),
            }
            if email['body'].strip():
                processed.append(email)
        return processed

    # ------------------ Training ------------------
    def train(self, data_path, test_size=0.2, random_state=42):
        logger.info("🚀 Starting Dual-Stage Training (KMeans + SOM + RF)...")
        data, labels = self.preprocess_data(data_path)
        if not data:
            raise ValueError("No data found.")

        logger.info("🔍 Extracting features...")
        features_df = self.feature_extractor.prepare_features_dataframe(data)
        self.feature_columns = features_df.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=test_size, stratify=labels, random_state=random_state)

        logger.info("⚙️ Scaling & Normalizing...")
        X_train_s = normalize(self.scaler.fit_transform(X_train))
        X_test_s = normalize(self.scaler.transform(X_test))

        # ---------- Stage 1: KMeans ----------
        logger.info("🌀 Stage 1: Training K-Means Clusters...")
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=random_state, n_init=20)
        km_train = self.kmeans_model.fit_predict(X_train_s)
        km_test = self.kmeans_model.predict(X_test_s)

        # ---------- Stage 2: SOM ----------
        logger.info("🧠 Stage 2: Training Self-Organizing Map...")
        if MiniSom is None:
            raise ImportError("Please install minisom (pip install minisom)")
        som = MiniSom(self.som_x, self.som_y, X_train_s.shape[1],
                      sigma=self.som_sigma, learning_rate=self.som_learning_rate, random_seed=random_state)
        som.random_weights_init(X_train_s)
        som.train_random(X_train_s, 1800)  # longer training
        som_train = np.array([hash(som.winner(x)) % self.n_clusters for x in X_train_s])
        som_test = np.array([hash(som.winner(x)) % self.n_clusters for x in X_test_s])
        self.som_model = som

        # ---------- Stage 3: Random Forest ----------
        logger.info("🌲 Stage 3: Merging features + training Random Forest...")
        X_train_m = np.hstack([X_train_s, km_train.reshape(-1, 1), som_train.reshape(-1, 1)])
        X_test_m = np.hstack([X_test_s, km_test.reshape(-1, 1), som_test.reshape(-1, 1)])

        self.rf_model = RandomForestClassifier(
            n_estimators=400, max_depth=28, min_samples_split=2, min_samples_leaf=1,
            random_state=random_state, n_jobs=-1, class_weight='balanced_subsample', bootstrap=True)
        self.rf_model.fit(X_train_m, y_train)

        preds = self.rf_model.predict(X_test_m)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        logger.info("\n📊 FINAL MODEL PERFORMANCE")
        logger.info(f"Accuracy  : {acc*100:.2f}%")
        logger.info(f"Precision : {prec*100:.2f}%")
        logger.info(f"Recall    : {rec*100:.2f}%")
        logger.info(f"F1-Score  : {f1*100:.2f}%")

        self.is_trained = True
        return dict(test_accuracy=acc, precision=prec, recall=rec, f1_score=f1)

    # ------------------ Prediction ------------------
    def _get_clusters(self, X_s):
        km = self.kmeans_model.predict(X_s)
        som_idx = np.array([hash(self.som_model.winner(x)) % self.n_clusters for x in X_s])
        return km, som_idx

    def predict(self, email):
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        f = self.feature_extractor.extract_all_features(email)
        txt = email.get('subject', '') + ' ' + email.get('body', '')
        tfidf = self.feature_extractor.get_tfidf_features(txt)
        vec = [tfidf[int(c.split('_')[1])] if c.startswith('tfidf_') and int(c.split('_')[1]) < len(tfidf)
               else f.get(c, 0) for c in self.feature_columns]
        X_s = normalize(self.scaler.transform([vec]))
        km, som = self._get_clusters(X_s)
        X_m = np.hstack([X_s, np.array([[km[0]]]), np.array([[som[0]]])])
        prob = self.rf_model.predict_proba(X_m)[0]
        label = int(np.argmax(prob))
        return {
            'prediction': 'spam' if label == 1 else 'legitimate',
            'confidence': float(max(prob)),
            'spam_probability': float(prob[1]),
            'legitimate_probability': float(prob[0])
        }

    # ------------------ Save / Load ------------------
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_extractor': self.feature_extractor,
                'feature_columns': self.feature_columns,
                'kmeans_model': self.kmeans_model,
                'som_model': self.som_model,
                'rf_model': self.rf_model,
                'is_trained': self.is_trained,
                'n_clusters': self.n_clusters
            }, f)
        logger.info(f"✅ Model saved to {path}")

    def load_model(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.scaler = d['scaler']
        self.feature_extractor = d['feature_extractor']
        self.feature_columns = d['feature_columns']
        self.kmeans_model = d['kmeans_model']
        self.som_model = d['som_model']
        self.rf_model = d['rf_model']
        self.is_trained = d['is_trained']
        self.n_clusters = d.get('n_clusters', 3)
        logger.info("✅ Dual-Stage model loaded successfully.")

    # ------------------ Enhanced Explanation ------------------
    def get_prediction_explanation(self, email, features):
        explanation = {
            "main_reasons": [],
            "suspicious_keywords": [],
            "suspicious_urls": [],
            "sender_issues": [],
            "structural_issues": []
        }
        
        # Extract text content
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        full_text = subject + " " + body
        
        # Check for suspicious keywords
        scam_keywords = [
            'urgent', 'immediately', 'verify', 'account', 'suspended', 'security',
            'password', 'login', 'click here', 'limited time', 'offer', 'free',
            'winner', 'prize', 'lottery', 'inheritance', 'million', 'billion',
            'dollar', 'euro', 'money', 'fund', 'transfer', 'bank', 'paypal',
            'western union', 'moneygram', 'bitcoin', 'crypto', 'investment',
            'risk-free', 'guaranteed', 'profit', 'opportunity', 'secret',
            'confidential', 'private', 'personal use', 'certificate', 'deposit'
        ]
        
        found_keywords = []
        for keyword in scam_keywords:
            if keyword in full_text:
                found_keywords.append(keyword)
        
        if found_keywords:
            explanation["suspicious_keywords"] = found_keywords[:10]  # Limit to 10
            explanation["main_reasons"].append(f"Suspicious keywords detected: {', '.join(found_keywords[:5])}")
        
        # Check for suspicious URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, full_text)
        
        suspicious_urls = []
        for url in urls:
            if any(suspicious in url.lower() for suspicious in ['bit.ly', 'tinyurl', 'shortener', 'redirect']):
                suspicious_urls.append(url)
        
        if suspicious_urls:
            explanation["suspicious_urls"] = suspicious_urls
            explanation["main_reasons"].append(f"Suspicious URLs detected: {len(suspicious_urls)}")
        
        # Check sender issues
        sender = email.get('sender', '').lower()
        if any(suspicious in sender for suspicious in ['paypal', 'bank', 'security', 'verify']):
            explanation["sender_issues"] = ["Suspicious sender name mimicking legitimate service"]
            explanation["main_reasons"].append("Sender appears to be impersonating a legitimate service")
        
        # Check structural issues
        structural_issues = []
        
        # Check for excessive capitalization
        if features.get('caps_ratio', 0) > 0.3:
            structural_issues.append("High percentage of capital letters")
            explanation["main_reasons"].append("Excessive use of capital letters")
        
        # Check for urgency indicators
        if features.get('urgent_words_count', 0) > 0:
            structural_issues.append("Urgent language detected")
        
        # Check for money-related content
        if features.get('money_words_count', 0) > 0:
            structural_issues.append("Money-related terminology")
        
        # Check for scam phrases
        if features.get('scam_phrases_count', 0) > 0:
            structural_issues.append("Common scam phrases identified")
        
        # Check for excessive punctuation
        if features.get('exclamation_count', 0) > 3:
            structural_issues.append("Excessive exclamation marks")
            explanation["main_reasons"].append("Overuse of exclamation marks")
        
        if structural_issues:
            explanation["structural_issues"] = structural_issues
        
        # If no specific issues found but email is suspicious, provide generic explanation
        if not explanation["main_reasons"] and features:
            explanation["main_reasons"] = [
                "Pattern analysis indicates suspicious characteristics",
                "Content structure matches known scam patterns",
                "Linguistic analysis reveals deceptive intent"
            ]
        
        return explanation