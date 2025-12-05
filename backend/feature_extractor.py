import re
import string
from urllib.parse import urlparse
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease


class EmailFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )

        # Suspicious keywords and patterns
        self.urgent_words = [
            'urgent', 'immediate', 'asap', 'emergency', 'quickly', 'hurry',
            'act now', 'limited time', 'expire', 'deadline', 'final notice'
        ]

        self.money_words = [
            'money', 'cash', 'prize', 'winner', 'lottery', 'million',
            'thousand', 'dollars', 'pounds', 'euros', 'reward', 'inherit',
            'beneficiary', 'fortune', 'jackpot', 'refund', 'compensation'
        ]

        self.scam_phrases = [
            'congratulations', 'you have won', 'claim your prize',
            'verify your account', 'update your information',
            'suspended account', 'click here', 'dear friend',
            'business proposal', 'confidence', 'transaction',
            'transfer funds', 'bank account', 'personal information'
        ]

        self.suspicious_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'aol.com', 'mail.com'  # When used for business emails
        ]

    def extract_basic_features(self, email_data):
        features = {}
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        sender = email_data.get('sender', '')

        features['subject_length'] = len(subject)
        features['body_length'] = len(body)
        features['total_length'] = len(subject) + len(body)

        features['caps_ratio'] = self._calculate_caps_ratio(subject + ' ' + body)
        features['exclamation_count'] = (subject + body).count('!')
        features['question_count'] = (subject + body).count('?')
        features['dollar_count'] = (subject + body).count('$')

        full_text = subject + ' ' + body
        features['punctuation_ratio'] = sum(1 for c in full_text if c in string.punctuation) / max(len(full_text), 1)

        return features

    def extract_content_features(self, email_data):
        features = {}
        subject = email_data.get('subject', '').lower()
        body = email_data.get('body', '').lower()
        full_text = subject + ' ' + body

        features['urgent_words_count'] = sum(1 for word in self.urgent_words if word in full_text)
        features['money_words_count'] = sum(1 for word in self.money_words if word in full_text)
        features['scam_phrases_count'] = sum(1 for phrase in self.scam_phrases if phrase in full_text)

        features['readability_score'] = flesch_reading_ease(full_text) if len(full_text.strip()) > 0 else 0

        features['number_count'] = len(re.findall(r'\d+', full_text))
        features['phone_pattern'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', full_text))

        return features

    def extract_url_features(self, email_data):
        features = {}
        body = email_data.get('body', '')

        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, body)

        features['url_count'] = len(urls)
        features['has_urls'] = len(urls) > 0

        suspicious_url_count, ip_url_count, short_url_count = 0, 0, 0

        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()

                if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
                    ip_url_count += 1

                shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'short.ly']
                if any(shortener in domain for shortener in shorteners):
                    short_url_count += 1

                suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
                if any(domain.endswith(tld) for tld in suspicious_tlds):
                    suspicious_url_count += 1

            except Exception:
                suspicious_url_count += 1

        features['suspicious_url_count'] = suspicious_url_count
        features['ip_url_count'] = ip_url_count
        features['short_url_count'] = short_url_count

        return features

    def extract_sender_features(self, email_data):
        features = {}
        sender = email_data.get('sender', '').lower()
        reply_to = email_data.get('reply_to', '').lower()

        features['sender_suspicious_domain'] = any(domain in sender for domain in self.suspicious_domains)
        features['sender_has_numbers'] = bool(re.search(r'\d', sender))
        features['sender_mismatch'] = sender != reply_to and reply_to != ''

        if '@' in sender:
            sender_domain = sender.split('@')[1] if '@' in sender else ''
            features['sender_domain_length'] = len(sender_domain)
            features['sender_subdomain_count'] = sender_domain.count('.') - 1
        else:
            features['sender_domain_length'] = 0
            features['sender_subdomain_count'] = 0

        return features

    def extract_all_features(self, email_data):
        features = {}
        features.update(self.extract_basic_features(email_data))
        features.update(self.extract_content_features(email_data))
        features.update(self.extract_url_features(email_data))
        features.update(self.extract_sender_features(email_data))
        return features

    def fit_tfidf(self, texts):
        self.tfidf_vectorizer.fit(texts)

    def get_tfidf_features(self, text):
        if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            return self.tfidf_vectorizer.transform([text]).toarray()[0]
        return np.zeros(1000)

    def _calculate_caps_ratio(self, text):
        if len(text) == 0:
            return 0
        caps_count = sum(1 for c in text if c.isupper())
        return caps_count / len(text)

    def prepare_features_dataframe(self, email_data_list):
        feature_list, texts = [], []

        for email_data in email_data_list:
            features = self.extract_all_features(email_data)
            feature_list.append(features)

            subject = email_data.get('subject', '')
            body = email_data.get('body', '')
            full_text = subject + ' ' + body
            texts.append(full_text)

        features_df = pd.DataFrame(feature_list)

        if len(texts) > 0:
            self.fit_tfidf(texts)
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()

            tfidf_columns = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_columns)

            features_df = pd.concat([features_df.reset_index(drop=True),
                                     tfidf_df.reset_index(drop=True)], axis=1)

        features_df = features_df.fillna(0)
        return features_df
