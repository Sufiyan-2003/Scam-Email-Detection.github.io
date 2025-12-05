import re
import email
from email import policy
from email.parser import BytesParser
import urllib.parse

class EmailAnalyzer:
    def __init__(self):
        self.scam_phrases = [
            "verify your account", "account suspended", "click here", "limited time",
            "risk free", "guaranteed profit", "urgent action required", "immediately",
            "personal use", "certificate of deposit", "no risk involved", "funds transfer"
        ]
        
        self.money_words = [
            "money", "fund", "transfer", "bank", "paypal", "western union", "moneygram",
            "bitcoin", "crypto", "investment", "profit", "million", "billion", "dollar",
            "euro", "inheritance", "prize", "lottery", "winner"
        ]
        
        self.urgent_words = [
            "urgent", "immediately", "asap", "right away", "instant", "quick",
            "emergency", "important", "action required", "verify now"
        ]

    def parse_email(self, content):
        try:
            if isinstance(content, bytes):
                msg = BytesParser(policy=policy.default).parsebytes(content)
            else:
                msg = BytesParser(policy=policy.default).parsestr(content)
            
            parsed = {
                'subject': msg.get('subject', ''),
                'body': self._extract_body(msg),
                'sender': msg.get('from', ''),
                'reply_to': msg.get('reply-to', ''),
                'to': msg.get('to', ''),
                'date': msg.get('date', '')
            }
            return parsed
        except:
            # Fallback for plain text
            lines = content.split('\n')
            parsed = {
                'subject': '',
                'body': content,
                'sender': '',
                'reply_to': '',
                'to': '',
                'date': ''
            }
            
            for i, line in enumerate(lines):
                lower_line = line.lower()
                if lower_line.startswith('subject:'):
                    parsed['subject'] = line[8:].strip()
                elif lower_line.startswith('from:'):
                    parsed['sender'] = line[5:].strip()
                elif lower_line.startswith('to:'):
                    parsed['to'] = line[3:].strip()
            
            return parsed

    def _extract_body(self, msg):
        if msg.is_multipart():
            for part in msg.iter_parts():
                if part.get_content_type() == 'text/plain':
                    return part.get_content()
        return msg.get_content()

    def extract_features(self, parsed_email):
        features = {}
        
        subject = parsed_email.get('subject', '').lower()
        body = parsed_email.get('body', '').lower()
        full_text = subject + " " + body
        
        # Basic text features
        features['text_length'] = len(full_text)
        features['word_count'] = len(full_text.split())
        
        # Suspicious phrases
        features['scam_phrases_count'] = sum(1 for phrase in self.scam_phrases if phrase in full_text)
        
        # Money-related words
        features['money_words_count'] = sum(1 for word in self.money_words if word in full_text)
        
        # Urgent language
        features['urgent_words_count'] = sum(1 for word in self.urgent_words if word in full_text)
        
        # URL analysis
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, full_text)
        features['url_count'] = len(urls)
        features['suspicious_url_count'] = sum(1 for url in urls if any(suspicious in url.lower() for suspicious in ['bit.ly', 'tinyurl']))
        
        # Capitalization analysis
        caps_words = re.findall(r'\b[A-Z]{3,}\b', full_text)
        features['caps_word_count'] = len(caps_words)
        features['caps_ratio'] = len(caps_words) / max(1, len(re.findall(r'\b\w+\b', full_text)))
        
        # Punctuation
        features['exclamation_count'] = full_text.count('!')
        features['question_count'] = full_text.count('?')
        
        # Sender analysis
        sender = parsed_email.get('sender', '').lower()
        features['sender_suspicious_domain'] = any(domain in sender for domain in ['paypal', 'bank', 'security'])
        
        return features

    def extract_all_features(self, parsed_email):
        return self.extract_features(parsed_email)