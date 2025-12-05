#!/usr/bin/env python3
"""
Data preprocessing utility for the Email Scam Detector
This script helps preprocess and analyze the datasets before training
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import logging
from email_analyzer import EmailAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.email_analyzer = EmailAnalyzer()
        
    def explore_datasets(self):
        """Explore and analyze all datasets in the directory"""
        logger.info("Exploring datasets...")
        
        dataset_info = {}
        
        for filename in os.listdir(self.data_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_path, filename)
                logger.info(f"\nAnalyzing {filename}...")
                
                try:
                    # Load dataset
                    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                    
                    info = {
                        'filename': filename,
                        'rows': len(df),
                        'columns': list(df.columns),
                        'column_count': len(df.columns),
                        'missing_values': df.isnull().sum().to_dict(),
                        'sample_data': df.head(2).to_dict('records')
                    }
                    
                    dataset_info[filename] = info
                    
                    logger.info(f"  Rows: {info['rows']}")
                    logger.info(f"  Columns: {info['columns']}")
                    logger.info(f"  Missing values: {sum(info['missing_values'].values())}")
                    
                except Exception as e:
                    logger.error(f"Error reading {filename}: {str(e)}")
                    continue
        
        return dataset_info
    
    def analyze_text_content(self, sample_size=1000):
        """Analyze text content across all datasets"""
        logger.info("Analyzing text content...")
        
        all_subjects = []
        all_bodies = []
        dataset_stats = {}
        
        # Dataset classifications
        legitimate_files = ['CEAS_08.csv', 'Enron.csv', 'Ling.csv']
        spam_files = ['phishing_email.csv', 'SpamAssain.csv', 'Nigerian_Fraud.csv', 'Nazario.csv']
        
        for filename in os.listdir(self.data_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_path, filename)
                logger.info(f"Processing {filename}...")
                
                try:
                    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                    
                    # Limit sample size for large datasets
                    if len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42)
                    
                    # Find text columns
                    text_columns = self._identify_text_columns(df)
                    
                    subjects = []
                    bodies = []
                    
                    for col in text_columns['subject']:
                        if col in df.columns:
                            subjects.extend(df[col].dropna().astype(str).tolist())
                    
                    for col in text_columns['body']:
                        if col in df.columns:
                            bodies.extend(df[col].dropna().astype(str).tolist())
                    
                    # If no specific columns found, use first text column as body
                    if not subjects and not bodies:
                        text_cols = df.select_dtypes(include=['object']).columns
                        if len(text_cols) > 0:
                            bodies.extend(df[text_cols[0]].dropna().astype(str).tolist())
                    
                    # Calculate statistics
                    avg_subject_len = np.mean([len(s) for s in subjects]) if subjects else 0
                    avg_body_len = np.mean([len(b) for b in bodies]) if bodies else 0
                    
                    dataset_stats[filename] = {
                        'subjects_count': len(subjects),
                        'bodies_count': len(bodies),
                        'avg_subject_length': avg_subject_len,
                        'avg_body_length': avg_body_len,
                        'is_spam': filename in spam_files
                    }
                    
                    all_subjects.extend(subjects)
                    all_bodies.extend(bodies)
                    
                    logger.info(f"  Subjects: {len(subjects)}, Bodies: {len(bodies)}")
                    logger.info(f"  Avg subject length: {avg_subject_len:.1f}")
                    logger.info(f"  Avg body length: {avg_body_len:.1f}")
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    continue
        
        # Analyze common words and patterns
        self._analyze_common_patterns(all_subjects, all_bodies, dataset_stats)
        
        return dataset_stats
    
    def _identify_text_columns(self, df):
        """Identify which columns likely contain subject and body text"""
        column_mappings = {
            'subject': ['Subject', 'subject', 'Subject Line', 'Subject_Line', 'SUBJECT'],
            'body': ['Body', 'body', 'Message', 'message', 'Content', 'content', 
                    'Text', 'text', 'Email', 'email', 'BODY', 'MESSAGE']
        }
        
        found_columns = {'subject': [], 'body': []}
        
        for field, possible_names in column_mappings.items():
            for col_name in possible_names:
                if col_name in df.columns:
                    found_columns[field].append(col_name)
        
        return found_columns
    
    def _analyze_common_patterns(self, subjects, bodies, dataset_stats):
        """Analyze common patterns in email content"""
        logger.info("Analyzing common patterns...")
        
        all_text = subjects + bodies
        
        # Common suspicious words
        suspicious_words = [
            'urgent', 'winner', 'congratulations', 'million', 'lottery',
            'click', 'verify', 'account', 'suspended', 'update',
            'prize', 'cash', 'money', 'inheritance', 'beneficiary'
        ]
        
        # Count suspicious word occurrences
        word_counts = Counter()
        for text in all_text:
            text_lower = text.lower()
            for word in suspicious_words:
                if word in text_lower:
                    word_counts[word] += 1
        
        logger.info("Top suspicious words found:")
        for word, count in word_counts.most_common(10):
            logger.info(f"  {word}: {count}")
        
        # URL analysis
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        total_urls = 0
        for text in all_text:
            urls = re.findall(url_pattern, text)
            total_urls += len(urls)
        
        logger.info(f"Total URLs found: {total_urls}")
        
        # Email address analysis
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        total_emails = 0
        for text in all_text:
            emails = re.findall(email_pattern, text)
            total_emails += len(emails)
        
        logger.info(f"Total email addresses found: {total_emails}")
    
    def create_balanced_dataset(self, output_path, max_samples_per_class=5000):
        """Create a balanced dataset for training"""
        logger.info("Creating balanced dataset...")
        
        legitimate_data = []
        spam_data = []
        
        # Dataset classifications
        legitimate_files = ['CEAS_08.csv', 'Enron.csv', 'Ling.csv']
        spam_files = ['phishing_email.csv', 'SpamAssain.csv', 'Nigerian_Fraud.csv', 'Nazario.csv']
        
        # Process legitimate emails
        for filename in legitimate_files:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                data = self._extract_email_data(filepath, is_spam=False)
                legitimate_data.extend(data)
        
        # Process spam emails
        for filename in spam_files:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                data = self._extract_email_data(filepath, is_spam=True)
                spam_data.extend(data)
        
        # Balance the dataset
        min_samples = min(len(legitimate_data), len(spam_data), max_samples_per_class)
        
        if len(legitimate_data) > min_samples:
            legitimate_data = np.random.choice(legitimate_data, min_samples, replace=False).tolist()
        
        if len(spam_data) > min_samples:
            spam_data = np.random.choice(spam_data, min_samples, replace=False).tolist()
        
        # Combine and shuffle
        combined_data = legitimate_data + spam_data
        np.random.shuffle(combined_data)
        
        # Save to CSV
        df = pd.DataFrame(combined_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Balanced dataset saved to {output_path}")
        logger.info(f"Legitimate emails: {len(legitimate_data)}")
        logger.info(f"Spam emails: {len(spam_data)}")
        logger.info(f"Total samples: {len(combined_data)}")
        
        return output_path
    
    def _extract_email_data(self, filepath, is_spam=False):
        """Extract standardized email data from a CSV file"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            
            # Identify columns
            text_columns = self._identify_text_columns(df)
            
            extracted_data = []
            
            for idx, row in df.iterrows():
                try:
                    # Extract subject
                    subject = ''
                    for col in text_columns['subject']:
                        if col in df.columns and pd.notna(row[col]):
                            subject = str(row[col])
                            break
                    
                    # Extract body
                    body = ''
                    for col in text_columns['body']:
                        if col in df.columns and pd.notna(row[col]):
                            body = str(row[col])
                            break
                    
                    # If no specific columns found, use first text column
                    if not subject and not body:
                        text_cols = df.select_dtypes(include=['object']).columns
                        if len(text_cols) > 0 and pd.notna(row[text_cols[0]]):
                            body = str(row[text_cols[0]])
                    
                    # Skip if both are empty
                    if not subject and not body:
                        continue
                    
                    # Extract sender information
                    sender = ''
                    sender_cols = ['From', 'from', 'Sender', 'sender']
                    for col in sender_cols:
                        if col in df.columns and pd.notna(row[col]):
                            sender = str(row[col])
                            break
                    
                    email_data = {
                        'subject': subject,
                        'body': body,
                        'sender': sender,
                        'reply_to': '',
                        'is_spam': 1 if is_spam else 0
                    }
                    
                    extracted_data.append(email_data)
                    
                except Exception as e:
                    continue
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting data from {filepath}: {str(e)}")
            return []
    
    def generate_report(self, output_file='dataset_analysis_report.txt'):
        """Generate a comprehensive analysis report"""
        logger.info("Generating analysis report...")
        
        dataset_info = self.explore_datasets()
        text_stats = self.analyze_text_content()
        
        with open(output_file, 'w') as f:
            f.write("EMAIL DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            for filename, info in dataset_info.items():
                f.write(f"\nDataset: {filename}\n")
                f.write(f"  Rows: {info['rows']}\n")
                f.write(f"  Columns: {info['column_count']}\n")
                f.write(f"  Column names: {', '.join(info['columns'])}\n")
                f.write(f"  Missing values: {sum(info['missing_values'].values())}\n")
            
            # Text statistics
            f.write("\n\nTEXT CONTENT STATISTICS\n")
            f.write("-" * 25 + "\n")
            for filename, stats in text_stats.items():
                f.write(f"\nDataset: {filename}\n")
                f.write(f"  Type: {'Spam/Phishing' if stats['is_spam'] else 'Legitimate'}\n")
                f.write(f"  Subjects: {stats['subjects_count']}\n")
                f.write(f"  Bodies: {stats['bodies_count']}\n")
                f.write(f"  Avg subject length: {stats['avg_subject_length']:.1f}\n")
                f.write(f"  Avg body length: {stats['avg_body_length']:.1f}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDANIONS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Use balanced sampling to avoid class imbalance\n")
            f.write("2. Apply text preprocessing (lowercase, remove HTML, etc.)\n")
            f.write("3. Extract features like URL count, suspicious words, etc.\n")
            f.write("4. Consider using TF-IDF for text vectorization\n")
            f.write("5. Implement cross-validation for robust evaluation\n")
        
        logger.info(f"Report saved to {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Email Datasets')
    parser.add_argument(
        '--data_path',
        type=str,
        default=r'D:\VS-Studio\email\spamdataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--create_balanced',
        action='store_true',
        help='Create balanced dataset'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='balanced_dataset.csv',
        help='Output path for balanced dataset'
    )
    parser.add_argument(
        '--generate_report',
        action='store_true',
        help='Generate analysis report'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        return
    
    preprocessor = DataPreprocessor(args.data_path)
    
    if args.generate_report:
        preprocessor.generate_report()
    
    if args.create_balanced:
        preprocessor.create_balanced_dataset(args.output_path)
    
    if not args.generate_report and not args.create_balanced:
        # Default: explore datasets
        dataset_info = preprocessor.explore_datasets()
        text_stats = preprocessor.analyze_text_content()

if __name__ == '__main__':
    main()