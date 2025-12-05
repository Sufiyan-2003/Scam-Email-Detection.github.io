# Email Scam Detector

A machine learning-powered web application designed to identify and flag potentially fraudulent, phishing, or spam emails using natural language processing (NLP) and email structure analysis.

## Features

- **Email Classification**: Classify emails as legitimate or spam/phishing using machine learning
- **Feature Extraction**: Extract comprehensive features from email content, structure, and metadata
- **Web Interface**: User-friendly web interface for email analysis
- **REST API**: Backend API for programmatic access
- **Explainable AI**: Provides explanations for predictions to build user trust
- **Multiple Datasets**: Trained on various spam and phishing email datasets

## Project Structure

```
scam-email-detector/
├── backend/
│   ├── app.py                 # Flask application (main API server)
│   ├── email_analyzer.py      # Email parsing and analysis utilities
│   ├── ml_model.py           # Machine learning model implementation
│   ├── feature_extractor.py  # Feature engineering from emails
│   ├── train_model.py        # Model training script
│   ├── data_preprocessor.py  # Data preprocessing utilities
│   ├── setup.py              # Environment setup script
│   ├── requirements.txt      # Python dependencies
│   └── data/
│       └── model.pkl         # Trained model (generated after training)
├── frontend/
│   ├── index.html            # Web interface
└── README.md                 # Project documentation
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Quick Setup

1. **Clone the repository** (if applicable) or ensure all files are in the correct structure

2. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

3. **Run the setup script**:
   ```bash
   python setup.py
   ```
   
   This will:
   - Install all required dependencies
   - Download necessary NLTK data
   - Create required directories
   - Analyze your datasets
   - Train the initial model (if datasets are available)

### Manual Setup (Alternative)

If the automatic setup doesn't work, follow these manual steps:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

3. **Update data path** in the training scripts to point to your dataset location

4. **Train the model**:
   ```bash
   python train_model.py --data_path "path/to/your/spamdataset"
   ```

## Dataset Configuration

The system is designed to work with the following datasets (place them in your `spamdataset` folder):

- **Legitimate Emails**:
  - `CEAS_08.csv` - CEAS 2008 dataset
  - `Enron.csv` - Enron email corpus
  - `Ling.csv` - Ling spam dataset (ham emails)

- **Spam/Phishing Emails**:
  - `phishing_email.csv` - Phishing emails dataset
  - `SpamAssain.csv` - SpamAssassin corpus
  - `Nigerian_Fraud.csv` - Nigerian fraud emails
  - `Nazario.csv` - Nazario phishing dataset

### Updating Dataset Path

To use your datasets, update the `data_path` in:
- `train_model.py` (line with `default=r'D:\VS-Studio\email\spamdataset'`)
- `data_preprocessor.py` (same location)
- Or use command line arguments when running scripts

## Usage

### Training the Model

#### Basic Training
```bash
python train_model.py
```

#### Advanced Training Options
```bash
# Train with specific model type
python train_model.py --model_type gradient_boosting

# Train with hyperparameter tuning
python train_model.py --tune_hyperparameters

# Train with custom data path
python train_model.py --data_path "/path/to/your/datasets"

# Train with custom test size
python train_model.py --test_size 0.3
```

### Running the Web Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Use the interface** to:
   - Paste email content for analysis
   - Upload email files
   - View prediction results and explanations

### Data Analysis

Analyze your datasets before training:

```bash
# Generate comprehensive analysis report
python data_preprocessor.py --generate_report

# Create balanced dataset
python data_preprocessor.py --create_balanced --output_path balanced_emails.csv

# Explore datasets
python data_preprocessor.py
```

## API Usage

The Flask application provides REST API endpoints:

### Analyze Email Text
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Congratulations! You won $1,000,000!",
    "body": "Click here to claim your prize...",
    "sender": "winner@lottery.com"
  }'
```

### Upload Email File
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "email=@sample_email.eml"
```

## Model Features

The system extracts various features from emails:

### Structural Features
- Email length (subject, body, total)
- Character analysis (caps ratio, punctuation)
- Special character counts (!, ?, $)

### Content Features
- Suspicious word detection
- Money-related terms
- Urgent language patterns
- Readability scores
- Number and phone patterns

### URL Features
- URL count and analysis
- Suspicious domain detection
- IP address URLs
- URL shortener detection

### Sender Features
- Domain analysis
- Sender-reply mismatch detection
- Suspicious email patterns

### Text Features
- TF-IDF vectorization
- N-gram analysis
- Stop word filtering

## Model Performance

The system uses ensemble methods (Random Forest, Gradient Boosting) and achieves:
- High accuracy on balanced datasets
- Low false positive rates
- Explainable predictions
- Cross-validation for robust evaluation

## Customization

### Adding New Features

1. Modify `feature_extractor.py` to add new feature extraction methods
2. Update the `extract_all_features()` method to include your features
3. Retrain the model with new features

### Using Different Models

The system supports:
- Random Forest (default)
- Gradient Boosting
- Easy extension to other scikit-learn models

### Custom Datasets

To use your own datasets:
1. Ensure CSV format with text columns (subject, body, message, etc.)
2. Update the dataset file mappings in `ml_model.py`
3. Modify the `_process_csv()` method if needed for your data format
4. Retrain the model

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure all requirements are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK Data Error**: Download required NLTK data
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

3. **File Path Issues**: Update data paths in configuration files or use absolute paths

4. **Memory Issues**: For large datasets, reduce sample size in `data_preprocessor.py`

5. **CSV Reading Errors**: Check file encoding, try different encodings (utf-8, latin-1, cp1252)

### Performance Optimization

1. **Large Datasets**: Use sampling for training to reduce memory usage
2. **Feature Selection**: Remove less important features to speed up training
3. **Model Optimization**: Use hyperparameter tuning for better performance

## Development

### File Descriptions

- **`app.py`**: Flask web server with REST API endpoints
- **`email_analyzer.py`**: Email parsing utilities (handles .eml files, headers, etc.)
- **`ml_model.py`**: Core ML functionality (training, prediction, model management)
- **`feature_extractor.py`**: Feature engineering pipeline
- **`train_model.py`**: Command-line training script
- **`data_preprocessor.py`**: Dataset analysis and preprocessing utilities
- **`setup.py`**: Automated environment setup

### Adding New Endpoints

To add new API endpoints, modify `app.py`:

```python
@app.route('/api/new_endpoint', methods=['POST'])
def new_endpoint():
    # Your logic here
    return jsonify({'result': 'success'})
```

### Extending Features

1. Add new feature extraction methods to `EmailFeatureExtractor` class
2. Update the `extract_all_features()` method
3. Retrain the model to include new features

## Security Considerations

- Input validation on all user inputs
- File upload restrictions
- No execution of email attachments
- Sanitization of email content before processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Future Enhancements

- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Real-time email monitoring
- [ ] Advanced explainability (LIME, SHAP)
- [ ] Multi-language support
- [ ] Image analysis for email attachments
- [ ] Integration with email clients
- [ ] User feedback learning
- [ ] Advanced phishing detection techniques

## Contact

For questions or issues, please check the troubleshooting section above or review the code comments for implementation details.

---

**Note**: This system is designed for educational and research purposes. Always use multiple validation methods for production email security systems.