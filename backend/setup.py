#!/usr/bin/env python3
"""
Setup script for the Email Scam Detector
This script sets up the environment and trains the initial model
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    logger.info("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False
    return True

def setup_nltk():
    """Download required NLTK data"""
    logger.info("Setting up NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        logger.info("NLTK data downloaded successfully!")
    except Exception as e:
        logger.error(f"Failed to setup NLTK: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    logger.info("Creating project directories...")
    
    directories = [
        'data',
        'logs',
        'models',
        'temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def verify_data_path():
    """Verify that the data path exists"""
    data_path = r'D:\VS-Studio\email\spamdataset'
    
    if os.path.exists(data_path):
        logger.info(f"Data path verified: {data_path}")
        
        # List available datasets
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            logger.info(f"  - {file}")
        
        return True
    else:
        logger.warning(f"Data path not found: {data_path}")
        logger.info("Please update the data path in the configuration files")
        return False

def train_initial_model():
    """Train the initial model"""
    logger.info("Training initial model...")
    
    try:
        # Import after requirements are installed
        from train_model import main as train_main
        
        # Set up arguments for training
        sys.argv = [
            'train_model.py',
            '--data_path', r'D:\VS-Studio\email\spamdataset',
            '--model_type', 'random_forest',
            '--output_path', 'data/model.pkl'
        ]
        
        train_main()
        logger.info("Initial model trained successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to train initial model: {e}")
        logger.info("You can train the model manually later using: python train_model.py")
        return False

def run_data_analysis():
    """Run data analysis"""
    logger.info("Running data analysis...")
    
    try:
        from data_preprocessor import DataPreprocessor
        
        data_path = r'D:\VS-Studio\email\spamdataset'
        if os.path.exists(data_path):
            preprocessor = DataPreprocessor(data_path)
            preprocessor.generate_report()
            logger.info("Data analysis completed!")
        else:
            logger.warning("Skipping data analysis - data path not found")
    
    except Exception as e:
        logger.error(f"Failed to run data analysis: {e}")

def main():
    """Main setup function"""
    logger.info("Starting Email Scam Detector Setup...")
    logger.info("="*50)
    
    # Step 1: Install requirements
    if not install_requirements():
        logger.error("Setup failed at requirements installation")
        return False
    
    # Step 2: Setup NLTK
    if not setup_nltk():
        logger.error("Setup failed at NLTK setup")
        return False
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Verify data path
    data_available = verify_data_path()
    
    # Step 5: Run data analysis (if data is available)
    if data_available:
        run_data_analysis()
    
    # Step 6: Train initial model (if data is available)
    if data_available:
        model_trained = train_initial_model()
        if model_trained:
            logger.info("Setup completed successfully!")
            logger.info("You can now run the application with: python app.py")
        else:
            logger.info("Setup completed with warnings - model training failed")
            logger.info("Please train the model manually before running the application")
    else:
        logger.info("Setup completed with warnings - no training data found")
        logger.info("Please update the data path and train the model before running the application")
    
    logger.info("="*50)
    logger.info("Setup process finished!")

if __name__ == '__main__':
    main()