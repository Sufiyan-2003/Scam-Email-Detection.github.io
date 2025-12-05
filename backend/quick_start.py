#!/usr/bin/env python3
"""
Quick start script for Email Scam Detector
This script helps you get up and running quickly
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    print("="*60)
    print("           EMAIL SCAM DETECTOR - QUICK START")
    print("="*60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_files():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'email_analyzer.py',
        'ml_model.py',
        'feature_extractor.py',
        'train_model.py',
        'data_preprocessor.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ Found: {file}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Try running manually: pip install -r requirements.txt")
        return False

def setup_nltk():
    """Setup NLTK data"""
    print("\n📚 Setting up NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except Exception:
        print("❌ Failed to setup NLTK data")
        return False

def check_dataset():
    """Check for dataset"""
    default_path = r'D:\VS-Studio\email\spamdataset'
    
    if os.path.exists(default_path):
        csv_files = [f for f in os.listdir(default_path) if f.endswith('.csv')]
        if csv_files:
            print(f"✅ Found dataset with {len(csv_files)} CSV files")
            return True
    
    print("⚠️  Dataset not found at default location")
    print("You'll need to:")
    print("  1. Update the data path in train_model.py")
    print("  2. Or place your CSV files in the expected location")
    return False

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'logs', 'temp']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

def run_training():
    """Ask user if they want to train the model"""
    print("\n🤖 Model Training")
    choice = input("Do you want to train the model now? (y/n): ").lower().strip()
    
    if choice == 'y':
        print("Training model... (this may take a few minutes)")
        try:
            subprocess.run([sys.executable, "train_model.py"], check=True)
            print("✅ Model trained successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Model training failed")
            print("You can train manually later with: python train_model.py")
            return False
    else:
        print("⏭️  Skipping model training")
        print("Remember to train before using: python train_model.py")
        return False

def show_next_steps(model_trained):
    """Show next steps to user"""
    print("\n🎉 SETUP COMPLETE!")
    print("\nNext Steps:")
    print("-" * 30)
    
    if model_trained:
        print("1. ✅ Start the web application:")
        print("   python app.py")
        print("\n2. ✅ Open your browser to:")
        print("   http://localhost:5000")
        print("\n3. ✅ Start analyzing emails!")
    else:
        print("1. 🔄 Train the model first:")
        print("   python train_model.py")
        print("\n2. 🚀 Then start the application:")
        print("   python app.py")
    
    print("\nUseful Commands:")
    print("- Analyze datasets: python data_preprocessor.py --generate_report")
    print("- Train with tuning: python train_model.py --tune_hyperparameters")
    print("- View help: python train_model.py --help")

def main():
    print_header()
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Check files
    print("\n📁 Checking files...")
    if not check_files():
        return False
    
    # Step 3: Install dependencies
    if not install_dependencies():
        return False
    
    # Step 4: Setup NLTK
    if not setup_nltk():
        return False
    
    # Step 5: Create directories
    print("\n📂 Creating directories...")
    create_directories()
    
    # Step 6: Check dataset
    print("\n💾 Checking dataset...")
    dataset_available = check_dataset()
    
    # Step 7: Train model (if dataset available)
    model_trained = False
    if dataset_available:
        model_trained = run_training()
    
    # Step 8: Show next steps
    show_next_steps(model_trained)
    
    return True

if __name__ == '__main__':
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("🎯 Quick start completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Quick start encountered issues")
        print("Please check the error messages above")
        print("="*60)
        sys.exit(1)