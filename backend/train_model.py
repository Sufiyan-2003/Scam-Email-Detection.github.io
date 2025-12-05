#!/usr/bin/env python3
import os, sys, argparse, logging
from ml_model import EmailClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Advanced Hybrid Email Scam Detector')
    parser.add_argument('--data_path', type=str, default=r'D:\VS-Studio\email\spamdataset')
    parser.add_argument('--output_path', type=str, default='data/hybrid_model.pkl')
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        logger.error(f"❌ Data path not found: {args.data_path}")
        sys.exit(1)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    try:
        logger.info("=" * 70)
        logger.info("🚀 Training Hybrid Email Scam Detector (KMeans + SOM + RF)")
        logger.info("=" * 70)

        classifier = EmailClassifier(n_clusters=3, som_x=12, som_y=12)
        results = classifier.train(args.data_path, test_size=args.test_size)

        logger.info("\n📊 PERFORMANCE METRICS")
        logger.info(f"Accuracy : {results['test_accuracy']*100:.2f}%")
        logger.info(f"Precision: {results['precision']*100:.2f}%")
        logger.info(f"Recall   : {results['recall']*100:.2f}%")
        logger.info(f"F1 Score : {results['f1_score']*100:.2f}%")

        classifier.save_model(args.output_path)
        logger.info("✅ Model saved successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
