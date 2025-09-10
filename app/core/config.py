import os
from dotenv import load_dotenv
import joblib
import logging

# Load .env
load_dotenv()

# Paths
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "artifacts/transformed/preprocessor.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model_trained/best_models/XGBoost_tuned_model.pkl")
LOG_DIR = os.getenv("LOG_DIR", "app/logs")
THRESHOLD = float(os.getenv("THRESHOLD", 0.3))
BATCH_OUTPUT_DIR = os.getenv("BATCH_OUTPUT_DIR", "app/batch_files")


# Create log directory
os.makedirs(LOG_DIR, exist_ok=True)

# Load model & preprocessor (used in predictor)
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to load model or preprocessor: {e}")
    model = None
    preprocessor = None
