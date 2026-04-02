import logging 
import json 
import hashlib
import time 
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple 

import numpy as np
import xgboost as xgb
import joblib

import sys  
import argparse  
import requests

try:
	import shap
	SHAP_AVAILABLE = True
except ImportError:
	SHAP_AVAILABLE = False

# Ensure these .py files are in the same directory or Python path

# ------------------------------------------------------------
# 1. Project Paths (GLOBAL SCOPE)
# ------------------------------------------------------------
# This finds the 'phishing detection system' folder
ROOT = Path(__file__).resolve().parent.parent 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)

# Define all file paths used by the Predictor
MODEL_PATH = MODEL_DIR / "phishguard_xgb.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
SCALER_PATH = MODEL_DIR / "manual_scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SHAP_BACKGROUND_PATH = MODEL_DIR / "shap_background.npy"
WAZUH_JSONL_PATH = LOG_DIR / "phishguard_predictions.jsonl"


try:
    from src.features.preprocess import production_preprocessing
    from src.features.fasttext_features import FastTextFeatureExtractor
    from src.features.feature_concat import FeatureBuilder
    from virus_total.VT_Client import VirusTotalClient
except ImportError as e:
    print(f"CRITICAL: Missing custom module: {e}")
    sys.exit(1)
# ------------------------------------------------------------
# 2. Thresholds & Logging
# ------------------------------------------------------------
SAFE_THRESHOLD = 0.30
PHISHING_THRESHOLD = 0.70

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("PhishGuard.Predictor")

class PhishGuardPredictor:
    def __init__(self, vt_api_key: Optional[str] = None):
        logger.info("Initializing PhishGuard Production Engine...")
        
        # 1. Load XGBoost Model (JSON format)
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))
        
        # 2. Load Feature Builder
        self.builder = FeatureBuilder()
        
        # 3. Load FastText (This loads the .bin file into RAM ONCE)
        self.ft_extractor = FastTextFeatureExtractor()
        
        # 4. Initialize VT Client
        # If vt_api_key is passed here, it overrides whatever is inside vt_client.py
        self.vt_client = VT_Client(api_key=vt_api_key) 

        # 5. Load SHAP
        self.feature_names = self._load_feature_names()
        self.explainer = self._init_shap()

# ... (rest of the class methods stay the same)

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="PhishGuard Predictor CLI")
    parser.add_argument("--vt-key", help="Your VirusTotal API Key (Overrides default)")
    
    args, unknown = parser.parse_known_args()
    print("\n \t You wanna bypass me, huh, really haha ha, let's see what u have: \n \n \t press ctrl+d when u finish")

    # Read the raw email from stdin
    raw_email_input = sys.stdin.read()

    if not raw_email_input.strip():
        print("Error: No email data provided in stdin.")
        sys.exit(1)

    # Pass the user-provided API key (if any) to the Predictor
    predictor = PhishGuardPredictor(vt_api_key=args.vt_key)
    
    output = predictor.predict(raw_email_input)
    print(output)
    print(json.dumps(output, indent=2, ensure_ascii=False))
