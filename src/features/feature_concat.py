#!/usr/bin/env python3
"""
prepare_for_xgboost.py
Loads the preprocessed CSV, computes FastText embeddings for clean_text,
extracts numeric features, concatenates them, and saves the combined matrix.
"""

import os
import numpy as np
import pandas as pd
import fasttext
import fasttext.util
import joblib
from pathlib import Path

# ==================== CONFIGURATION ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "phishguard_features.csv"
FASTTEXT_MODEL_PATH = PROJECT_ROOT / "models" / "cc.en.300.bin"   # adjust as needed
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = "xgboost_features.npz"

# If you want to save the scaler for manual features (optional)
SAVE_SCALER = True
SCALER_PATH = PROJECT_ROOT / "models" / "manual_scaler.pkl"

# ==================== LOAD FASTTEXT MODEL ====================
print("📥 Loading FastText model...")
if not FASTTEXT_MODEL_PATH.exists():
    # Download if not present
    fasttext.util.download_model('en', if_exists='ignore')  # downloads to current dir
    # Move it to the models folder
    import shutil
    shutil.move('cc.en.300.bin', FASTTEXT_MODEL_PATH)
ft = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
print("✅ FastText model loaded.")

# ==================== LOAD CSV ====================
print(f"📥 Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
print(f"   Loaded {len(df)} rows.")

# ==================== DEFINE NUMERIC FEATURES ====================
# These are the columns that are already numeric (or will be derived)
numeric_base = [
    'urgent_words_count', 'digit_ratio', 'body_entropy', 'html_present',
    'auth_headers_present', 'spf_result', 'dkim_result', 'dmarc_result',
    'received_count'
]

# Convert these columns to float (they might be strings in CSV)
for col in numeric_base:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
    else:
        print(f"⚠️  Warning: column '{col}' not found, creating with zeros.")
        df[col] = 0.0

# ==================== CREATE COUNT FEATURES FROM LIST COLUMNS ====================
list_columns = ['urls', 'domains', 'ip_urls', 'attachment_names']
for col in list_columns:
    if col in df.columns:
        # Count semicolon-separated items; if empty string, count = 0
        df[f'{col}_count'] = df[col].apply(lambda x: len(x.split(';')) if x.strip() else 0).astype(np.float32)
    else:
        print(f"⚠️  Warning: column '{col}' not found, skipping count.")
        df[f'{col}_count'] = 0.0

count_features = ['urls_count', 'domains_count', 'ip_urls_count', 'attachment_names_count']

# Combine all manual numeric features
manual_feature_names = numeric_base + count_features
print(f"📊 Manual feature names: {manual_feature_names}")

# ==================== EXTRACT LABELS ====================
if 'label' not in df.columns:
    raise ValueError("CSV must contain a 'label' column")
y = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(np.int32)
# Filter out rows with invalid label (-1)
valid = y != -1
df = df[valid].reset_index(drop=True)
y = y[valid].reset_index(drop=True)
print(f"   Valid rows after label check: {len(df)}")

# ==================== GENERATE EMBEDDINGS ====================
print("🧠 Generating FastText embeddings for clean_text...")
embeddings = []
for idx, row in df.iterrows():
    text = row.get('clean_text', '')
    if not text or text == '<EMPTY>':
        # Fallback: use a zero vector
        emb = np.zeros(300, dtype=np.float32)
    else:
        emb = ft.get_sentence_vector(text).astype(np.float32)
    embeddings.append(emb)
    if (idx + 1) % 5000 == 0:
        print(f"   Processed {idx+1}/{len(df)} emails")

X_emb = np.stack(embeddings)
print(f"✅ Embeddings shape: {X_emb.shape}")

# ==================== EXTRACT MANUAL NUMERIC FEATURES ====================
X_manual = df[manual_feature_names].values.astype(np.float32)
print(f"✅ Manual features shape: {X_manual.shape}")

# (Optional) Scale manual features
if SAVE_SCALER:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_manual_scaled = scaler.fit_transform(X_manual)
    joblib.dump(scaler, SCALER_PATH)
    print(f"   Scaler saved to {SCALER_PATH}")
    X_manual = X_manual_scaled

# ==================== CONCATENATE ====================
X_combined = np.hstack([X_emb, X_manual])
print(f"✅ Combined feature matrix shape: {X_combined.shape}")

# ==================== SAVE ====================
output_path = OUTPUT_DIR / OUTPUT_FILE
np.savez_compressed(output_path, X=X_combined, y=y.values, 
                    feature_names=manual_feature_names)
print(f"💾 Combined features saved to {output_path}")

# Also save a small sample for inspection
sample_path = OUTPUT_DIR / "xgboost_features_sample.npz"
np.savez(sample_path, X=X_combined[:100], y=y.values[:100])
print(f"   Sample (first 100) saved to {sample_path}")

print("\n✅ Preparation complete. Now you can train XGBoost by loading:")
print("   data = np.load('data/processed/xgboost_features.npz')")
print("   X, y = data['X'], data['y']")
