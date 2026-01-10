"""
train_pipeline.py
SOC-grade phishing detector training
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

# ================= CONFIG =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/processed/combined_training_data.csv"
MODEL_PATH = PROJECT_ROOT / "models/phishing_detector.pkl"
METRICS_PATH = PROJECT_ROOT / "models/training_metrics.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"]

text_col = "cleaned_text"
numeric_cols = [c for c in X.columns if c != text_col]

# ================= PIPELINE =================
pipeline = Pipeline([
    ("features", ColumnTransformer([
        ("tfidf", TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True
        ), text_col),
        ("num", StandardScaler(), numeric_cols)
    ])),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2
    ))
])

# ================= CROSS VALIDATION =================
cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = cross_validate(
    pipeline, X, y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("📊 CV Recall:", cv_results["test_recall"].mean())
print("📊 CV Precision:", cv_results["test_precision"].mean())

# ================= TRAIN / TEST =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

pipeline.fit(X_train, y_train)

# ================= THRESHOLD TUNING =================
proba = pipeline.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, proba)

best_threshold = thresholds[np.argmax(recall)]
print(f"🎯 Optimized threshold for max recall: {best_threshold:.3f}")

# ================= SAVE =================
MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump({
    "model": pipeline,
    "threshold": float(best_threshold)
}, MODEL_PATH)

metrics = {
    "cv_recall": float(cv_results["test_recall"].mean()),
    "cv_precision": float(cv_results["test_precision"].mean()),
    "threshold": float(best_threshold),
    "trained_at": datetime.now().isoformat()
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Training complete")
print(f"📦 Model saved: {MODEL_PATH}")
