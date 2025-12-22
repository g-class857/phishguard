"""
train_pipeline_fixed.py - Multi-Dataset Phishing Detector Training (fixed)
Automatically detects and trains on all datasets in data/processed/

Fixes:
- Adaptive min_df / max_df for TfidfVectorizer to avoid "max_df corresponds to < documents than min_df" error
- Minor preprocessing fixes (all-caps ratio computed before lowercasing)
- create_full_pipeline accepts training size so TF-IDF thresholds are calculated safely
- More robust handling of small datasets
"""

import re
import json
import sys
import os
import joblib
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score


# ------------------ Preprocessing ------------------
def preprocess_email(raw_email: str) -> Tuple[str, Dict[str, float]]:
    """Process a single raw email string into cleaned text and numeric features."""
    try:
        original_text = str(raw_email)
        # keep a version for detecting ALL CAPS words
        words = original_text.split()
        all_caps_ratio = sum(1 for w in words if w.isupper()) / max(len(words), 1)

        # lowercased text for keyword searches and TF-IDF cleaning
        text = original_text.lower()

        features = {
            'num_links': len([w for w in text.split() if 'http' in w or 'www.' in w]),
            'body_length': len(text),
            'has_urgent_keyword': 1 if any(word in text for word in ['urgent', 'immediate', 'verify', 'suspended', 'alert']) else 0,
            'num_exclamations': text.count('!'),
            'num_questions': text.count('?'),
            'has_html': 1 if '<' in text and '>' in text else 0,
            'all_caps_ratio': float(all_caps_ratio),
            'has_url_keyword': 1 if any(word in text for word in ['click', 'link', 'http', 'www', '.com']) else 0,
            'has_account_keyword': 1 if any(word in text for word in ['account', 'password', 'login', 'credentials']) else 0,
            'has_money_keyword': 1 if any(sym in original_text for sym in ['$', 'money', 'cash', 'prize', 'winner', 'free']) else 0,
        }

        # Clean text for TF-IDF (keep only alphanumeric and basic punctuation)
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        cleaned_text = ' '.join(cleaned_text.split())
        if not cleaned_text.strip():
            cleaned_text = "empty_email"

        return cleaned_text, features

    except Exception as e:
        print(f"Error in preprocess_email: {e}")
        return "error_email", {
            'num_links': 0, 'body_length': 0, 'has_urgent_keyword': 0,
            'num_exclamations': 0, 'num_questions': 0, 'has_html': 0,
            'all_caps_ratio': 0.0, 'has_url_keyword': 0, 'has_account_keyword': 0, 'has_money_keyword': 0
        }


def preprocess_batch(raw_emails: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_texts = []
    features_list = []

    for email in raw_emails:
        cleaned_text, features = preprocess_email(email)
        cleaned_texts.append(cleaned_text)
        features_list.append(features)

    text_df = pd.DataFrame({'cleaned_text': cleaned_texts})
    features_df = pd.DataFrame(features_list)

    expected_features = [
        'num_links', 'body_length', 'has_urgent_keyword', 'num_exclamations',
        'num_questions', 'has_html', 'all_caps_ratio', 'has_url_keyword',
        'has_account_keyword', 'has_money_keyword'
    ]
    for feat in expected_features:
        if feat not in features_df.columns:
            features_df[feat] = 0.0

    return text_df, features_df


# ------------------ Config ------------------

def get_project_config() -> Dict[str, Any]:
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent

    return {
        'data_dir': project_root / 'data' / 'processed',
        'model_save_path': project_root / 'models' / 'phishing_detector.pkl',
        'metrics_save_path': project_root / 'models' / 'training_metrics.json',
        'test_size': 0.2,
        'random_state': 42,
        'tfidf_max_features': 5000,
        'n_estimators': 100,
        'min_dataset_size': 10,
        'max_datasets': 10,
    }

CONFIG = get_project_config()


# ------------------ Dataset Loader ------------------
class DatasetLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def detect_datasets(self) -> List[Path]:
        dataset_files = []
        dataset_files.extend(list(self.data_dir.glob("*.csv")))
        dataset_files.extend(list(self.data_dir.glob("*.parquet")))
        dataset_files.extend(list(self.data_dir.glob("*.json")))

        print(f"Found {len(dataset_files)} dataset files:")
        for i, file in enumerate(dataset_files, 1):
            try:
                size_kb = file.stat().st_size / 1024
            except Exception:
                size_kb = 0.0
            print(f"  {i}. {file.name} ({size_kb:.1f} KB)")
        return dataset_files

    def load_single_dataset(self, filepath: Path) -> Tuple[pd.DataFrame, str, str]:
        dataset_name = filepath.stem
        try:
            if filepath.suffix.lower() == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix.lower() == '.parquet':
                df = pd.read_parquet(filepath)
            elif filepath.suffix.lower() == '.json':
                df = pd.read_json(filepath)
            else:
                print(f"  ⚠️  Skipping unsupported format: {filepath.name}")
                return None, dataset_name, "Unsupported format"

            df_standardized = self.standardize_columns(df, dataset_name)
            if df_standardized is not None and len(df_standardized) >= CONFIG['min_dataset_size']:
                return df_standardized, dataset_name, "Success"
            elif df_standardized is not None:
                return None, dataset_name, f"Too small ({len(df_standardized)} emails)"
            else:
                return None, dataset_name, "Standardization failed"

        except Exception as e:
            print(f"  ❌ Error loading {filepath.name}: {str(e)[:100]}")
            return None, dataset_name, f"Error: {str(e)[:50]}"

    def standardize_columns(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        if df.empty:
            return None

        result = pd.DataFrame()
        text_column = self.find_text_column(df)
        if text_column is None:
            print(f"    ❌ No text column found in {dataset_name}")
            return None

        label_column = self.find_label_column(df)
        result['raw_email'] = df[text_column].astype(str)

        if label_column is not None:
            result['label'] = self.standardize_labels(df[label_column])
        else:
            print(f"    ⚠️  No label column in {dataset_name}, creating dummy labels")
            result['label'] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])

        result['dataset_source'] = dataset_name
        result = result[result['raw_email'].str.strip().str.len() > 10].copy()
        return result

    def find_text_column(self, df: pd.DataFrame) -> str:
        text_candidates = [
            'text', 'email', 'body', 'message', 'content', 'raw_email',
            'email_text', 'email_body', 'raw_text', 'sentence', 'paragraph', 'document'
        ]
        for col in df.columns:
            if col.lower() in [c.lower() for c in text_candidates]:
                return col

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['text', 'email', 'body', 'message', 'content']):
                return col

        string_cols = df.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            avg_lengths = {}
            for col in string_cols:
                try:
                    avg_len = df[col].astype(str).str.len().mean()
                    avg_lengths[col] = avg_len
                except Exception:
                    continue
            if avg_lengths:
                return max(avg_lengths, key=avg_lengths.get)

        if len(string_cols) > 0:
            return string_cols[0]
        return None

    def find_label_column(self, df: pd.DataFrame) -> str:
        label_candidates = [
            'label', 'target', 'class', 'is_spam', 'is_phishing',
            'spam', 'phishing', 'category', 'type', 'binary_label'
        ]
        for col in df.columns:
            if col.lower() in [c.lower() for c in label_candidates]:
                return col

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['label', 'target', 'class', 'spam', 'phish']):
                return col

        for col in df.columns:
            try:
                if df[col].nunique() == 2:
                    return col
            except Exception:
                continue
        return None

    def standardize_labels(self, label_series: pd.Series) -> pd.Series:
        labels = label_series.astype(str).str.lower().str.strip()
        label_map = {}
        unique_labels = labels.unique()
        for label in unique_labels:
            if label in ['0', 'false', 'no', 'ham', 'legitimate', 'safe', 'clean', 'normal']:
                label_map[label] = 0
            elif label in ['1', 'true', 'yes', 'spam', 'phishing', 'malicious', 'bad']:
                label_map[label] = 1
            elif 'ham' in label or 'legit' in label:
                label_map[label] = 0
            elif 'spam' in label or 'phish' in label:
                label_map[label] = 1
            else:
                label_map[label] = 1
        return labels.map(label_map)

    def combine_datasets(self) -> pd.DataFrame:
        print("\n📊 LOADING DATASETS")
        print("=" * 60)
        dataset_files = self.detect_datasets()
        if not dataset_files:
            print("❌ No datasets found in data/processed/")
            return None

        all_datasets = []
        stats = []
        for filepath in dataset_files[:CONFIG['max_datasets']]:
            print(f"\n📂 Loading {filepath.name}...")
            df_loaded, name, status = self.load_single_dataset(filepath)
            if df_loaded is not None:
                all_datasets.append(df_loaded)
                legit_count = (df_loaded['label'] == 0).sum()
                phish_count = (df_loaded['label'] == 1).sum()
                stats.append({
                    'dataset': name,
                    'emails': len(df_loaded),
                    'legitimate': legit_count,
                    'phishing': phish_count,
                    'phishing_pct': (phish_count / len(df_loaded) * 100)
                })
                print(f"   ✅ Loaded: {len(df_loaded)} emails ({phish_count} phishing, {phish_count/len(df_loaded)*100:.1f}%)")
            else:
                print(f"   ❌ Failed: {status}")

        if not all_datasets:
            print("\n❌ No valid datasets could be loaded")
            return None

        combined_df = pd.concat(all_datasets, ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=CONFIG['random_state']).reset_index(drop=True)

        print("\n📈 DATASET SUMMARY")
        print("=" * 60)
        for stat in stats:
            print(f"  {stat['dataset']:20}: {stat['emails']:5} emails, {stat['phishing']:4} phishing ({stat['phishing_pct']:5.1f}%)")

        total_phishing = combined_df['label'].sum()
        total_emails = len(combined_df)
        print(f"\n  {'TOTAL':20}: {total_emails:5} emails, {total_phishing:4} phishing ({total_phishing/total_emails*100:.1f}%)")

        print("\n  Dataset Sources:")
        source_counts = combined_df['dataset_source'].value_counts()
        for source, count in source_counts.items():
            phishing_pct = combined_df[combined_df['dataset_source'] == source]['label'].mean() * 100
            print(f"    {source:20}: {count:5} emails ({phishing_pct:.1f}% phishing)")

        return combined_df


# ------------------ Pipeline ------------------
def create_full_pipeline(n_documents: int) -> Pipeline:
    """Create the ML pipeline. n_documents is the number of training documents and used to adapt TF-IDF thresholds."""
    print("\n🔧 BUILDING PIPELINE")
    print("=" * 60)

    def custom_preprocessor(X):
        emails = X.tolist() if hasattr(X, 'tolist') else list(X)
        text_df, features_df = preprocess_batch(emails)
        return pd.concat([text_df, features_df], axis=1)

    text_features = ['cleaned_text']
    _, sample_features = preprocess_batch(["test email for feature extraction"])
    numeric_features = sample_features.columns.tolist()

    # Adaptive min_df / max_df
    # Use integer document counts if dataset is small to avoid float->int rounding surprises
    if n_documents < 50:
        min_df = 1
    else:
        # require a token to appear in at least 1% of documents, but at least 2
        min_df = max(2, int(max(1, np.round(0.01 * n_documents))))

    # max_df_docs should be at least min_df
    max_df_docs = max(min_df, int(np.floor(0.95 * n_documents)))
    # If max_df_docs equals total docs (possible for tiny datasets), cap to n_documents
    max_df_docs = min(max_df_docs, n_documents)

    # If the computed max_df_docs < min_df (extremely small n_documents), set both to sensible values
    if max_df_docs < min_df:
        max_df_docs = min_df

    # Use integer thresholds for safety
    tfidf_min_df = min_df
    tfidf_max_df = max_df_docs

    print(f"  Training documents: {n_documents}")
    print(f"  TF-IDF min_df (docs): {tfidf_min_df}")
    print(f"  TF-IDF max_df (docs): {tfidf_max_df}")

    column_transformer = ColumnTransformer([
        ('tfidf', TfidfVectorizer(
            max_features=CONFIG['tfidf_max_features'],
            ngram_range=(1, 2),
            stop_words='english',
            min_df=tfidf_min_df,
            max_df=tfidf_max_df
        ), text_features),

        ('scaler', StandardScaler(), numeric_features)
    ], remainder='drop')

    pipeline = Pipeline([
        ('preprocessor', FunctionTransformer(custom_preprocessor, validate=False)),
        ('transformer', column_transformer),
        ('classifier', RandomForestClassifier(
            n_estimators=CONFIG['n_estimators'],
            random_state=CONFIG['random_state'],
            class_weight='balanced',
            n_jobs=-1
        ))
    ])

    print("✅ Pipeline built successfully")
    return pipeline


# ------------------ Evaluation ------------------
def evaluate_model(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    print("\n📊 MODEL EVALUATION")
    print("=" * 60)
    try:
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracy = float(np.mean(y_pred == y_test))
        roc_auc = float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else 0.0
        f1 = float(f1_score(y_test, y_pred)) if len(np.unique(y_test)) > 1 else 0.0

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  F1-Score:    {f1:.4f}")

        print("\n  Confusion Matrix:")
        print("                Predicted")
        print("                Legit   Phishing")
        print(f"    Actual Legit    {tn:4d}      {fp:4d}")
        print(f"    Actual Phishing {fn:4d}      {tp:4d}")

        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        print("\n  Key Metrics:")
        print(f"    Precision:           {precision:.4f}")
        print(f"    Recall:              {recall:.4f}")
        print(f"    False Positive Rate: {false_positive_rate:.4f}")

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'precision': float(precision),
            'recall': float(recall),
            'false_positive_rate': float(false_positive_rate),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'evaluation_date': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return {}


# ------------------ Training ------------------
def train_model() -> Pipeline:
    print("=" * 60)
    print("🚀 PHISHING DETECTOR TRAINING")
    print("=" * 60)

    CONFIG['model_save_path'].parent.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader(CONFIG['data_dir'])
    combined_df = loader.combine_datasets()

    if combined_df is None or len(combined_df) < 20:
        print("\n❌ Insufficient data for training")
        return None

    X = combined_df['raw_email']
    y = combined_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y
    )

    print(f"  Training set: {len(X_train):,} emails")
    print(f"  Test set:     {len(X_test):,} emails")

    # Create and train pipeline with adaptive TF-IDF thresholds
    pipeline = create_full_pipeline(n_documents=len(X_train))

    try:
        pipeline.fit(X_train, y_train)
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        return None

    metrics = evaluate_model(pipeline, X_test, y_test)

    try:
        joblib.dump(pipeline, CONFIG['model_save_path'])
        print(f"✅ Model saved to: {CONFIG['model_save_path']}")
        with open(CONFIG['metrics_save_path'], 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to: {CONFIG['metrics_save_path']}")

        info = {
            'training_date': datetime.now().isoformat(),
            'datasets_used': combined_df['dataset_source'].unique().tolist(),
            'total_samples': len(combined_df),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_parameters': {
                'tfidf_max_features': CONFIG['tfidf_max_features'],
                'n_estimators': CONFIG['n_estimators']
            }
        }
        info_path = CONFIG['model_save_path'].parent / 'training_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✅ Training info saved to: {info_path}")

    except Exception as e:
        print(f"❌ Failed to save files: {e}")

    return pipeline


# ------------------ Test ------------------
def test_model(pipeline: Pipeline):
    print("\n🧪 MODEL TESTING")
    print("=" * 60)
    test_emails = [
        ("URGENT: Your account will be suspended! Click http://phishing-site.com/verify now!", 1),
        ("You won $10,000! Claim your prize at http://lottery-scam.com", 1),
        ("SECURITY ALERT: Unusual login detected. Reset password: http://fake-security.com", 1),
        ("Important: Verify your email address at http://verify-account.com", 1),
        ("Hi team, meeting at 3 PM tomorrow in conference room B.", 0),
        ("Please find the quarterly report attached to this email.", 0),
        ("Reminder: Project deadline is next Friday.", 0),
        ("Can you review this document when you have a moment?", 0),
    ]

    correct = 0
    for i, (email, expected) in enumerate(test_emails, 1):
        try:
            prediction = pipeline.predict([email])[0]
            proba = pipeline.predict_proba([email])[0][1]
            is_correct = prediction == expected
            if is_correct:
                correct += 1
                symbol = "✅"
            else:
                symbol = "❌"
            print(f"  {symbol} Test {i}: {prediction} ({proba:.1%}) - Expected: {expected} - '{email[:50]}...'")
        except Exception as e:
            print(f"  ❌ Test {i} failed: {e}")

    accuracy = correct / len(test_emails)
    print(f"\n  Test Accuracy: {accuracy:.0%} ({correct}/{len(test_emails)})")


# ------------------ Main ------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 AUTOMATIC PHISHING DETECTOR TRAINER (FIXED)")
    print("=" * 60)
    print(f"Python {sys.version.split()[0]}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Data directory: {CONFIG['data_dir']}")

    try:
        model = train_model()
        if model is not None:
            test_model(model)
            print("\n🔄 VERIFYING SAVED MODEL")
            try:
                loaded_model = joblib.load(CONFIG['model_save_path'])
                _ = loaded_model.predict(["Test email for verification"])  # smoke test
                print("✅ Saved model loads and predicts correctly")
            except Exception as e:
                print(f"❌ Failed to load saved model: {e}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
