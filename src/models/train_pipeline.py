"""
train_pipeline.py - Professional Phishing Detector Training
Loads preprocessed data from manual_features.py and trains the model.
Run from project root: `python src/models/train_pipeline.py`
"""

import pandas as pd
import numpy as np
import joblib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from src.features import manual_features

# Suppress warnings
warnings.filterwarnings('ignore')

# Scikit-learn components
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# ==================== CONFIGURATION ====================
class TrainingConfig:
    """Centralized configuration for the training pipeline"""
    
    def __init__(self):
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        
        # Paths
        self.PROJECT_ROOT = project_root
        self.PREPROCESSED_DATA_PATH = project_root / 'data' / 'processed' / 'combined_training_data.csv'
        self.MODEL_SAVE_PATH = project_root / 'models' / 'phishing_detector.pkl'
        self.METRICS_SAVE_PATH = project_root / 'models' / 'training_metrics.json'
        
        # Training parameters
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        self.CV_FOLDS = 5
        
        # TF-IDF parameters (FIXED: Adjusted to prevent "no terms remain" error)
        self.TFIDF_MAX_FEATURES = 5000
        self.TFIDF_NGRAM_RANGE = (1, 2)
        self.TFIDF_MIN_DF = 2          # Minimum document frequency
        self.TFIDF_MAX_DF = 0.95       # Maximum document frequency
        self.TFIDF_STOP_WORDS = 'english'
        
        # Model parameters
        self.N_ESTIMATORS = 100
        self.CLASS_WEIGHT = 'balanced'
        
        # Feature selection
        self.MIN_FEATURE_IMPORTANCE = 0.001
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

CONFIG = TrainingConfig()

# ==================== DATA LOADER ====================
class PreprocessedDataLoader:
    """Loads and validates the preprocessed dataset from manual_features.py

    If the expected 'cleaned_text' column is missing but a 'raw_email' column
    exists, this loader will call manual_features.preprocess_batch to generate
    cleaned_text and the manual numeric features on-the-fly.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path

        # Recommended/expected manual features produced by manual_features.py.
        # These are optional: we will warn if missing but not crash (except cleaned_text/label).
        self.optional_manual_features = [
            'num_links', 'has_url', 'has_urgent_keyword', 'num_exclamations',
            'num_questions', 'has_html', 'all_caps_ratio', 'body_length',
            'num_recipients', 'has_attachment_keyword', 'subject_length',
            'subject_has_urgent'
        ]
    
    def load_and_validate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Load preprocessed data and split into features and target.

        Returns:
            Tuple of (text_features_df, numeric_features_df, labels)
        """
        print("📥 Loading preprocessed data...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at: {self.data_path}\n"
                "Please run manual_features.py first to create the dataset, or "
                "ensure your CSV includes a 'raw_email' column so this loader can preprocess it."
            )
        
        # Load the data
        df = pd.read_csv(self.data_path)
        print(f"   Loaded {len(df):,} preprocessed emails")
        print(f"   Available columns: {list(df.columns)}")
        
        # If cleaned_text is missing but raw_email exists, generate preprocessing now
        if 'cleaned_text' not in df.columns:
            if 'raw_email' in df.columns:
                print("ℹ️  'cleaned_text' missing but 'raw_email' found — generating cleaned_text and manual features using manual_features.preprocess_batch(...)")
               
                
                # Ensure raw_email column is string
                raw_emails = df['raw_email'].astype(str).tolist()
                text_df, features_df = manual_features.preprocess_batch(raw_emails)
                
                # Merge generated data into original df (preserve label and other columns)
                df = pd.concat([df.reset_index(drop=True), text_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
                print("   ✅ Generated and merged cleaned_text + manual features into DataFrame.")
            else:
                # Neither cleaned_text nor raw_email present: cannot proceed
                raise ValueError("Dataset must contain 'cleaned_text' column or a 'raw_email' column to generate it.")
        
        # Basic checks: cleaned_text and label must exist now
        if 'cleaned_text' not in df.columns:
            raise ValueError("Dataset must contain 'cleaned_text' column")
        if 'label' not in df.columns:
            raise ValueError("Dataset must contain 'label' column")
        
        # Warn about optional manual features that are missing
        missing_optional = [c for c in self.optional_manual_features if c not in df.columns]
        if missing_optional:
            print(f"⚠️  Warning: Optional/manual feature columns missing: {missing_optional}")
        
        # Split data
        text_features = df[['cleaned_text']].copy()
        
        # Get numeric features (all columns except cleaned_text and label)
        numeric_cols = [col for col in df.columns 
                       if col not in ['cleaned_text', 'label'] 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        # Fallback: if no numeric_cols detected (maybe dtype inference failed), take any non-text columns
        if not numeric_cols:
            numeric_cols = [col for col in df.columns if col not in ['cleaned_text', 'label']]
        
        numeric_features = df[numeric_cols].copy()
        labels = df['label'].copy()
        
        # Data validation
        self._validate_data(text_features, numeric_features, labels)
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Text samples: {len(text_features):,}")
        print(f"   - Numeric features: {len(numeric_features.columns)}")
        try:
            print(f"   - Class distribution: {labels.value_counts().to_dict()}")
            print(f"   - Phishing rate: {labels.mean():.2%}")
        except Exception:
            pass
        
        return text_features, numeric_features, labels
    
    def _validate_data(self, text_df: pd.DataFrame, numeric_df: pd.DataFrame, labels: pd.Series):
        """Validate the loaded data"""
        
        # Check for empty text
        empty_text = text_df['cleaned_text'].astype(str).str.strip().eq('').sum()
        if empty_text > 0:
            print(f"⚠️  Warning: {empty_text} emails have empty cleaned_text")
        
        # Check for duplicates
        duplicates = text_df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  Warning: {duplicates} duplicate emails found")
        
        # Check label distribution
        try:
            label_counts = labels.value_counts()
            if len(label_counts) != 2:
                print(f"⚠️  Warning: Expected binary labels, found {len(label_counts)} classes")
        except Exception:
            print("⚠️  Warning: Could not compute label distribution (labels may be non-numeric).")
        
        # Check for NaN values
        text_nan = text_df.isna().sum().sum()
        numeric_nan = numeric_df.isna().sum().sum()
        labels_nan = labels.isna().sum() if hasattr(labels, 'isna') else 0
        
        if text_nan > 0 or numeric_nan > 0 or labels_nan > 0:
            print(f"⚠️  Warning: Found NaN values - Text: {text_nan}, Numeric: {numeric_nan}, Labels: {labels_nan}")


# ==================== PIPELINE BUILDER ====================
class ModelPipelineBuilder:
    """Builds and configures the ML pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def build_pipeline(self, numeric_features: pd.DataFrame) -> Pipeline:
        """
        Build the complete ML pipeline.
        
        Args:
            numeric_features: DataFrame of numeric features for column names
        
        Returns:
            Configured scikit-learn Pipeline
        """
        print("\n🔧 Building ML pipeline...")
        
        # Get numeric feature names
        numeric_feature_names = numeric_features.columns.tolist()
        
        # Create the TF-IDF vectorizer with safe parameters
        tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.TFIDF_MAX_FEATURES,
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            stop_words=self.config.TFIDF_STOP_WORDS,
            min_df=self.config.TFIDF_MIN_DF,      # FIXED: Avoids "no terms remain"
            max_df=self.config.TFIDF_MAX_DF,      # FIXED: Filters common terms
            lowercase=True,                       # Already lowercase from preprocessing
            analyzer='word',
            token_pattern=r'(?u)\b\w\w+\b',       # Match words with 2+ chars
            sublinear_tf=True                     # Use 1+log(tf)
        )
        
        print(f"   TF-IDF Configuration:")
        print(f"     - Max features: {self.config.TFIDF_MAX_FEATURES}")
        print(f"     - N-gram range: {self.config.TFIDF_NGRAM_RANGE}")
        print(f"     - Min DF: {self.config.TFIDF_MIN_DF}")
        print(f"     - Max DF: {self.config.TFIDF_MAX_DF}")
        print(f"     - Stop words: {self.config.TFIDF_STOP_WORDS}")
        
        # Create ColumnTransformer
        column_transformer = ColumnTransformer([
            ('tfidf', tfidf_vectorizer, 'cleaned_text'),
            ('scaler', StandardScaler(), numeric_feature_names)
        ])
        
        # Create Random Forest classifier
        classifier = RandomForestClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            random_state=self.config.RANDOM_STATE,
            class_weight=self.config.CLASS_WEIGHT,
            n_jobs=-1,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            oob_score=True,
            verbose=0
        )
        
        print(f"   Random Forest Configuration:")
        print(f"     - Number of trees: {self.config.N_ESTIMATORS}")
        print(f"     - Class weight: {self.config.CLASS_WEIGHT}")
        
        # Build the pipeline
        pipeline = Pipeline([
            ('preprocessor', column_transformer),
            ('classifier', classifier)
        ])
        
        print("✅ Pipeline built successfully")
        return pipeline
    
    def diagnose_tfidf_issues(self, text_data: pd.DataFrame):
        """
        Diagnostic function to check TF-IDF vocabulary issues.
        
        Args:
            text_data: DataFrame with 'cleaned_text' column
        """
        print("\n🔍 Running TF-IDF diagnostics...")
        
        sample_texts = text_data['cleaned_text'].tolist()
        
        # Try different configurations
        test_configs = [
            {'min_df': 1, 'max_df': 1.0, 'name': 'No filtering'},
            {'min_df': 1, 'max_df': 0.99, 'name': 'Very loose'},
            {'min_df': 1, 'max_df': 0.95, 'name': 'Standard'},
            {'min_df': 2, 'max_df': 0.95, 'name': 'Current config'},
        ]
        
        for config in test_configs:
            try:
                vectorizer = TfidfVectorizer(
                    min_df=config['min_df'],
                    max_df=config['max_df'],
                    stop_words='english'
                )
                
                # Try to build vocabulary
                X = vectorizer.fit_transform(sample_texts)
                vocab_size = len(vectorizer.get_feature_names_out())
                
                print(f"   {config['name']}: min_df={config['min_df']}, "
                      f"max_df={config['max_df']} → Vocabulary: {vocab_size:,} terms")
                
                if vocab_size == 0:
                    print(f"   ⚠️  WARNING: Zero vocabulary with this configuration!")
                    
            except Exception as e:
                print(f"   ❌ {config['name']}: Failed with error - {str(e)[:100]}")

# ==================== MODEL TRAINER ====================
class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def train_and_evaluate(self, pipeline: Pipeline, X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train the model and evaluate performance.
        
        Returns:
            Dictionary with training metrics
        """
        print("\n🎯 Training model...")
        
        # Train the model
        pipeline.fit(X_train, y_train)
        print("✅ Model training complete")
        
        # Evaluate
        metrics = self.evaluate_model(pipeline, X_test, y_test)
        
        # Feature importance analysis
        self.analyze_feature_importance(pipeline, X_train)
        
        return metrics
    
    def evaluate_model(self, pipeline: Pipeline, X_test: pd.DataFrame,y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance (robust to single-class training/test sets)."""
        import math

        print("\n📊 Evaluating model performance...")

        # Predictions (labels)
        y_pred = pipeline.predict(X_test)

        # Safely get probability for the positive class (assumed label '1')
        y_pred_proba = None
        try:
            prob_matrix = pipeline.predict_proba(X_test)
            # Determine which column corresponds to positive label '1'
            clf = pipeline.named_steps['classifier']
            classes = getattr(clf, "classes_", None)
            if classes is not None and 1 in classes:
                pos_idx = int(np.where(classes == 1)[0][0])
                # If prob_matrix has only one column and it's the positive class, ok
                if prob_matrix.shape[1] > pos_idx:
                    y_pred_proba = prob_matrix[:, pos_idx]
                else:
                    # Unexpected shape -- fall back to zeros
                    y_pred_proba = np.zeros(len(X_test))
            else:
                # Positive class not seen during training -> probability for positive = 0
                y_pred_proba = np.zeros(len(X_test))
        except Exception:
            # If predict_proba not available or failed, fall back to zeros
            y_pred_proba = np.zeros(len(X_test))

        # Calculate accuracy and safe f1 (set zero_division to avoid exceptions)
        accuracy = np.mean(y_pred == y_test)
        try:
            f1 = f1_score(y_test, y_pred, zero_division=0)
        except Exception:
            f1 = 0.0

        # Compute confusion matrix forcing labels [0,1] to guarantee 2x2 output
        try:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        except Exception:
            # Fallback: construct confusion from counts
            tn = int(((y_test == 0) & (y_pred == 0)).sum())
            fp = int(((y_test == 0) & (y_pred == 1)).sum())
            fn = int(((y_test == 1) & (y_pred == 0)).sum())
            tp = int(((y_test == 1) & (y_pred == 1)).sum())

        # Safe precision/recall calculation (avoid ZeroDivision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Safe ROC-AUC: only compute if both classes present in y_test and proba array is valid
        roc_auc = float('nan')
        try:
            if len(np.unique(y_test)) > 1 and y_pred_proba is not None:
                roc_auc = float(roc_auc_score(y_test, y_pred_proba))
            else:
                roc_auc = float('nan')
        except Exception:
            roc_auc = float('nan')

        # Print results
        print(f"   📈 Performance Metrics:")
        print(f"      Accuracy:           {accuracy:.4f}")
        print(f"      ROC-AUC:            {roc_auc if not math.isnan(roc_auc) else 'N/A'}")
        print(f"      F1-Score:           {f1:.4f}")
        print(f"      Precision:          {precision:.4f}")
        print(f"      Recall:             {recall:.4f}")
        print(f"      False Positive Rate: {false_positive_rate:.4f}")

        print(f"\n   🎯 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Legit   Phishing")
        print(f"      Actual Legit    {tn:6d}      {fp:6d}")
        print(f"      Actual Phishing {fn:6d}      {tp:6d}")

        print(f"\n   📋 Classification Report:")
        try:
            print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'], zero_division=0))
        except Exception as e:
            print(f"   ⚠️  Could not produce classification report: {e}")

        return {
            'accuracy': float(accuracy),
            'roc_auc': (roc_auc if not math.isnan(roc_auc) else None),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'false_positive_rate': float(false_positive_rate),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'test_samples': len(X_test),
            'evaluation_date': datetime.now().isoformat()
        }

    
    def analyze_feature_importance(self, pipeline: Pipeline, X_train: pd.DataFrame):
        """Analyze feature importance if available"""
        print("\n🔍 Analyzing feature importance...")
        
        try:
            classifier = pipeline.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                # Get feature names
                preprocessor = pipeline.named_steps['preprocessor']
                
                # Get TF-IDF feature names
                tfidf_transformer = preprocessor.named_transformers_['tfidf']
                if hasattr(tfidf_transformer, 'get_feature_names_out'):
                    tfidf_features = tfidf_transformer.get_feature_names_out().tolist()
                else:
                    tfidf_features = [f"word_{i}" for i in range(tfidf_transformer.max_features)]
                
                # Get numeric feature names
                numeric_features = [col for col in X_train.columns if col != 'cleaned_text']
                
                # Combine all features
                all_features = tfidf_features + numeric_features
                
                # Get importances
                importances = classifier.feature_importances_
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': all_features[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Filter by threshold
                important_features = importance_df[
                    importance_df['importance'] > self.config.MIN_FEATURE_IMPORTANCE
                ]
                
                print(f"   Top 10 most important features:")
                for idx, row in important_features.head(10).iterrows():
                    print(f"      {row['feature']:30} : {row['importance']:.6f}")
                
                # Save feature importance
                importance_path = self.config.MODEL_SAVE_PATH.parent / 'feature_importance.csv'
                importance_df.to_csv(importance_path, index=False)
                print(f"\n   💾 Feature importance saved to: {importance_path}")
                
        except Exception as e:
            print(f"   ⚠️  Feature importance analysis skipped: {str(e)[:100]}")

# ==================== MAIN TRAINING WORKFLOW ====================
def main_training_workflow() -> Optional[Pipeline]:
    """
    Main training workflow.
    
    Returns:
        Trained pipeline if successful, None otherwise
    """
    print("=" * 70)
    print("🤖 PROFESSIONAL PHISHING DETECTOR TRAINING")
    print("=" * 70)
    print(f"Project: {CONFIG.PROJECT_ROOT.name}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Ensure model directory exists
        CONFIG.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load preprocessed data
        data_loader = PreprocessedDataLoader(CONFIG.PREPROCESSED_DATA_PATH)
        text_features, numeric_features, labels = data_loader.load_and_validate()
        
        # Combine features for train/test split
        X = pd.concat([text_features, numeric_features], axis=1)
        y = labels
        
        # Step 2: Split data
        print(f"\n📊 Splitting data (test_size={CONFIG.TEST_SIZE})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=CONFIG.TEST_SIZE,
            random_state=CONFIG.RANDOM_STATE,
            stratify=y
        )
        
        print(f"   Training set: {len(X_train):,} emails")
        print(f"   Test set:     {len(X_test):,} emails")
        print(f"   Phishing rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
        
        # Step 3: Run TF-IDF diagnostics if needed
        if CONFIG.TFIDF_MIN_DF > 1 or CONFIG.TFIDF_MAX_DF < 1.0:
            pipeline_builder = ModelPipelineBuilder(CONFIG)
            pipeline_builder.diagnose_tfidf_issues(X_train)
        
        # Step 4: Build pipeline
        pipeline_builder = ModelPipelineBuilder(CONFIG)
        pipeline = pipeline_builder.build_pipeline(numeric_features)
        
        # Step 5: Train and evaluate
        trainer = ModelTrainer(CONFIG)
        metrics = trainer.train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
        
        # Step 6: Save model and metrics
        print(f"\n💾 Saving artifacts...")
        
        # Save model
        joblib.dump(pipeline, CONFIG.MODEL_SAVE_PATH)
        print(f"   ✅ Model saved to: {CONFIG.MODEL_SAVE_PATH}")
        
        # Save metrics
        with open(CONFIG.METRICS_SAVE_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   ✅ Metrics saved to: {CONFIG.METRICS_SAVE_PATH}")
        
        # Save training configuration
        config_path = CONFIG.MODEL_SAVE_PATH.parent / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(CONFIG.to_dict(), f, indent=2, default=str)
        print(f"   ✅ Configuration saved to: {config_path}")
        
        # Step 7: Test with examples
                # ---------------------------
        # ---------------------------
        print(f"\n🧪 Model testing with examples...")
        test_examples = [
            ("urgent verify your account now click http secure login com", 1),
            ("hi team meeting at 3 pm tomorrow conference room", 0),
        ]

        correct = 0
        for text, expected in test_examples:
            test_df = pd.DataFrame({
                'cleaned_text': [text],
                **{col: [0] for col in numeric_features.columns}
            })

            # Prediction (label)
            pred = pipeline.predict(test_df)[0]

            # Safe probability for the positive class (label == 1)
            try:
                prob_matrix = pipeline.predict_proba(test_df)
                clf = pipeline.named_steps['classifier']
                classes = getattr(clf, "classes_", None)

                if classes is not None and 1 in classes:
                    pos_idx = int(np.where(classes == 1)[0][0])
                    if prob_matrix.shape[1] > pos_idx:
                        proba = float(prob_matrix[0, pos_idx])
                    else:
                        proba = 0.0
                else:
                    # Positive class not seen during training -> probability = 0
                    proba = 0.0
            except Exception:
                # If predict_proba not available or failed, use 0.0
                proba = 0.0

            is_correct = pred == expected
            if is_correct:
                correct += 1

            status = "✅" if is_correct else "❌"
            print(f"   {status} '{text[:50]}...' → "
                  f"Pred: {pred} ({proba:.1%}), Expected: {expected}")

        print(f"\n   Test accuracy: {correct}/{len(test_examples)}")

        
                # Final summary
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Final Model Performance:")

        def _fmt(val, digits=4):
            """Format numeric metric or return 'N/A' if None or not a number."""
            try:
                if val is None:
                    return "N/A"
                if isinstance(val, float) and np.isnan(val):
                    return "N/A"
                return f"{val:.{digits}f}"
            except Exception:
                return "N/A"

        print(f"   Accuracy: {_fmt(metrics.get('accuracy'))}")
        print(f"   ROC-AUC:  {_fmt(metrics.get('roc_auc'))}")
        print(f"   F1-Score: {_fmt(metrics.get('f1_score'))}")



        
        print(f"\n🚀 Deployment Instructions:")
        print(f"   1. Load model: model = joblib.load('{CONFIG.MODEL_SAVE_PATH.relative_to(CONFIG.PROJECT_ROOT)}')")
        print(f"   2. Prepare input: DataFrame with columns: {list(numeric_features.columns)} + ['cleaned_text']")
        print(f"   3. Predict: model.predict(new_data)")
        
        return pipeline
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}")
    print(f"Working directory: {Path.cwd()}")
    
    try:
        # Run the training workflow
        trained_model = main_training_workflow()
        
        if trained_model is not None:
            print("\n" + "=" * 70)
            print("✅ All tasks completed successfully!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("❌ Training failed. Please check the error messages above.")
            print("=" * 70)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
