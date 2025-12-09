"""
manual_features.py
Contains the core functions for preprocessing raw emails and extracting features.
This logic is used identically during training and in production.
"""

import re
import html
from typing import Tuple, Dict, List
import pandas as pd

# ==================== CORE PREPROCESSING FUNCTION ====================
def preprocess_email(raw_email: str) -> Tuple[str, Dict[str, float]]:
    """
    Transforms a single raw email string into clean text and a dictionary of features.
    This is the main function that will be called for every email in production.
    
    Args:
        raw_email (str): The complete raw email text, including headers and body.
    
    Returns:
        Tuple[str, Dict]: 
            - cleaned_text (str): The normalized, cleaned body text for TF-IDF.
            - features (Dict[str, float]): A dictionary of extracted structural features.
    """
    
    # === 1. INITIAL CLEANING & DECODING ===
    # Handle HTML entities and common encoding artifacts
    text = html.unescape(raw_email)
    text = re.sub(r'=\s*\n', '', text)  # Remove soft line breaks (common in email encoding)
    text = re.sub(r'=([A-F0-9]{2})', lambda m: chr(int(m.group(1), 16)), text)  # Decode hex chars like =2E
    
    # === 2. EXTRACT STRUCTURAL FEATURES (BEFORE MODIFYING TEXT) ===
    # These features are crucial for phishing detection and are extracted from the raw text
    features = {}
    
    # Link/URL features
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    features['num_links'] = len(urls)
    features['has_url'] = 1.0 if features['num_links'] > 0 else 0.0
    
    # Urgency and tone features
    urgent_keywords = ['urgent', 'immediate', 'asap', 'action required', 'verify', 'suspend', 'account', 'security', 'alert']
    urgent_pattern = r'\b(' + '|'.join(urgent_keywords) + r')\b'
    features['has_urgent_keyword'] = 1.0 if re.search(urgent_pattern, text, re.IGNORECASE) else 0.0
    features['num_exclamations'] = text.count('!')
    features['num_questions'] = text.count('?')
    
    # Formatting features (common in phishing)
    features['has_html'] = 1.0 if bool(re.search(r'<[^>]+>', text)) else 0.0
    features['all_caps_ratio'] = len(re.findall(r'\b[A-Z]{2,}\b', text)) / max(len(re.findall(r'\b[A-Za-z]+\b', text)), 1)
    
    # Structural features
    features['body_length'] = len(text)
    features['num_recipients'] = len(re.findall(r'To:|CC:|BCC:', text, re.IGNORECASE))
    features['has_attachment_keyword'] = 1.0 if re.search(r'\b(attachment|attached|enclosed)\b', text, re.IGNORECASE) else 0.0
    
    # === 3. SEPARATE HEADERS FROM BODY ===
    # Find the first empty line (standard email header-body separator)
    parts = re.split(r'\n\s*\n', text, maxsplit=1)
    headers = parts[0] if len(parts) > 0 else ''
    body = parts[1] if len(parts) > 1 else headers
    
    # Extract subject for additional features
    subject_match = re.search(r'Subject:\s*(.*?)(?:\n|$)', headers, re.IGNORECASE | re.DOTALL)
    subject = subject_match.group(1).strip() if subject_match else ''
    features['subject_length'] = len(subject)
    features['subject_has_urgent'] = 1.0 if re.search(urgent_pattern, subject, re.IGNORECASE) else 0.0
    
    # === 4. CLEAN BODY TEXT FOR TF-IDF ANALYSIS ===
    # Remove HTML tags but preserve their text content
    body = re.sub(r'<script.*?</script>', ' ', body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r'<style.*?</style>', ' ', body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r'<[^>]+>', ' ', body)
    
    # Replace specific patterns with tokens (helps model learn patterns, not specifics)
    body = re.sub(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', ' URL_TOKEN ', body, flags=re.IGNORECASE)
    body = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' EMAIL_TOKEN ', body)
    body = re.sub(r'\$\d+(?:\.\d{1,2})?|\d+\s*(?:dollars?|usd|eur|gbp)\b', ' MONEY_TOKEN ', body, flags=re.IGNORECASE)
    body = re.sub(r'\b\d{10,}\b', ' NUMBER_TOKEN ', body)  # Long numbers (like phone/account)
    
    # Remove remaining special characters but keep basic sentence structure
    body = re.sub(r'[^\w\s\.\,\!\?]', ' ', body)
    
    # Normalize whitespace and convert to lowercase
    body = ' '.join(body.split())
    cleaned_text = body.lower().strip()
    
    # Ensure we have at least some text
    if not cleaned_text:
        cleaned_text = "empty_email"
    
    return cleaned_text, features

# ==================== BATCH PROCESSING FUNCTION ====================
def preprocess_batch(raw_emails: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a batch of raw emails for training or batch prediction.
    
    Args:
        raw_emails (List[str]): List of raw email strings.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - text_df: DataFrame with 'cleaned_text' column for TF-IDF.
            - features_df: DataFrame with all extracted numerical features.
    """
    cleaned_texts = []
    features_list = []
    
    for email in raw_emails:
        cleaned_text, features = preprocess_email(email)
        cleaned_texts.append(cleaned_text)
        features_list.append(features)
    
    # Create DataFrames with consistent column order
    text_df = pd.DataFrame({'cleaned_text': cleaned_texts})
    features_df = pd.DataFrame(features_list)
    
    # Ensure all expected feature columns exist (fill missing with 0)
    expected_features = [
        'num_links', 'has_url', 'has_urgent_keyword', 'num_exclamations',
        'num_questions', 'has_html', 'all_caps_ratio', 'body_length',
        'num_recipients', 'has_attachment_keyword', 'subject_length',
        'subject_has_urgent'
    ]
    
    for feat in expected_features:
        if feat not in features_df.columns:
            features_df[feat] = 0.0
    
    # Reorder columns for consistency
    features_df = features_df[expected_features]
    
    return text_df, features_df

# ==================== INTEGRATION WITH PIPELINE ====================
def create_preprocessor():
    """
    Creates a scikit-learn compatible transformer for the preprocessing pipeline.
    This is what you'll use in your ColumnTransformer.
    """
    from sklearn.preprocessing import FunctionTransformer
    
    def transform_func(X):
        """X is expected to be a pandas Series or array of raw email strings."""
        text_df, features_df = preprocess_batch(X.tolist() if hasattr(X, 'tolist') else X)
        # Combine for ColumnTransformer
        result_df = pd.concat([text_df, features_df], axis=1)
        return result_df
    
    return FunctionTransformer(transform_func, validate=False)

# ==================== TEST THE FUNCTION ====================
if __name__ == "__main__":
    # Test with a sample phishing email
    test_email = """From: "Bank Security" <security@your-bank.com>
Subject: URGENT: Account Suspension Notice!
To: customer@gmail.com
Content-Type: text/html
MIME-Version: 1.0

<html>
<head><title>IMPORTANT</title></head>
<body>
<h1>SECURITY ALERT</h1>
<p>Dear Customer,</p>
<p>We detected UNUSUAL activity on your account. Your account will be SUSPENDED unless you verify your identity IMMEDIATELY.</p>
<p>Click here to verify: <a href="https://phishing-site.com/verify?user=123">Verify Account</a></p>
<p>This is URGENT! You have 24 HOURS to respond!</p>
<p>Thank you,<br>Security Team</p>
</body>
</html>"""
    
    print("🧪 Testing preprocessing function...")
    cleaned, features = preprocess_email(test_email)
    
    print("\n✅ Cleaned Text (first 200 chars):")
    print(f"{cleaned[:200]}...")
    
    print("\n✅ Extracted Features:")
    for key, value in features.items():
        print(f"  {key:25}: {value}")
    
    print(f"\n✅ Total features extracted: {len(features)}")
