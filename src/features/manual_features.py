"""
Core functions for preprocessing raw emails and extracting features used identically during training and in production.
"""

import re  # Regular expressions for pattern matching 
import html  # For handling HTML entities 
from typing import Tuple, Dict, List  # Type hints for function signatures
import pandas as pd  # For batch processing and DataFrame creation

# ************* CORE PREPROCESSING FUNCTION ====================
def preprocess_email(raw_email: str) -> Tuple[str, Dict[str, float]]:
    """
    transforms single raw email string into clean text and a dictionary of features, will be called for every email in production.
    args:
        raw_email (str): raw email , including headers and body.
    
    return:
        Tuple[str, Dict]: 
            - cleaned_text (str): the normalized, cleaned body text for TF-IDF.
            - features (Dict[str, float]):   dictionary of extracted structural features.
    """
    
    #  1. initial cleaning and decoding
    # handle html entities and common encoding artifacts
    text = html.unescape(raw_email) # convert html entities to characters (&amp; → &, &lt; → <)
    text = re.sub(r'=\s*\n', '', text)  #   remove  line breaks (common in email encoding)
    text = re.sub(r'=([A-F0-9]{2})', lambda m: chr(int(m.group(1), 16)), text)  # decode hex char to integers then to chars, lambda:  anonymous function and m is a parameter. chr(int("41", 16), text) -> A
    
    #  2. EXTRACT  FEATURES (BEFORE MODIFYING TEXT) 
    #    crucial for phishing detection 
    features = {}
    
    # link features
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+' # matches http/https URLs and www addresses
    urls = re.findall(url_pattern, text, re.IGNORECASE) 
    features['num_links'] = len(urls) # phishing links usually has long characters ( malware often use domain generation algorithms )
    features['has_url'] = 1.0 if features['num_links'] > 0 else 0.0 # does the email has a link or not
    
    # urgency  features: detects common psychological pressure tactics in phishing
    urgent_keywords = ['urgent', 'immediate', 'asap', 'action required', 'verify', 'suspend', 'account', 'security', 'alert'] 
    urgent_pattern = r'\b(' + '|'.join(urgent_keywords) + r')\b' # \b: Word boundary (ensures whole word matching) 
    features['has_urgent_keyword'] = 1.0 if re.search(urgent_pattern, text, re.IGNORECASE) else 0.0 #  if any urgent word appears in text (case-insensitive)

    features['num_exclamations'] = text.count('!')
    features['num_questions'] = text.count('?')
    
    # formatting features (common in phishing) 
    #   pattern: < followed by non-> characters, then > .detects html formatting (phishing emails often use HTML for legitimacy)
    features['has_html'] = 1.0 if bool(re.search(r'<[^>]+>', text)) else 0.0
    features['all_caps_ratio'] = len(re.findall(r'\b[A-Z]{2,}\b', text)) / max(len(re.findall(r'\b[A-Za-z]+\b', text)), 1) # words with 2+ uppercase letters common in phishing
    
    #  features
    features['body_length'] = len(text)
    features['num_recipients'] = len(re.findall(r'To:|CC:|BCC:', text, re.IGNORECASE)) # counts recipient fields (mass emails might indicate spam)
    features['has_attachment_keyword'] = 1.0 if re.search(r'\b(attachment|attached|enclosed)\b', text, re.IGNORECASE) else 0.0   #  detects mention of attachments
    
    # === 3. SEPARATE HEADERS FROM BODY ===
    # Find the first empty line (standard email header-body separator)
    parts = re.split(r'\n\s*\n', text, maxsplit=1) # maxsplit=1 Only split on first occurrence
    headers = parts[0] if len(parts) > 0 else ''
    body = parts[1] if len(parts) > 1 else headers #Handles edge cases: If no blank line, treat everything as body
    
    # Extract subject for additional features: capture until newline or end (?:\n|$) , 
    subject_match = re.search(r'Subject:\s*(.*?)(?:\n|$)', headers, re.IGNORECASE | re.DOTALL) # re.DOTALL make . matches newline
    subject = subject_match.group(1).strip() if subject_match else '' # search for subject and read after it, strip white spaces till the end or new line
    features['subject_length'] = len(subject)
    features['subject_has_urgent'] = 1.0 if re.search(urgent_pattern, subject, re.IGNORECASE) else 0.0
    
    # === 4. CLEAN BODY TEXT FOR TF-IDF ANALYSIS ===
    # Remove HTML tags but preserve their text content
    # Remove JavaScript, CSS and style (malicious or irrelevant for text analysis)
    body = re.sub(r'<script.*?</script>', ' ', body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r'<style.*?</style>', ' ', body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r'<[^>]+>', ' ', body) # Remove remaining HTML tags but keep text content
    
    # Replace specific patterns with tokens (helps model learn patterns without memorizing specific ones)
    body = re.sub(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', ' URL_TOKEN ', body, flags=re.IGNORECASE) # URL pattern
    body = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' EMAIL_TOKEN ', body) # email pattern
    body = re.sub(r'\$\d+(?:\.\d{1,2})?|\d+\s*(?:dollars?|usd|eur|gbp)\b', ' MONEY_TOKEN ', body, flags=re.IGNORECASE) # money pattern
    body = re.sub(r'\b\d{10,}\b', ' NUMBER_TOKEN ', body)  # Long numbers (like phone/account)
    
    # Remove remaining special characters but keep basic senence structure
    body = re.sub(r'[^\w\s\.\,\!\?]', ' ', body)
    
    # Normalization: split on whitespaces and rejoin with single one and convert to lowercase
    body = ' '.join(body.split())
    cleaned_text = body.lower().strip()
    
    # Ensure we have at least some text
    if not cleaned_text:
        cleaned_text = "empty_email"
    
    return cleaned_text, features

# ==================== BATCH PROCESSING FUNCTION ====================
def preprocess_batch(raw_emails: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]: # process multiple emails efficiently, used in training and batch prediction
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
    
    for email in raw_emails: # iterate through emails and collecting results
        cleaned_text, features = preprocess_email(email)
        cleaned_texts.append(cleaned_text)
        features_list.append(features)
    
    # Create DataFrames with consistent column order. DataFrames are Compatible with scikit-learn and pandas pipelines
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
    #  Ensures consistent feature set across all emails, Some emails might not have certain features (e.g., no subject), but ML models expect same columns
# ==================== INTEGRATION WITH PIPELINE ====================
def create_preprocessor():
    """
    Creates a scikit-learn compatible transformer for the preprocessing pipeline.
    This is what will be used in ColumnTransformer  with different processing for text vs numerical features.
    FunctionTransformer: Wraps custom function into sklearn transformer
    
    """
    from sklearn.preprocessing import FunctionTransformer
    
    def transform_func(X):
        """X is expected to be a pandas Series or array of raw email strings."""
        text_df, features_df = preprocess_batch(X.tolist() if hasattr(X, 'tolist') else X)
        # Combine for ColumnTransformer
        result_df = pd.concat([text_df, features_df], axis=1)
        return result_df
    
    return FunctionTransformer(transform_func, validate=False) # validate=False: Skips sklearn input validation (we handle our own)

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
