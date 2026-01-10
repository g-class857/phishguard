"""
manual_features.py
Core preprocessing and feature extraction for phishing detection.
Used identically in training and production (SOC-grade).
"""

import re
import html
from typing import Tuple, Dict, List
import pandas as pd

# ==================== CORE PREPROCESSING FUNCTION ====================
def preprocess_email(raw_email: str) -> Tuple[str, Dict[str, float]]:
    """
    Transforms a raw email into cleaned text + SOC-grade feature set.
    """

    # === 1. INITIAL CLEANING & DECODING ===
    text = html.unescape(raw_email)
    text = re.sub(r'=\s*\n', '', text)  # Remove soft line breaks
    text = re.sub(r'=([A-F0-9]{2})', lambda m: chr(int(m.group(1), 16)), text)

    features: Dict[str, float] = {}

    # === 2. URL & LINK FEATURES ===
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    urls = re.findall(url_pattern, text, re.IGNORECASE)

    features['num_links'] = len(urls)
    features['has_url'] = 1.0 if urls else 0.0

    # Shortened URLs (SOC-grade)
    features['has_short_url'] = 1.0 if re.search(
        r'\b(bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly)\b',
        text, re.IGNORECASE
    ) else 0.0

    # === 3. URGENCY & SOCIAL ENGINEERING ===
    urgent_keywords = [
        'urgent', 'immediate', 'asap', 'action required',
        'verify', 'suspend', 'account', 'security', 'alert'
    ]
    urgent_pattern = r'\b(' + '|'.join(urgent_keywords) + r')\b'

    features['has_urgent_keyword'] = 1.0 if re.search(
        urgent_pattern, text, re.IGNORECASE
    ) else 0.0

    # Imperative verbs (SOC signal)
    features['num_imperative_verbs'] = len(re.findall(
        r'\b(click|verify|confirm|update|login|reset|open|download)\b',
        text, re.IGNORECASE
    ))

    features['num_exclamations'] = text.count('!')
    features['num_questions'] = text.count('?')

    # === 4. FORMATTING & VISUAL TRICKS ===
    features['has_html'] = 1.0 if re.search(r'<[^>]+>', text) else 0.0

    words = re.findall(r'\b[A-Za-z]+\b', text)
    all_caps_words = re.findall(r'\b[A-Z]{2,}\b', text)

    features['all_caps_ratio'] = (
        len(all_caps_words) / max(len(words), 1)
    )

    # === 5. STRUCTURAL FEATURES ===
    features['body_length'] = len(text)

    features['num_recipients'] = len(re.findall(
        r'\b(To:|CC:|BCC:)\b', text, re.IGNORECASE
    ))

    features['has_attachment_keyword'] = 1.0 if re.search(
        r'\b(attachment|attached|enclosed)\b',
        text, re.IGNORECASE
    ) else 0.0

    # === 6. HEADER / BODY SPLIT ===
    parts = re.split(r'\n\s*\n', text, maxsplit=1)
    headers = parts[0] if parts else ''
    body = parts[1] if len(parts) > 1 else headers

    # Subject analysis
    subject_match = re.search(
        r'Subject:\s*(.*?)(?:\n|$)',
        headers,
        re.IGNORECASE | re.DOTALL
    )

    subject = subject_match.group(1).strip() if subject_match else ''
    features['subject_length'] = len(subject)
    features['subject_has_urgent'] = 1.0 if re.search(
        urgent_pattern, subject, re.IGNORECASE
    ) else 0.0

    # === 7. SENDER DOMAIN MISMATCH (SOC FEATURE) ===
    from_match = re.search(r'From:.*?@([\w\.-]+)', headers, re.IGNORECASE)
    sender_domain = from_match.group(1) if from_match else ''

    link_domains = re.findall(
        r'https?://([\w\.-]+)', text, re.IGNORECASE
    )

    features['domain_mismatch'] = 1.0 if (
        sender_domain and any(
            sender_domain not in d for d in link_domains
        )
    ) else 0.0

    # === 8. LINK TEXT VS URL MISMATCH ===
    features['link_mismatch'] = 1.0 if re.search(
        r'<a\s+href="([^"]+)">([^<]+)</a>',
        text, re.IGNORECASE
    ) else 0.0

    # === 9. CLEAN BODY FOR TF-IDF ===
    body = re.sub(r'<script.*?</script>', ' ', body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r'<style.*?</style>', ' ', body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r'<[^>]+>', ' ', body)

    body = re.sub(url_pattern, ' URL_TOKEN ', body, flags=re.IGNORECASE)
    body = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' EMAIL_TOKEN ', body)
    body = re.sub(r'\$\d+(?:\.\d{1,2})?', ' MONEY_TOKEN ', body)
    body = re.sub(r'\b\d{10,}\b', ' NUMBER_TOKEN ', body)

    body = re.sub(r'[^\w\s\.\,\!\?]', ' ', body)
    body = ' '.join(body.split()).lower().strip()

    cleaned_text = body if body else "empty_email"

    return cleaned_text, features

# ==================== BATCH PROCESSING ====================
def preprocess_batch(raw_emails: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_texts = []
    features_list = []

    for email in raw_emails:
        text, feats = preprocess_email(email)
        cleaned_texts.append(text)
        features_list.append(feats)

    text_df = pd.DataFrame({'cleaned_text': cleaned_texts})
    features_df = pd.DataFrame(features_list)

    # Expected SOC-grade feature set
    expected_features = [
        'num_links', 'has_url', 'has_short_url',
        'has_urgent_keyword', 'num_imperative_verbs',
        'num_exclamations', 'num_questions',
        'has_html', 'all_caps_ratio',
        'body_length', 'num_recipients',
        'has_attachment_keyword',
        'subject_length', 'subject_has_urgent',
        'domain_mismatch', 'link_mismatch'
    ]

    for feat in expected_features:
        if feat not in features_df.columns:
            features_df[feat] = 0.0

    features_df = features_df[expected_features]

    return text_df, features_df
