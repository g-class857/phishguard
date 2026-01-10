"""
manual_features.py
SOC-grade manual feature extraction for phishing detection
"""

import re
import pandas as pd
from typing import List, Tuple

# ================= TEXT CLEANING =================
def clean_text(text: str) -> str:
    """Basic safe text cleaning"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ================= SOC FEATURES =================
def extract_features(text: str) -> dict:
    """Extract SOC-relevant phishing indicators"""
    text = text or ""
    words = text.split()

    return {
        "num_links": len(re.findall(r"http[s]?://", text)),
        "has_url": int("http" in text),
        "has_urgent_keyword": int(any(w in text for w in [
            "urgent", "verify", "immediately", "action required", "suspended"
        ])),
        "num_exclamations": text.count("!"),
        "num_questions": text.count("?"),
        "has_html": int(bool(re.search(r"<[^>]+>", text))),
        "all_caps_ratio": sum(1 for w in words if w.isupper()) / (len(words) + 1),
        "body_length": len(words),
        "has_attachment_keyword": int(any(w in text for w in [
            "attachment", "invoice", "pdf", "document"
        ])),
        "credential_request": int(any(w in text for w in [
            "password", "login", "credentials", "verify account"
        ])),
        "financial_keyword": int(any(w in text for w in [
            "bank", "payment", "refund", "wire", "transaction"
        ])),
        "external_sender_hint": int(any(w in text for w in [
            "external", "outside", "unrecognized sender"
        ]))
    }


# ================= BATCH PROCESSING =================
def preprocess_batch(raw_emails: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts raw emails into:
    - cleaned_text (for TF-IDF)
    - numeric SOC features
    """
    cleaned_texts = []
    features = []

    for email in raw_emails:
        cleaned = clean_text(email)
        cleaned_texts.append(cleaned)
        features.append(extract_features(email))

    text_df = pd.DataFrame({"cleaned_text": cleaned_texts})
    features_df = pd.DataFrame(features)

    return text_df, features_df
