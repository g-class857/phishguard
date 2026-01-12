#!/usr/bin/env python3
"""
PhishGuard – Email Preprocessing Engine
Parallel, Deduplicated, Continuous Learning Safe
"""

import os
import re
import time
import math
import hashlib
import logging
import email
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import Dict, List

import pandas as pd
from bs4 import BeautifulSoup
from urlextract import URLExtract
from urllib.parse import urlparse

# =============================
# CONFIG
# =============================
RAW_DATA_PATH = r"C:\Users\hassan\Desktop\phishing_detection_system\data\raw"
PROCESSED_DATA_PATH = r"C:\Users\hassan\Desktop\phishing_detection_system\data\processed"
OUTPUT_FILE = "phishguard_features.csv"

MAX_WORKERS = max(1, cpu_count() - 1)
URL_EXTRACTOR = URLExtract()

URGENT_WORDS = {
    "urgent", "verify", "confirm", "action required",
    "password", "login", "invoice", "payment", "suspend"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# =============================
# UTILITIES
# =============================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum((c / len(s)) * math.log2(c / len(s)) for c in counts.values())

def compute_hash(subject: str, body: str, sender: str) -> str:
    base = f"{subject}|{body}|{sender}"
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()

# =============================
# EML PARSING
# =============================
def parse_eml(path: str) -> Dict:
    try:
        raw = open(path, "rb").read()
        msg = email.message_from_bytes(raw)
    except Exception:
        return None

    subject = normalize(msg.get("Subject", ""))
    sender = normalize(msg.get("From", ""))
    body = ""
    html_present = 0
    attachments = []

    for part in msg.walk():
        ctype = part.get_content_type()
        disp = str(part.get_content_disposition() or "")

        if ctype == "text/plain" and "attachment" not in disp:
            body += part.get_payload(decode=True).decode(errors="ignore")
        elif ctype == "text/html":
            html_present = 1
            soup = BeautifulSoup(
                part.get_payload(decode=True).decode(errors="ignore"),
                "html.parser"
            )
            body += soup.get_text(" ")

        if part.get_filename():
            attachments.append(part.get_filename())

    body = normalize(body)
    urls = URL_EXTRACTOR.find_urls(body)

    return build_features(
        subject, body, sender, urls, html_present, attachments
    )

# =============================
# CSV PARSING (SAFE AUTO-DETECT)
# =============================
def parse_csv_row(row: Dict) -> Dict:
    text_columns = {
        k: normalize(v) for k, v in row.items()
        if isinstance(v, str) and len(normalize(v)) > 0
    }

    if not text_columns:
        return None

    sorted_cols = sorted(
        text_columns.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    body = sorted_cols[0][1]
    subject = sorted_cols[1][1] if len(sorted_cols) > 1 else ""
    sender = row.get("from", row.get("sender", ""))

    urls = URL_EXTRACTOR.find_urls(body)

    return build_features(
        subject, body, sender, urls, 0, []
    )

# =============================
# FEATURE ENGINEERING
# =============================
def build_features(subject, body, sender, urls, html_present, attachments):
    return {
        "subject": subject,
        "body": body,
        "sender": sender,
        "num_urls": len(urls),
        "num_domains": len(set(urlparse(u).netloc for u in urls)),
        "ip_urls": sum(1 for u in urls if re.match(r"https?://\d+\.", u)),
        "urgent_words": sum(w in body.lower() for w in URGENT_WORDS),
        "exclamation_count": body.count("!"),
        "digit_ratio": sum(c.isdigit() for c in body) / max(len(body), 1),
        "body_entropy": shannon_entropy(body),
        "html_present": html_present,
        "attachment_names": ";".join(attachments)
    }

# =============================
# DATA LOADING
# =============================
def process_emls(files: List[str]) -> List[Dict]:
    with Pool(MAX_WORKERS) as pool:
        results = pool.map(parse_eml, files)
    return [r for r in results if r]

def load_raw_data() -> pd.DataFrame:
    records = []
    eml_files, csv_files = [], []

    for root, _, files in os.walk(RAW_DATA_PATH):
        for f in files:
            full = os.path.join(root, f)
            if f.lower().endswith(".eml"):
                eml_files.append(full)
            elif f.lower().endswith(".csv"):
                csv_files.append(full)

    logging.info("Found %d EML files, %d CSV files", len(eml_files), len(csv_files))

    if eml_files:
        records.extend(process_emls(eml_files))

    for csv in csv_files:
        logging.info("Reading CSV: %s", csv)
        df = pd.read_csv(csv, dtype=str).fillna("")
        for _, row in df.iterrows():
            rec = parse_csv_row(row.to_dict())
            if rec:
                records.append(rec)

    return pd.DataFrame(records)

# =============================
# MAIN PIPELINE
# =============================
def main():
    start = time.time()
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, OUTPUT_FILE)

    old_df = pd.read_csv(output_path).fillna("") if os.path.exists(output_path) else pd.DataFrame()
    new_df = load_raw_data()

    logging.info("New records extracted: %d", len(new_df))

    combined = pd.concat([old_df, new_df], ignore_index=True)

    if combined.empty:
        logging.warning("No data loaded. Check raw dataset path.")
        return

    combined["_tmp_hash"] = combined.apply(
        lambda r: compute_hash(r["subject"], r["body"], r["sender"]),
        axis=1
    )

    before = len(combined)
    combined.drop_duplicates("_tmp_hash", inplace=True)
    after = len(combined)

    combined.drop(columns="_tmp_hash", inplace=True)

    combined.to_csv(output_path, index=False)

    logging.info("Dedup removed %d duplicates", before - after)
    logging.info("Final dataset size: %d", after)
    logging.info("Finished in %.2fs", time.time() - start)

if __name__ == "__main__":
    main()
