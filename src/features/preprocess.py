#!/usr/bin/env python3
"""
PhishGuard – Email Preprocessing Engine
Refactored, Unicode-safe, multiprocessing-friendly.

Features:
- Loads .eml and .csv from RAW_DATA_PATH
- Robust CSV reading (utf-8 fallback to latin-1; skip bad rows)
- Unicode-safe URL extraction (avoids urlextract UnicodeDecodeError)
- Per-process lazy URLExtract initialization (safe with multiprocessing.Pool)
- Stores urls and domains in output (semicolon-separated)
- Normalizes spam/ham labels into `label` column (1 = phishing/spam, 0 = ham/legit, -1 = unknown)
- Deduplicates output and saves CSV
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
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from urlextract import URLExtract
from urllib.parse import urlparse

# =============================
# CONFIG - adjust paths if needed
# =============================
RAW_DATA_PATH = r"C:\Users\hassan\Desktop\phishing_detection_system\data\raw"
PROCESSED_DATA_PATH = r"C:\Users\hassan\Desktop\phishing_detection_system\data\processed"
OUTPUT_FILE = "phishguard_features.csv"

MAX_WORKERS = max(1, cpu_count() - 1)

# Label mapping: normalized strings -> numeric label
LABEL_MAP = {
    "spam": 1,
    "phish": 1,
    "phishing": 1,
    "1": 1,
    "yes": 1,
    "true": 1,
    "malicious": 1,
    "ham": 0,
    "legit": 0,
    "legitimate": 0,
    "0": 0,
    "no": 0,
    "false": 0
}

URGENT_WORDS = {
    "urgent", "verify", "confirm", "action required",
    "password", "login", "invoice", "payment", "suspend"
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# =============================
# Globals / per-process cache
# =============================
# We will lazily initialize a URLExtract instance per process to avoid pickling issues.
_process_url_extractor = None

def get_url_extractor() -> URLExtract:
    """
    Lazily create and return a URLExtract instance for the current process.
    This avoids pickling URLExtract across processes and allows us to silence its logger.
    """
    global _process_url_extractor
    if _process_url_extractor is None:
        _process_url_extractor = URLExtract()
        # silence internal logging/noise
        logging.getLogger("urlextract").setLevel(logging.CRITICAL)
    return _process_url_extractor

# =============================
# UTILITIES
# =============================
def normalize(text: Optional[str]) -> str:
    """Safe whitespace normalization"""
    return re.sub(r"\s+", " ", str(text or "")).strip()

def shannon_entropy(s: str) -> float:
    """Shannon entropy of a string (character-level)"""
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum((c / len(s)) * math.log2(c / len(s)) for c in counts.values())

def compute_hash(subject: str, body: str, sender: str) -> str:
    """Stable SHA-256 hash for deduplication"""
    base = f"{subject}|{body}|{sender}"
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()

def normalize_label(value) -> int:
    """Map a raw label value to 1/0/-1 (1 phishing/spam, 0 ham/legit, -1 unknown)"""
    if value is None:
        return -1
    raw = str(value).strip().lower()
    return LABEL_MAP.get(raw, -1)

def safe_text_for_extraction(text: str) -> str:
    """
    Make a best-effort safe unicode string for downstream libraries.
    We drop invalid bytes and return a UTF-8 string.
    """
    if not text:
        return ""
    # First try to ensure it's a str; if already str, encode/decode to drop invalid surrogates
    try:
        # If it's bytes disguised as str (rare), ensure bytes decode
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        # otherwise force-encode/decode to drop invalid sequences
        return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return str(text).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

def safe_find_urls(text: str) -> List[str]:
    """
    Safely extract URLs from possibly malformed or mixed-encoding text.
    Uses per-process URLExtract instance.
    """
    if not text:
        return []
    safe_text = safe_text_for_extraction(text)
    try:
        extractor = get_url_extractor()
        urls = extractor.find_urls(safe_text)
        return urls if urls else []
    except Exception as e:
        # Do not crash ingestion; log at debug level for later inspection
        logging.debug("safe_find_urls failed: %s", str(e)[:200])
        return []

# =============================
# FEATURE ENGINE
# =============================
def build_features(subject: str,
                   body: str,
                   sender: str,
                   urls: List[str],
                   html_present: int,
                   attachments: List[str],
                   label: int) -> Dict:
    """Return a feature dict for a single email"""
    domains = [urlparse(u).netloc for u in urls if urlparse(u).netloc]
    # limit lengths to avoid super long CSV fields (truncate long url lists/domains)
    urls_field = ";".join(urls)[:32000]  # keep under typical CSV field limits
    domains_field = ";".join(dict.fromkeys(domains))[:32000]

    word_body = body or ""
    word_count = len(word_body.split())

    return {
        "subject": subject,
        "body": body,
        "sender": sender,

        # URL intelligence (stored, not printed)
        "urls": urls_field,
        "domains": domains_field,

        # Numeric features
        "num_urls": len(urls),
        "num_domains": len(set(domains)),
        "ip_urls": sum(1 for u in urls if re.match(r"https?://\d+\.", u)),
        "urgent_words": sum(w in (word_body or "").lower() for w in URGENT_WORDS),
        "exclamation_count": (word_body or "").count("!"),
        "digit_ratio": sum(c.isdigit() for c in (word_body or "")) / max(len(word_body or ""), 1),
        "body_entropy": shannon_entropy(word_body),
        "html_present": int(bool(html_present)),
        "attachment_names": ";".join(attachments) if attachments else "",
        # Label: 1 = phishing/spam, 0 = ham/legit, -1 unknown
        "label": int(label)
    }

# =============================
# EML parsing (worker-safe)
# =============================
def parse_eml(path: str) -> Optional[Dict]:
    """Parse an .eml file and extract features; label = -1 (unknown)"""
    try:
        raw = open(path, "rb").read()
        msg = email.message_from_bytes(raw)
    except Exception as e:
        logging.debug("Failed to read eml %s: %s", path, e)
        return None

    subject = normalize(msg.get("Subject", ""))
    sender = normalize(msg.get("From", ""))
    body_parts = []
    html_present = 0
    attachments = []

    for part in msg.walk():
        try:
            ctype = part.get_content_type()
            disp = str(part.get_content_disposition() or "")
            if ctype == "text/plain" and "attachment" not in disp:
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode("utf-8", errors="ignore"))
            elif ctype == "text/html":
                html_present = 1
                payload = part.get_payload(decode=True)
                if payload:
                    soup = BeautifulSoup(payload.decode("utf-8", errors="ignore"), "html.parser")
                    body_parts.append(soup.get_text(" ", strip=False))
            if part.get_filename():
                attachments.append(part.get_filename())
        except Exception:
            # per-part failure should not break processing
            continue

    body = normalize(" ".join(body_parts))
    urls = safe_find_urls(body)
    return build_features(subject, body, sender, urls, html_present, attachments, label=-1)

# =============================
# CSV row parsing (label-aware)
# =============================
def parse_csv_row(row: Dict) -> Optional[Dict]:
    """
    Parse a CSV row dict. Try to find meaningful text columns (body, subject),
    normalize label if present, and extract safe URLs.
    """
    # Normalize label from known columns
    raw_label = row.get("phish") or row.get("label") or row.get("class") or row.get("spam")
    label = normalize_label(raw_label)

    # Collect candidate text columns (prefer longer text for body)
    text_columns = {
        k: normalize(v)
        for k, v in row.items()
        if isinstance(v, str) and normalize(v)
    }

    if not text_columns:
        return None

    # Sort by length to pick body then subject
    sorted_cols = sorted(text_columns.items(), key=lambda x: len(x[1]), reverse=True)
    body = sorted_cols[0][1]
    subject = sorted_cols[1][1] if len(sorted_cols) > 1 else ""
    sender = normalize(row.get("from") or row.get("sender") or "")

    # Safe URL extraction (avoids UnicodeDecodeError)
    urls = safe_find_urls(body)

    return build_features(subject, body, sender, urls, html_present=0, attachments=[], label=label)

# =============================
# DATA LOADING
# =============================
def process_emls(files: List[str]) -> List[Dict]:
    """Process EML files in parallel using Pool"""
    if not files:
        return []
    # Use multiprocessing Pool; parse_eml is top-level so picklable
    with Pool(MAX_WORKERS) as pool:
        results = pool.map(parse_eml, files)
    return [r for r in results if r]

def _read_csv_tolerant(path: Path) -> Optional[pd.DataFrame]:
    """
    Read CSV with tolerance: try UTF-8, fall back to latin-1, skip bad lines.
    Returns DataFrame or None if unrecoverable.
    """
    try:
        return pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip").fillna("")
    except Exception as e1:
        logging.debug("utf-8 read failed for %s: %s", path, e1)
        try:
            return pd.read_csv(path, dtype=str, engine="python", encoding="latin-1", on_bad_lines="skip").fillna("")
        except Exception as e2:
            logging.warning("Skipping CSV %s due to read errors (%s / %s)", path, e1, e2)
            return None

def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """
    Walk RAW_DATA_PATH, find .eml and .csv, parse them, and return a DataFrame of feature dicts.
    """
    records = []
    eml_files = []
    csv_files = []

    for root, _, files in os.walk(raw_dir):
        for f in files:
            full = os.path.join(root, f)
            if f.lower().endswith(".eml"):
                eml_files.append(full)
            elif f.lower().endswith(".csv"):
                csv_files.append(full)

    logging.info("Found %d EML files, %d CSV files", len(eml_files), len(csv_files))

    # Process EMLs in parallel (label = -1)
    if eml_files:
        logging.info("Processing EML files (parallel)...")
        records.extend(process_emls(eml_files))

    # Process CSVs one by one (safer for mixed-encoding corpora)
    for csv_path in csv_files:
        p = Path(csv_path)
        logging.info("Reading CSV: %s", csv_path)
        df = _read_csv_tolerant(p)
        if df is None:
            continue

        # Parse rows
        for _, row in df.iterrows():
            try:
                rec = parse_csv_row(row.to_dict())
                if rec:
                    records.append(rec)
            except Exception as e:
                logging.debug("Row parse failed (%s): %s", p.name, e)
                continue

    result_df = pd.DataFrame(records)
    # If no records found, return empty DF with columns expected (optional)
    return result_df

# =============================
# MAIN
# =============================
def main():
    start = time.time()
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, OUTPUT_FILE)

    # Load existing processed file if exists, else empty DF
    old_df = pd.read_csv(output_path, dtype=str).fillna("") if os.path.exists(output_path) else pd.DataFrame()

    # Load and parse raw data
    new_df = load_raw_data(RAW_DATA_PATH)

    logging.info("New records extracted: %d", len(new_df))

    # Ensure label column exists (if missing, fill with -1)
    if "label" not in new_df.columns:
        new_df["label"] = -1
    # Ensure columns in old_df if present
    if not old_df.empty and "label" not in old_df.columns:
        old_df["label"] = -1

    # Combine and deduplicate
    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)

    if combined.empty:
        logging.warning("No data loaded.")
        return

    # Compute dedupe hash (handle missing columns safely)
    combined["_hash"] = combined.apply(
        lambda r: compute_hash(str(r.get("subject", "")),
                               str(r.get("body", "")),
                               str(r.get("sender", ""))),
        axis=1
    )

    before = len(combined)
    combined.drop_duplicates("_hash", inplace=True)
    combined.drop(columns=["_hash"], inplace=True)

    # Persist - ensure label is stored as int
    try:
        combined["label"] = combined["label"].astype(int)
    except Exception:
        combined["label"] = combined["label"].apply(lambda v: int(v) if str(v).isdigit() else -1)

    combined.to_csv(output_path, index=False)
    logging.info("Deduplicated %d records", before - len(combined))
    logging.info("Final dataset size: %d", len(combined))
    logging.info("Finished in %.2fs", time.time() - start)

if __name__ == "__main__":
    main()

