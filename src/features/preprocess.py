#!/usr/bin/env python3
"""
PhishGuard – Email Preprocessing Engine
Refactored: header extraction + removed specified fields.
Features:
- Chunked CSV ingestion (memory-safe)
- Unicode-safe URL extraction
- Extracts authentication headers (SPF/DKIM/DMARC)
- Extracts additional headers: from, recipient, return-path, to, message-id,
  x-mailing (x_mailer), x-originating-ip, content-type
- Keeps urls/domains/ip_urls, urgent_words, digit_ratio, body_entropy, html_present,
  attachment_names, auth features, label, subject, body, sender
- Deduplication and output CSV
"""

import os   # walk directories, handle paths
import re   # regex for parsing headers and urls
import time # measure pipeline runtime
import math # shanon entropy calculations
import hashlib  # use hashing for deduplicate
import logging  
import email    
from multiprocessing import Pool, cpu_count # paralel processing to improve performance
from collections import Counter #entropy calculations 
from typing import Dict, List, Optional, Iterable   # makes code safe, readable and mainrainable
from pathlib import Path

import pandas as pd # for data analysis and manipulation
from bs4 import BeautifulSoup   # strip html safely
from urlextract import URLExtract   # robust url detection
from urllib.parse import urlparse   # url extraction

import csv
import sys

import re
from email.utils import parseaddr   #safely parse email addresses from header text.

# Allow very large email bodies in CSVs to be parsed to prevent crashing 
_max_int = sys.maxsize
while True: # try to allow large csv fields if python crashes, reduce size and try again (defensive programming)
    try:
        csv.field_size_limit(_max_int)
        break
    except OverflowError:
        _max_int = int(_max_int / 10)

# =============================
# CONFIG
# =============================
RAW_DATA_PATH = r"C:\Users\hassan\Desktop\phishing_detection_system\data\raw"
PROCESSED_DATA_PATH = r"C:\Users\hassan\Desktop\phishing_detection_system\data\processed"
OUTPUT_FILE = "phishguard_features.csv"

MAX_WORKERS = max(1, cpu_count() - 1)   # use CPUs except one
CSV_CHUNKSIZE = 5000  # read 5000 rows at a time. memory-safe ingestion (don't load entire file to memory)

# silence noisy logger from urlextract
logging.getLogger("urlextract").setLevel(logging.CRITICAL)

LABEL_MAP = {
    "spam": 1, "phish": 1, "phishing": 1, "1": 1, "yes": 1, "true": 1, "malicious": 1,
    "ham": 0, "legit": 0, "legitimate": 0, "0": 0, "no": 0, "false": 0
}

URGENT_WORDS = {
    "urgent", "verify", "confirm", "action required",
    "password", "login", "invoice", "payment", "suspend"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =============================
# UTILITIES
# =============================
def normalize(text: Optional[str]) -> str: # takes string or none as input
    return re.sub(r"\s+", " ", str(text or "")).strip() # removes extra whitespaces, converts none safely and prevents feature noise. strip() removes leading and trailing spaces

# Phishing emails often contain: Obfuscated text, Random strings, Higher entropy → more suspicious. Used in real malware detection.
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum((c / len(s)) * math.log2(c / len(s)) for c in counts.values())

# use sha-256 to prevent dataset poisoning by duplicates
def compute_hash(subject: str, body: str, sender: str) -> str:
    base = f"{subject}|{body}|{sender}"
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()


def normalize_label(value) -> int:
    if value is None:
        return -1
    raw = str(value).strip().lower()
    return LABEL_MAP.get(raw, -1)

# =============================
# SAFE URL EXTRACTION (global extractor)
# =============================
_URL_EXTRACTOR = URLExtract()

# emails contain broken unicode that libraries can crash on bad input. this function sanitize text, fails gracefully and never stops the pipeline
def safe_find_urls(text: str) -> List[str]:
    if not text:
        return []
    try:
        safe_text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        return _URL_EXTRACTOR.find_urls(safe_text) or []
    except Exception:
        logging.debug("safe_find_urls encountered exception", exc_info=True)
        return []
""" emails often contains: 
Broken UTF-8 characters
Mixed encodings
Invisible control bytes
Malware obfuscation tricks

encode then decode to remove garbage bytes and urls still detectable """
# =============================
# AUTH HEADER PARSING (SPF/DKIM/DMARC heuristics)
# =============================
_SPFFLAGS = {"pass": 1, "fail": 0, "softfail": 0, "neutral": -1, "none": -1}
_DKIMFLAGS = {"pass": 1, "fail": 0, "neutral": -1, "none": -1}
_DMARCFLAGS = {"pass": 1, "fail": 0, "none": -1, "quarantine": 0, "reject": 0}

_SPF_RE = re.compile(r'\bspf=([a-zA-Z0-9_-]+)\b', flags=re.IGNORECASE)  # compile() is like pre-built machine stored in memory and reused
_DKIM_RE = re.compile(r'\bdkim=([a-zA-Z0-9_-]+)\b', flags=re.IGNORECASE)
_DMARC_RE = re.compile(r'\bdmarc=([a-zA-Z0-9_-]+)\b', flags=re.IGNORECASE)

def _extract_flag(pattern: re.Pattern, text: str, mapping: Dict[str,int], default: int = -1) -> int:
    if not text:
        return default
    m = pattern.search(text)
    if not m:
        return default
    val = m.group(1).lower()
    return mapping.get(val, default)

def parse_auth_from_headers(headers_text: str) -> Dict[str, object]:
    result = {
        "auth_headers_present": False,
        "spf_result": -1,   #spf validation
        "dkim_result": -1,  #dkim signature 
        "dmarc_result": -1, # dmarc policy
        "return_path_domain": "", 
        "received_count": 0 # mail hop count
    }
    if not headers_text:
        return result

    lower = headers_text.lower()
    result["auth_headers_present"] = any(k in lower for k in ("authentication-results:", "received-spf", "dkim-signature", "dmarc=", "authentication_results"))

    result["spf_result"] = _extract_flag(_SPF_RE, headers_text, _SPFFLAGS)
    result["dkim_result"] = _extract_flag(_DKIM_RE, headers_text, _DKIMFLAGS) 
    result["dmarc_result"] = _extract_flag(_DMARC_RE, headers_text, _DMARCFLAGS)


    # ensure headers_text is a string
    headers_text = headers_text or ""

    # 1) Try to extract Return-Path email address and domain (robustly)
    # - Use regex to capture the address-ish part after Return-Path:
    m_rp = re.search(r'(?mi)^Return-Path:\s*<?([^>\r\n]+)>?', headers_text)
    if m_rp:
        rp_raw = m_rp.group(1).strip()
        # parseaddr is safer: returns ("Name", "local@domain")
        rp_addr = parseaddr(rp_raw)[1]
        if "@" in rp_addr:
            result["return_path_domain"] = rp_addr.split("@", 1)[1].lower()
        else:
            # fallback: try to extract domain-looking token from the raw capture
            dom_match = re.search(r'@([A-Za-z0-9\.\-]+)', rp_raw)
            result["return_path_domain"] = dom_match.group(1).lower() if dom_match else ""
        result["return_path_missing"] = 0
    else:
        result["return_path_domain"] = ""
        result["return_path_missing"] = 1

    # 2) Extract Received domains 
    rcv = re.findall(r'(?mi)^Received:.*?from\s+([A-Za-z0-9\.\-]+)', headers_text) # r= raw string, prevent python from interpreting \s, \. as escape character ## mi = flags for multiline & ignore case
    # store as semicolon-separated list (unique, preserve order)
    seen = []
    for d in rcv:
        dl = d.lower()
        if dl not in seen:
            seen.append(dl)
    result["received_domains"] = ";".join(seen)
    # 'first_received_domain' meaning: earliest hop (last in list) — choose intentionally
    result["first_received_domain"] = seen[-1] if seen else ""

    # 3) Count Received header occurrences
    result["received_count"] = len(re.findall(r'(?mi)^\s*Received:', headers_text))


# =============================
# FEATURE ENGINE 
# =============================
def build_features(subject: str,
                   body: str,
                   sender: str,
                   urls: List[str],
                   html_present: int,
                   attachments: List[str],
                   label: int,
                   auth_info: Optional[Dict[str, object]] = None,
                   header_fields: Optional[Dict[str, str]] = None) -> Dict:
    domains = [urlparse(u).netloc for u in urls if urlparse(u).netloc] # urlparse() splits url into parts. netloc: get domain name
    urls_field = ";".join(urls)[:32000]# join, split with ; and truncate the result to 32000 char
    domains_field = ";".join(dict.fromkeys(domains))[:32000] # fromkeys() removes duplicate while preserving order. not use set()

    auth_info = auth_info or {}
    header_fields = header_fields or {}

    # Keep existing features
    return {
        "subject": subject,
        "body": body,
        "sender": sender,

        # header fields requested
        "from_header": header_fields.get("from_header", ""),
        "recipient": header_fields.get("recipient", ""),
        "return_path": header_fields.get("return_path", ""),
        "to_header": header_fields.get("to_header", ""),
        "message_id": header_fields.get("message_id", ""),
        "x_mailer": header_fields.get("x_mailer", ""),
        "x_originating_ip": header_fields.get("x_originating_ip", ""),
        "content_type": header_fields.get("content_type", ""),

        # URL intelligence (stored)
        "urls": urls_field,
        "domains": domains_field,
        # ip_urls 
        "ip_urls": sum(1 for u in urls if re.match(r"^https?://(?:\d{1,3}\.){3}\d{1,3}\b", u)),

        # retained numeric SOC features
        "urgent_words": sum(w in (body or "").lower() for w in URGENT_WORDS),
        # exclamation_count removed per request
        "digit_ratio": sum(c.isdigit() for c in (body or "")) / max(len(body or ""), 1),
        "body_entropy": shannon_entropy(body or ""),
        "html_present": int(bool(html_present)),
        "attachment_names": ";".join(attachments) if attachments else "",

        # auth features
        "auth_headers_present": int(auth_info.get("auth_headers_present", 0)),
        "spf_result": int(auth_info.get("spf_result", -1)),
        "dkim_result": int(auth_info.get("dkim_result", -1)),
        "dmarc_result": int(auth_info.get("dmarc_result", -1)),
        "return_path_domain": auth_info.get("return_path_domain", ""),
        "received_count": int(auth_info.get("received_count", 0)),

        # label
        "label": int(label)
    }

# =============================
# EML PARSING (now extracts headers)
# =============================
def parse_eml(path: str) -> Optional[Dict]:
    try:
        raw_bytes = open(path, "rb").read() # rb: read binary. files are raw email byte streams, not plain text reading as text ("r") can corrupt encodings
        msg = email.message_from_bytes(raw_bytes)
    except Exception:
        logging.debug("Failed to read eml %s", path, exc_info=True)
        return None

    subject = normalize(msg.get("Subject", ""))
    sender = normalize(msg.get("From", ""))

    body_parts: List[str] = []
    html_present = 0
    attachments: List[str] = []

    for part in msg.walk(): # walk() loops over each part.
        try:
            ctype = part.get_content_type()
            disp = str(part.get_content_disposition() or "")    # used to detect attachments vs. plain text
            payload = part.get_payload(decode=True)
            if payload:
                decoded = payload.decode("utf-8", errors="ignore")
            else:
                decoded = ""
            if ctype == "text/plain" and "attachment" not in disp:
                body_parts.append(decoded)  # append body text to body parts
            elif ctype == "text/html":
                html_present = 1
                body_parts.append(BeautifulSoup(decoded, "html.parser").get_text(" ", strip=False)) # strip html tags and keep readable text . strip=False avoids aggressive trimming
            if part.get_filename():
                attachments.append(part.get_filename())
        except Exception:
            continue

    body = normalize(" ".join(body_parts))
    urls = safe_find_urls(body)

    # Collect headers requested
    from_header = normalize(msg.get("From", ""))
    to_header = normalize(msg.get("To", ""))
    recipient = normalize(msg.get("Delivered-To", "") or msg.get("Envelope-To", "") or msg.get("X-Original-To", "") or to_header)
    return_path = normalize(msg.get("Return-Path", "") or msg.get("Return-path", ""))
    message_id = normalize(msg.get("Message-ID", "") or msg.get("Message-Id", ""))
    x_mailer = normalize(msg.get("X-Mailer", "") or msg.get("X-Mailing", "") or msg.get("X-Mailer", ""))
    x_originating_ip = normalize(msg.get("X-Originating-IP", "") or "")
    # content-type header of whole message (may be multipart)
    content_type = normalize(msg.get("Content-Type", "") or "")

    header_fields = {
        "from_header": from_header,
        "recipient": recipient,
        "return_path": return_path,
        "to_header": to_header,
        "message_id": message_id,
        "x_mailer": x_mailer,
        "x_originating_ip": x_originating_ip,
        "content_type": content_type
    }

    # Build headers text for auth parsing
    headers_text = "\n".join(f"{k}: {v}" for k, v in msg.items())   # converts headers from structured format to key value pairs to make easy to search
    auth_info = parse_auth_from_headers(headers_text)

    return build_features(subject, body, sender, urls, html_present, attachments, label=-1, auth_info=auth_info, header_fields=header_fields)

# =============================
# CSV PARSING (extract headers if present in columns)
# =============================
HEADER_COLUMN_CANDIDATES = [
    "from", "from_header", "sender", "to", "recipient", "delivered-to",
    "return-path", "return_path", "message-id", "message_id",
    "x-mailer", "x_mailer", "x-mailing", "x-originating-ip", "x_originating_ip",
    "content-type", "content_type", "authentication-results", "headers", "raw_headers"
]

def parse_csv_row(row: Dict) -> Optional[Dict]:
    """
    Detects headers heuristically

    Extracts auth info if available

    Picks best fields for body/subject

    Remains robust across datasets"""
    label = normalize_label(row.get("phish") or row.get("label") or row.get("class") or row.get("spam"))

    # Try to assemble headers_text if present in any candidate column
    headers_text_parts: List[str] = []
    header_fields: Dict[str, str] = {}
    # lowercase keys for matching
    lowered = {k.lower(): v for k, v in row.items() if isinstance(k, str)}

    # map specific header fields when present
    header_fields["from_header"] = normalize(row.get("from") or row.get("From") or row.get("from_header") or "")
    header_fields["to_header"] = normalize(row.get("to") or row.get("To") or row.get("to_header") or "")
    header_fields["recipient"] = normalize(row.get("delivered-to") or row.get("Delivered-To") or row.get("envelope-to") or row.get("recipient") or header_fields["to_header"])
    header_fields["return_path"] = normalize(row.get("return-path") or row.get("Return-Path") or row.get("return_path") or "")
    header_fields["message_id"] = normalize(row.get("message-id") or row.get("Message-ID") or row.get("message_id") or "")
    header_fields["x_mailer"] = normalize(row.get("x-mailer") or row.get("X-Mailer") or row.get("x_mailer") or row.get("x-mailing") or "")
    header_fields["x_originating_ip"] = normalize(row.get("x-originating-ip") or row.get("X-Originating-IP") or row.get("x_originating_ip") or "")
    header_fields["content_type"] = normalize(row.get("content-type") or row.get("Content-Type") or row.get("content_type") or "")

    # If CSV has a raw headers column, include it in auth parsing
    for candidate in ("authentication-results", "authentication_results", "auth_results", "headers", "raw_headers", "message_headers"):
        val = row.get(candidate) or row.get(candidate.replace("-", "_"))
        if val:
            headers_text_parts.append(str(val))

    headers_text = "\n".join(headers_text_parts) if headers_text_parts else ""

    auth_info = parse_auth_from_headers(headers_text) if headers_text else {}

    # Extract textual fields for body/subject as before
    text_fields = {k: normalize(v) for k, v in row.items() if isinstance(v, str) and normalize(v)}
    if not text_fields:
        return None

    sorted_fields = sorted(text_fields.items(), key=lambda x: len(x[1]), reverse=True) # sort based on length of the value in descending order longest to shortest
    body = sorted_fields[0][1]  # [key][value]  . pick first longest as body and second longest as subject if exist
    subject = sorted_fields[1][1] if len(sorted_fields) > 1 else ""
    sender = normalize(row.get("from") or row.get("sender") or header_fields.get("from_header") or "")

    urls = safe_find_urls(body)

    return build_features(subject, body, sender, urls, html_present=0, attachments=[], label=label, auth_info=auth_info, header_fields=header_fields)

# =============================
# CHUNKED CSV READER
# =============================
def iter_csv_rows(path: str) -> Iterable[Dict]:
    try:
        for chunk in pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip", chunksize=CSV_CHUNKSIZE):
            chunk = chunk.fillna("")
            for record in chunk.to_dict(orient="records"):  # convert dataframe to records.
                yield record    # return item, stop function , resumes
    except Exception as e:
        logging.error("Failed reading CSV %s: %s", path, e)

# =============================
# DATA LOADING
# =============================
def process_emls(files: List[str]) -> List[Dict]:   # reads files, collect feature dictionaries and convert to dataframe
    if not files:
        return []
    with Pool(MAX_WORKERS) as pool:
        results = pool.map(parse_eml, files)
    return [r for r in results if r]

def load_raw_data(raw_dir: str) -> pd.DataFrame:
    records: List[Dict] = []
    eml_files: List[str] = []
    csv_files: List[str] = []

    for root, _, files in os.walk(raw_dir): # walk every folder, subfolder and file. "_" means i don't care about this value
        for f in files:
            full = os.path.join(root, f)
            if f.lower().endswith(".eml"):
                eml_files.append(full)
            elif f.lower().endswith(".csv"):
                csv_files.append(full)

    logging.info("Found %d EML files, %d CSV files", len(eml_files), len(csv_files))

    if eml_files:
        logging.info("Processing EML files (parallel)...")
        records.extend(process_emls(eml_files))

    for csv_path in csv_files:
        logging.info("Reading CSV (chunked): %s", csv_path)
        for row in iter_csv_rows(csv_path):
            try:
                rec = parse_csv_row(row)
                if rec:
                    records.append(rec)
            except Exception:
                logging.debug("Failed parsing row from %s", csv_path, exc_info=True)
                continue

    return pd.DataFrame(records)

# =============================
# MAIN
# =============================
def main():
    start = time.time()
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, OUTPUT_FILE)

    #read the file if exist and creat it if not exist.
    old_df = pd.read_csv(output_path, dtype=str).fillna("") if os.path.exists(output_path) else pd.DataFrame()
    new_df = load_raw_data(RAW_DATA_PATH)

    logging.info("New records extracted: %d", len(new_df))

    # Ensure label present
    if "label" not in new_df.columns:
        new_df["label"] = -1
    if not old_df.empty and "label" not in old_df.columns:
        old_df["label"] = -1

    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)

    if combined.empty:
        logging.warning("No data loaded.")
        return
    # run for each row . str: prevet error if the value is none. r.get() safely reads the value
    combined["_hash"] = combined.apply(lambda r: compute_hash(str(r.get("subject", "")), str(r.get("body", "")), str(r.get("sender", ""))), axis=1)

    before = len(combined)
    combined.drop_duplicates("_hash", inplace=True)
    combined.drop(columns=["_hash"], inplace=True)

    try: # safely convert labels to integers . if fails clean them using isdigit()
        combined["label"] = combined["label"].astype(int)
    except Exception:
        combined["label"] = combined["label"].apply(lambda v: int(v) if str(v).isdigit() else -1)

    combined.to_csv(output_path, index=False)
    logging.info("Deduplicated %d records", before - len(combined))
    logging.info("Final dataset size: %d", len(combined))
    logging.info("Finished in %.2fs", time.time() - start)

if __name__ == "__main__":
    main()
