#!/usr/bin/env python3
"""
PhishGuard Reputation Checker (VirusTotal) — Updated with ML JSON columns

Outputs per-row:
 - vt_url_scores (multiline human readable)
 - vt_domain_scores (multiline human readable)
 - vt_ip_scores (multiline human readable)
 - vt_urls_json  (JSON list of {artifact,score})
 - vt_domains_json (JSON list ...)
 - vt_ips_json   (JSON list ...)

Features:
 - Uses VT 'reputation' attribute unchanged
 - Robust artifact splitting
 - Caching (vt_cache.json)
 - Progress tracking (vt_progress.json)
 - Rate limiting & daily limit protection
 - Appends JSON machine-readable columns for ML
"""

import os
import csv
import json
import time # use time to Sleep between API calls to respect rate limits
import base64 # Encode URLs for VirusTotal’s API (required for the /urls/ endpoint)
import hashlib # Create SHA‑256 cache keys to avoid storing raw values as keys
import logging # print informative messages 
import requests # make http requests to VT 
import re # use regex to split artifact strings
from typing import Dict, List, Optional, Any # for readability 

csv.field_size_limit(10**9) # to prevent csv.error when dealing w huge emails

# -----------------------
# CONFIG
# -----------------------
VT_API_KEY = os.getenv(
    "VT_API_KEY",
    "6eb1277ac814620da44a53cc049d2bfe7a4239c6ea6c1c5d74e86707d20baa54" 
)

INPUT_CSV = r"C:\Users\hassan\Desktop\phishing_detection_system\data\processed\phishguard_features.csv"

CACHE_FILE = "vt_cache.json"
PROGRESS_FILE = "vt_progress.json"

REQUESTS_PER_DAY_LIMIT = 500
REQUEST_DELAY_SECONDS = 16  # safe for free api limit
BASE_URL = "https://www.virustotal.com/api/v3"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("VT") # creates a logger named vt and print messages without timestamp

# -----------------------
# Utilities
# -----------------------
def load_json(path: str, default: Any): # reads json file if exists
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path: str, data: Any): # write to file 
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

_SPLIT_RE = re.compile(r"[,\;\|\n\r]+") # match on these delimeters ,;|\n\r

def split_artifacts_raw(value: Optional[str]) -> List[str]:
    """
    Split a CSV cell containing multiple artifacts.
    Splits on comma, semicolon, pipe, newline. Strips quotes/whitespace.
    Filters empty tokens, 'nan', '[]'. Preserves order and dedupes.
    """
    if not value:
        return []
    s = str(value)
    parts = _SPLIT_RE.split(s)
    seen = set() 
    out = []
    for p in parts:
        p = p.strip().strip('"').strip("'")
        if not p:
            continue
        if p.lower() in ("nan", "none", "[]"):
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

# -----------------------
# VT client
# -----------------------
class VirusTotalClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session() # Reuses a requests.Session for connection pooling
        self.session.headers.update({"x-apikey": api_key})
        self.cache: Dict[str, Dict[str, Any]] = load_json(CACHE_FILE, {}) # loads previously cached results; each cache entry is a dict with score and timestamp
        self.progress: Dict[str, int] = load_json(PROGRESS_FILE, {"last_row": 0}) # load last processed row 
        self.requests_today = 0 # counts api calls 

    def _cache_key(self, art_type: str, value: str) -> str: # creat a hash on key:value to ensure uniquness
        return hashlib.sha256(f"{art_type}:{value}".encode()).hexdigest()

    def _save_state(self): # save previous calls and last row
        save_json(CACHE_FILE, self.cache)
        save_json(PROGRESS_FILE, self.progress)

    def _query_vt_once(self, artifact_type: str, value: str) -> Optional[float]:
        """
        Query VT and return the 'reputation' attribute exactly as VT provides.
        Returns:
          - float (reputation)
          - -1.0 if artifact unknown or error
          - None if daily limit reached (caller should save state and exit)
        """
        if not value: # if value is empty
            return -1.0

        key = self._cache_key(artifact_type, value) # compute cache key; if found returns it's score
        if key in self.cache:
            try:
                return float(self.cache[key]["score"])
            except Exception:
                # malformed cache: fallback
                return float(self.cache[key].get("score", -1.0))

        if self.requests_today >= REQUESTS_PER_DAY_LIMIT: # check rate limit
            logger.info("Daily API limit reached; saving progress and cache.")
            self._save_state()
            return None

        # Build endpoint
        if artifact_type == "url":
            encoded = base64.urlsafe_b64encode(value.encode()).decode().rstrip("=") 
            # value.encode() converts the string value (the URL) into a bytes object using UTF‑8 encoding. because The base64 module works with bytes, not strings. Encoding turns the URL into a raw byte sequence that can be processed by the base64 functions.
            # base64.urlsafe_b64encode() takes the bytes and applies Base64 encoding, but uses a URL‑safe alphabet. becuase The encoded string will be placed in a URL path (as part of the API endpoint) returns bytes of encoded data 
            # decode() converts the encoded bytes back into a string (using UTF‑8) because we need to manipulate it further to insert it in API url. its easier
            # rstrip("=") removes any trailing = characters from the right side of the string as virustotal requires 
            # all that because base64.urlsafe_b64encode requires byte strings
            endpoint = f"{BASE_URL}/urls/{encoded}"
        elif artifact_type == "domain":
            endpoint = f"{BASE_URL}/domains/{value}"
        elif artifact_type == "ip":
            endpoint = f"{BASE_URL}/ip_addresses/{value}"
        else:
            return -1.0

        try:
            resp = self.session.get(endpoint, timeout=15)
            self.requests_today += 1

            if resp.status_code == 200:
                data = resp.json()
                reputation = data.get("data", {}).get("attributes", {}).get("reputation", None) 
                # if the full path is exists reputation is set to that value safeley returns none if data, arrtibutes, and reputation is missing
                """
                if "data" in data and "attributes" in data["data"]:
                    reputation = data["data"]["attributes"].get("reputation")
                else:
                    reputation = None
                """
                score = float(reputation) if reputation is not None else 0.0
            elif resp.status_code == 404:
                score = -1.0
            else:
                score = -1.0

        except requests.RequestException as e:
            logger.error(f"Network error querying VT for {value!r}: {e}")
            score = -1.0
        except Exception as e:
            logger.error(f"Unexpected error querying VT for {value!r}: {e}")
            score = -1.0

        # Cache and sleep for rate-limit
        self.cache[key] = {"score": score, "ts": int(time.time())} # key is the artifact type (url,ip,domain). maps cache key to dictionary containing actual data containing reputation and time of check
        time.sleep(REQUEST_DELAY_SECONDS)
        return score

    def get_scores_for_list(self, artifact_type: str, artifacts: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        For a list of artifacts, return list of {"artifact":..., "score":...}
        Returns None if daily limit hit (so caller can save/exit)
        """
        out = []
        for art in artifacts:
            score = self._query_vt_once(artifact_type, art)
            if score is None:
                return None
            out.append({"artifact": art, "score": score})
        return out

    @staticmethod # standalone function that happens to be defined inside the class for organisational purposes.
    def format_group_lines(rep_list: List[Dict[str, Any]]) -> str:
        """
        format human-readable group:
        [
        artifact1 : score1
        artifact2 : score2
        ]
        """
        if not rep_list:
            return "[\n]"
        lines = ["["]
        for item in rep_list:
            lines.append(f"{item['artifact']} : {item['score']}")
        lines.append("]")
        return "\n".join(lines)

    def process_csv(self, csv_path: str):
        # Read CSV
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError("CSV has no header")
            fieldnames = list(reader.fieldnames)
            rows = list(reader)

        # Ensure columns exist; add them if missing
        vt_cols_human = ["vt_url_scores", "vt_domain_scores", "vt_ip_scores"]
        vt_cols_json = ["vt_urls_json", "vt_domains_json", "vt_ips_json"]
        for col in vt_cols_human + vt_cols_json:
            if col not in fieldnames:
                fieldnames.append(col)

        start = int(self.progress.get("last_row", 0)) # determines where to start
        logger.info(f"Starting from row: {start}")

        for idx in range(start, len(rows)): # extract and split for each row
            row = rows[idx]

            urls = split_artifacts_raw(row.get("urls", ""))
            domains = split_artifacts_raw(row.get("domains", ""))
            ips = split_artifacts_raw(row.get("ip_urls", ""))

            # print message if rate limit is reached 
            url_reps = self.get_scores_for_list("url", urls)
            if url_reps is None:
                logger.info("Stopped due to daily limit while processing URLs.")
                return

            domain_reps = self.get_scores_for_list("domain", domains)
            if domain_reps is None:
                logger.info("Stopped due to daily limit while processing domains.")
                return

            ip_reps = self.get_scores_for_list("ip", ips)
            if ip_reps is None:
                logger.info("Stopped due to daily limit while processing IPs.")
                return

            # Human-readable grouped multiline strings
            row["vt_url_scores"] = self.format_group_lines(url_reps)
            row["vt_domain_scores"] = self.format_group_lines(domain_reps)
            row["vt_ip_scores"] = self.format_group_lines(ip_reps)

            # JSON ML-ready columns (strings containing JSON arrays)
            row["vt_urls_json"] = json.dumps(url_reps, ensure_ascii=False)
            row["vt_domains_json"] = json.dumps(domain_reps, ensure_ascii=False)
            row["vt_ips_json"] = json.dumps(ip_reps, ensure_ascii=False)

            # Update progress and persist frequently, minimizes data loss if interupted
            self.progress["last_row"] = idx + 1
            self._save_state()

            # Log a readable summary
            logger.info(f"\nRow {idx} processed")
            logger.info(f"URLs:\n{row['vt_url_scores']}")
            logger.info(f"Domains:\n{row['vt_domain_scores']}")
            logger.info(f"IPs:\n{row['vt_ip_scores']}")

        # Write back CSV (overwrite). Ensure every row has all fieldnames.guarantees that every row has exactly the same set of keys
        with open(csv_path, "w", encoding="utf-8", newline="") as f: # newline="" tells python not to translate newlines to not malform the csv rows
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL) # write dictionaries to csv. quoting=csv.QUOTE_MINIMAL tells the writer to only quote fields that contain special characters
            writer.writeheader()
            for r in rows:
                for col in fieldnames:
                    if col not in r: 
                        r[col] = "" # fill missing values with empty string
                writer.writerow(r)

        # Final save & reset progress
        self._save_state()
        logger.info("CSV updated successfully; all rows processed.")
        self.progress["last_row"] = 0
        self._save_state()

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    if not VT_API_KEY or VT_API_KEY == "PUT_YOUR_API_KEY_HERE":
        logger.error("VT_API_KEY is not set. Export VT_API_KEY env var or update the script.")
        raise SystemExit(1)

    client = VirusTotalClient(VT_API_KEY)
    client.process_csv(INPUT_CSV)
