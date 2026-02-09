"""
reputation.py

VirusTotal client module for PhishGuard pipeline.
Supports reading from CSV and outputting reports.
"""

import os
import time
import json
import hashlib
import logging
import sqlite3 
import requests
import csv
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    errors: int = 0

class SimpleDiskCache:
    """Lightweight SQLite cache to minimize API calls and costs."""
    def __init__(self, db_path: str = ".vt_cache.db", default_ttl_seconds: int = 86400):
        self.db_path = db_path
        self.default_ttl = default_ttl_seconds
        self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, expires_at REAL)")
        except sqlite3.Error as e:
            logger.error(f"Cache DB Init Error: {e}")

    def get(self, key: str) -> Optional[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row and time.time() < row[1]:
                    return json.loads(row[0])
        except Exception:
            pass
        return None

    def set(self, key: str, value: Dict) -> None:
        try:
            expires_at = time.time() + self.default_ttl
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(value), expires_at))
        except sqlite3.Error as e:
            logger.warning(f"Cache Write Error: {e}")

class VirusTotalClient:
    BASE_URL = "https://www.virustotal.com/api/v3"

    def __init__(self, api_key: str, cache_path: str = ".vt_cache.db"):
        self.api_key = api_key
        self.cache = SimpleDiskCache(cache_path)
        self.stats = CacheStats()
        self.session = requests.Session()
        self.session.headers.update({"x-apikey": self.api_key})

    def _calculate_score(self, stats: Dict[str, int]) -> float:
        """Weighted risk score: Malicious (1.0), Suspicious (0.5)."""
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        total = sum(stats.values())
        if total == 0: return 0.0
        return round(min((malicious + (0.5 * suspicious)) / total, 1.0), 3)

    def _get_score(self, artifact_type: str, value: str) -> float:
        if not value or value.lower() == 'nan': return 0.0
        
        cache_key = hashlib.sha256(f"{artifact_type}:{value}".encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached:
            self.stats.hits += 1
            return cached['score']

        self.stats.misses += 1
        endpoint_map = {"url": "urls", "domain": "domains", "ip": "ip_addresses"}
        url = f"{self.BASE_URL}/{endpoint_map[artifact_type]}"
        
        # URL IDs must be base64 encoded for VT v3
        if artifact_type == "url":
            import base64
            url = f"{url}/{base64.urlsafe_b64encode(value.encode()).decode().strip('=')}"
        else:
            url = f"{url}/{value}"

        try:
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                stats = resp.json()['data']['attributes']['last_analysis_stats']
                score = self._calculate_score(stats)
                self.cache.set(cache_key, {"score": score})
                return score
            elif resp.status_code == 404:
                return 0.0
        except Exception as e:
            logger.error(f"API Error for {value}: {e}")
            self.stats.errors += 1
        return 0.0

    def process_csv(self, file_path: str):
        """Reads CSV and prints reputation check results."""
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found.")
            return

        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                # Extract and clean artifacts (handle comma-separated strings in cells)
                urls = [u.strip() for u in row.get('urls', '').split(',') if u.strip()]
                domains = [d.strip() for d in row.get('domains', '').split(',') if d.strip()]
                ips = [ip.strip() for ip in row.get('ip_urls', '').split(',') if ip.strip()]

                print(f"\n--- Email Row {i+1} reputation check results ---")
                print("{")
                
                if urls:
                    print("    URLs {")
                    for u in urls: print(f'        "{u}:{self._get_score("url", u)}",\n')
                    print("    }")
                
                if domains:
                    print("    domains {")
                    for d in domains: print(f'        "{d}:{self._get_score("domain", d)}",')
                    print("    }")
                
                if ips:
                    print("    IPs {")
                    for ip in ips: print(f'        "{ip}:{self._get_score("ip", ip)}",')
                    print("    }")
                
                print("}")

# ==========================================
#  Execution Stub
# ==========================================
if __name__ == "__main__":
    # 1. Create a dummy CSV for testing if it doesn't exist
    CSV_FILE = r"C:\Users\hassan\Desktop\phishing_detection_system\data\processed\phishguard_features.csv"
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["urls", "domains", "ip_urls"])
            writer.writerow([
                "http://testsafebrowsing.appspot.com/s/phishing.html", 
                "malware.wicar.org", 
                "8.8.8.8"
            ])

    # 2. Run the client
    VT_KEY = os.getenv("VT_API_KEY", "6eb1277ac814620da44a53cc049d2bfe7a4239c6ea6c1c5d74e86707d20baa54")
    client = VirusTotalClient(api_key=VT_KEY)
    client.process_csv(CSV_FILE)
