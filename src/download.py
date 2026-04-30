"""
Download MSCI SEC filings (10-K FY2025, 10-Q Q1 2026) from EDGAR.
Saves raw .htm files to the data/raw/ directory.
"""

import os
import time
import requests

from paths import RAW_DIR

FILINGS = [
    {
        "name": "MSCI_10K_FY2025",
        "url": "https://www.sec.gov/Archives/edgar/data/1408198/000140819826000011/msci-20251231.htm",
        "filename": "msci_10k_fy2025.htm",
        "description": "10-K Annual Report — Fiscal Year 2025 (period ending Dec 31, 2025)",
    },
    {
        "name": "MSCI_10Q_Q1_2026",
        "url": "https://www.sec.gov/Archives/edgar/data/1408198/000140819826000034/msci-20260331.htm",
        "filename": "msci_10q_q1_2026.htm",
        "description": "10-Q Quarterly Report — Q1 2026 (period ending Mar 31, 2026)",
    },
]

# SEC requires a descriptive User-Agent: https://www.sec.gov/os/accessing-edgar-data
HEADERS = {
    "User-Agent": "MSCI-GraphRAG-Project vivek@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


def download_filing(filing: dict) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = str(RAW_DIR / filing["filename"])

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1_048_576
        print(f"[SKIP] {filing['name']} already exists ({size_mb:.1f} MB) → {dest}")
        return dest

    print(f"[DOWN] {filing['name']}")
    print(f"       {filing['description']}")
    print(f"       URL: {filing['url']}")

    response = requests.get(filing["url"], headers=HEADERS, timeout=60)
    response.raise_for_status()

    with open(dest, "wb") as f:
        f.write(response.content)

    size_mb = os.path.getsize(dest) / 1_048_576
    print(f"       Saved → {dest} ({size_mb:.1f} MB)")
    return dest


def main():
    print("=" * 60)
    print("MSCI Knowledge Graph RAG — Document Downloader")
    print("=" * 60)

    downloaded = []
    for i, filing in enumerate(FILINGS):
        path = download_filing(filing)
        downloaded.append(path)
        if i < len(FILINGS) - 1:
            time.sleep(1)  # be polite to SEC servers

    print()
    print("Done. Files saved:")
    for path in downloaded:
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
