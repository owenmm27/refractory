"""Earnings-release dates from SEC EDGAR: 8-K filings with Item 2.02 ("Results of Operations").

Writes data/earnings/earnings_8k_202.csv (symbol, cik, filing_date, accepted_utc, items).
Uses a generic client name (no personal details); throttled well under SEC's 10 requests/second.

Usage: python3 -I src/download_earnings.py <project_root>
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "data" / "earnings"
UA = {"User-Agent": "stockmarket-research-script research@example.org"}


def get(url, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except (requests.RequestException, json.JSONDecodeError):
            pass
        time.sleep(1.5 * (k + 1))
    return None


def rows_from(block, sym, cik):
    out = []
    for i, form in enumerate(block["form"]):
        if form in ("8-K", "8-K/A") and "2.02" in (block["items"][i] or ""):
            out.append((sym, cik, block["filingDate"][i], block["acceptanceDateTime"][i], block["items"][i], form))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None
    meta = pd.read_csv(ROOT / "data" / "meta" / "constituents.csv")
    if symbols is not None:
        meta = meta[meta["Symbol"].isin(symbols["symbol"])]
    rows, missing = [], []
    for n, (sym, cik) in enumerate(zip(meta["Symbol"], meta["CIK"])):
        d = get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        time.sleep(0.15)
        if d is None:
            missing.append(sym)
            continue
        rows += rows_from(d["filings"]["recent"], sym, int(cik))
        for f in d["filings"].get("files", []):
            if f.get("filingTo", "9999") < "2003-01-01":
                continue
            older = get(f"https://data.sec.gov/submissions/{f['name']}")
            time.sleep(0.15)
            if older:
                rows += rows_from(older, sym, int(cik))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(meta)} companies, {len(rows)} filings", flush=True)
    df = pd.DataFrame(rows, columns=["symbol", "cik", "filing_date", "accepted_utc", "items", "form"]).drop_duplicates()
    df.to_csv(OUT / "earnings_8k_202.csv", index=False)
    print(f"saved {len(df)} filings for {df['symbol'].nunique()} companies; missing: {missing}")


if __name__ == "__main__":
    main()
