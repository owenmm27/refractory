"""Download daily closes for major stock indices outside the US (plus the S&P 500 for reference) from Yahoo.

Timestamps are converted to each exchange's local time before taking the date (Asian and Australian sessions
otherwise land on the previous UTC day). Saves data/intl/<code>.csv with columns date, close.

Usage: python3 -I src/download_intl.py <project_root>
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "data" / "intl"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
INDICES = {
    "GSPC": "^GSPC", "FTSE": "^FTSE", "DAX": "^GDAXI", "CAC": "^FCHI", "SMI": "^SSMI", "AEX": "^AEX",
    "N225": "^N225", "HSI": "^HSI", "ASX200": "^AXJO", "TSX": "^GSPTSE", "KOSPI": "^KS11", "TAIEX": "^TWII",
    "SENSEX": "^BSESN", "BOVESPA": "^BVSP", "IPC": "^MXX",
}


def fetch(ysym):
    p1 = int(pd.Timestamp("1960-01-01").timestamp())
    p2 = int(pd.Timestamp.now().timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}?period1={p1}&period2={p2}&interval=1d"
    for k in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=40)
            if r.status_code == 200:
                return json.loads(r.text)["chart"]["result"][0]
            if r.status_code in (400, 404):
                return None
        except (requests.RequestException, KeyError, ValueError):
            pass
        time.sleep(2 * (k + 1))
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for code, ysym in INDICES.items():
        res = fetch(ysym)
        if res is None:
            print(code, "FAILED", flush=True)
            continue
        off = int(res["meta"].get("gmtoffset", 0))
        ts = pd.to_datetime([t + off for t in res["timestamp"]], unit="s").normalize()
        close = res["indicators"]["quote"][0]["close"]
        df = pd.DataFrame({"close": close}, index=ts).dropna()
        df = df[~df.index.duplicated(keep="last")]
        df.index.name = "date"
        df.to_csv(OUT / f"{code}.csv")
        print(f"{code:8s} {ysym:8s} {len(df):6d} days  {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
