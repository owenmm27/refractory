"""Download free daily market data into data/.

Sources
- Ken French data library: 49 industry portfolios (daily, from 1926) and the
  Fama-French market factor (daily, from 1926).
- CBOE: VIX daily history (from 1990).
- Yahoo Finance chart API: adjusted closes for current S&P 500 members (from 1990).
  NOTE: current members only, so this panel has survivorship bias.

Usage: python3 -I src/download.py <project_root>
"""
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(sys.argv[1]).resolve()
DATA = ROOT / "data"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "french/49_Industry_Portfolios_daily_CSV.zip": FRENCH + "49_Industry_Portfolios_daily_CSV.zip",
    "french/F-F_Research_Data_Factors_daily_CSV.zip": FRENCH + "F-F_Research_Data_Factors_daily_CSV.zip",
    "vix/VIX_History.csv": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "meta/constituents.csv": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
}


def fetch(url, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=40)
            if r.status_code == 200:
                return r
            if r.status_code in (404, 400):
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def get_static():
    for rel, url in FILES.items():
        out = DATA / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            continue
        r = fetch(url)
        if r is None:
            print("FAILED", url)
            continue
        out.write_bytes(r.content)
        print("saved", rel, len(r.content), "bytes")
        if rel.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for name in z.namelist():
                    # only plain csv members, flattened into the same folder
                    if name.lower().endswith(".csv") and "/" not in name and "\\" not in name:
                        (out.parent / name).write_bytes(z.read(name))
                        print("  extracted", name)


def yahoo_one(sym):
    ysym = sym.replace(".", "-")
    p1 = int(pd.Timestamp("1990-01-01").timestamp())
    p2 = int(pd.Timestamp.now().timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits")
    r = fetch(url)
    if r is None:
        return sym, None
    try:
        res = json.loads(r.text)["chart"]["result"][0]
        ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        vol = res["indicators"]["quote"][0]["volume"]
        df = pd.DataFrame({"adj": adj, "vol": vol}, index=ts)
        df = df[~df.index.duplicated(keep="last")]
        return sym, df
    except Exception:
        return sym, None


def get_yahoo():
    out_p = DATA / "yahoo" / "sp500_adjclose.csv"
    out_v = DATA / "yahoo" / "sp500_volume.csv"
    if out_p.exists() and out_v.exists():
        print("yahoo panel already present")
        return
    out_p.parent.mkdir(parents=True, exist_ok=True)
    syms = pd.read_csv(DATA / "meta" / "constituents.csv")["Symbol"].astype(str).tolist()
    syms = syms + ["^GSPC"]
    px, vo, failed = {}, {}, []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (sym, df) in enumerate(ex.map(yahoo_one, syms)):
            if df is None or df["adj"].notna().sum() < 250:
                failed.append(sym)
            else:
                px[sym] = df["adj"]
                vo[sym] = df["vol"]
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(syms)} done, {len(failed)} failed", flush=True)
    pd.DataFrame(px).sort_index().to_csv(out_p)
    pd.DataFrame(vo).sort_index().to_csv(out_v)
    print("yahoo: saved", len(px), "series; failed:", failed)


if __name__ == "__main__":
    get_static()
    get_yahoo()
