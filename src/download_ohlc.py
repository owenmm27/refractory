"""Download daily open/high/low/close from Yahoo for the model-free range checks (Test 15b).

  * the 15 stock indices of Test 14, plus SPY      -> data/ohlc/<code>.csv  (date, open, high, low, close)
  * the stock panel of Tests 10-13 (current S&P 500 members with >= 98% coverage since 1990)
                                                   -> data/ohlc/stocks_{open,high,low,close}.csv (wide)
Prices are as reported (not split-adjusted). The range checks use only same-day ratios (high/low, close/open)
and close-to-close returns from the adjusted-close file, so splits don't matter.

Usage: python3 -I src/download_ohlc.py <project_root>
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_intl import INDICES

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "data" / "ohlc"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}


def fetch(ysym, start="1960-01-01"):
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp.now().timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}?period1={p1}&period2={p2}&interval=1d"
    for k in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=40)
            if r.status_code == 200:
                res = json.loads(r.text)["chart"]["result"][0]
                off = int(res["meta"].get("gmtoffset", 0))
                ts = pd.to_datetime([t + off for t in res["timestamp"]], unit="s").normalize()
                q = res["indicators"]["quote"][0]
                df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")}, index=ts).dropna()
                df = df[~df.index.duplicated(keep="last")]
                df.index.name = "date"
                return df
            if r.status_code in (400, 404):
                return None
        except (requests.RequestException, KeyError, ValueError, TypeError):
            pass
        time.sleep(2 * (k + 1))
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for code, ysym in list(INDICES.items()) + [("SPY", "SPY")]:
        df = fetch(ysym)
        if df is None:
            print(code, "FAILED", flush=True)
            continue
        df.to_csv(OUT / f"{code}.csv")
        ok = (df["high"] > df["low"]).mean()
        print(f"{code:8s} {len(df):6d} days {df.index[0].date()} -> {df.index[-1].date()}  high>low on {ok:.0%}", flush=True)
        time.sleep(0.5)

    px = pd.read_csv(ROOT / "data" / "yahoo" / "sp500_adjclose.csv", index_col=0, parse_dates=True)
    px = px.drop(columns=["^GSPC"])
    syms = list(px.columns[px.notna().mean() >= 0.98])
    print(f"{len(syms)} panel stocks", flush=True)

    def one(sym):
        time.sleep(0.2)
        return sym, fetch(sym.replace(".", "-"), "1990-01-01")

    with ThreadPoolExecutor(max_workers=4) as ex:
        got = dict(ex.map(one, syms))
    failed = [s for s, d in got.items() if d is None]
    print("failed:", failed, flush=True)
    for field in ("open", "high", "low", "close"):
        wide = pd.DataFrame({s: d[field] for s, d in got.items() if d is not None})
        wide.to_csv(OUT / f"stocks_{field}.csv")
    print("saved stock OHLC", flush=True)


if __name__ == "__main__":
    main()
