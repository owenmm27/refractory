"""Test 14 - Does the market unit replicate outside the US?

For 14 non-US stock indices (and the S&P 500 on the same code), by era (full sample, pre-2000, 2000-2012,
2013-2026), fit the index-level pieces of the market-brain model:
  * two-speed volatility state: likelihood ratio of 2 units vs 1, half-lives;
  * refractory term (Test 9/10): likelihood ratio, next-day variance after a 3-SD surprise; and the overreaction
    signature a plain 2-unit model leaves (corr of |z| on consecutive days);
  * reversal: lag-1 autocorrelation, variance ratios VR(5), VR(21), lag-1 autocorrelation in turbulent vs calm
    periods (trailing 21-day volatility above/below median);
  * stress-keyed reversal (Test 12 v2 market unit, 2000-2026): likelihood ratio and strength c.
Data: Yahoo daily closes (price indices). Exact repeat closes (holiday stubs) and |r| > 40% glitches are dropped.

Usage: python3 -I src/test14_intl.py <project_root>
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import test10_refractory_where as T10
import test12c_model_v2 as C
from common import fit_multi
from test4_balance import ac1, variance_ratio

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
DATA = ROOT / "data" / "intl"
ERAS = {"full": (None, None), "pre-2000": (None, "1999-12-31"), "2000-2012": ("2000-01-01", "2012-12-31"),
        "2013-2026": ("2013-01-01", "2026-12-31"), "2000-2026": ("2000-01-01", "2026-12-31")}
NAMES = {"GSPC": "S&P 500 (US)", "FTSE": "FTSE 100 (UK)", "DAX": "DAX (Germany)", "CAC": "CAC 40 (France)",
         "SMI": "SMI (Switzerland)", "AEX": "AEX (Netherlands)", "N225": "Nikkei 225 (Japan)", "HSI": "Hang Seng (Hong Kong)",
         "ASX200": "ASX 200 (Australia)", "TSX": "TSX (Canada)", "KOSPI": "KOSPI (Korea)", "TAIEX": "TAIEX (Taiwan)",
         "SENSEX": "Sensex (India)", "BOVESPA": "Bovespa (Brazil)", "IPC": "IPC (Mexico)"}


def load(code):
    s = pd.read_csv(DATA / f"{code}.csv", parse_dates=["date"]).set_index("date")["close"]
    s = s[(s > 0) & (s.index < "2026-11-25")]
    r = (np.log(s).diff() * 100).dropna()
    info = {"n_zero_dropped": int((r == 0).sum()), "n_glitch_dropped": int((r.abs() >= 40).sum())}
    r = r[(r != 0) & (r.abs() < 40)]
    return r, info


def era_job(args):
    code, era, r = args
    x = r.values - r.values.mean()
    out = {"code": code, "era": era, "n": len(x), "start": str(r.index[0].date()), "end": str(r.index[-1].date())}
    f2, f1 = fit_multi(x, 2), fit_multi(x, 1)
    out["LR_two_speed"] = float(2 * (f2["loglik"] - f1["loglik"]))
    out["half_lives"] = [float(h) for h in f2["half_lives"]]
    mask = np.ones(len(x), bool)
    base = T10.base_fit(x, mask)
    own = T10.extend(x, mask, base, own=True)
    out["LR_refractory"] = own["LR"]
    out["rho"] = own["rho_own"]
    out["next_day_after_3sd"] = own["next_day_after_3sd_own"]
    z = base["z"]
    out["absz_lag1_corr"] = float(np.corrcoef(np.abs(z[:-1]), np.abs(z[1:]))[0, 1])
    out["ac1"] = ac1(x)
    out["VR5"] = variance_ratio(x, 5)
    out["VR21"] = variance_ratio(x, 21)
    sig = pd.Series(x ** 2).rolling(21).mean().shift(1).values
    ok = np.isfinite(sig[1:])
    hi = (sig[1:] > np.nanmedian(sig)) & ok
    lo = ~hi & ok
    out["ac1_turbulent"] = float(np.corrcoef(x[:-1][hi], x[1:][hi])[0, 1])
    out["ac1_calm"] = float(np.corrcoef(x[:-1][lo], x[1:][lo])[0, 1])
    if era == "2000-2026":
        m2 = C.fit_market2(x)
        m0 = C.fit_market2(x, statedep=False)
        out["LR_stress_keyed_reversal"] = float(2 * (m2["ll"] - m0["ll"]))
        out["c_stress_keyed"] = m2["c"]
    return out


def main():
    jobs, info = [], {}
    for code in NAMES:
        r, inf = load(code)
        info[code] = inf
        for era, (a, b) in ERAS.items():
            sub = r.loc[a:b]
            if len(sub) >= 2000:
                jobs.append((code, era, sub))
    print(f"{len(jobs)} market-era fits", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for i, o in enumerate(ex.map(era_job, jobs, chunksize=1)):
            rows.append(o)
            if i % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} {o['code']} {o['era']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["name"] = df["code"].map(NAMES)
    df.to_csv(ROOT / "results" / "test14_intl.csv", index=False)
    (ROOT / "results" / "test14.json").write_text(json.dumps({"cleaning": info, "rows": rows}, indent=2, default=float))
    cols = ["n", "LR_two_speed", "LR_refractory", "next_day_after_3sd", "absz_lag1_corr", "ac1", "VR21", "ac1_turbulent", "ac1_calm"]
    for era in ERAS:
        sub = df[df["era"] == era].set_index("code")
        print(f"\n=== {era}")
        print(sub[cols + (["LR_stress_keyed_reversal", "c_stress_keyed"] if era == "2000-2026" else [])].round(3).to_string())


if __name__ == "__main__":
    main()
