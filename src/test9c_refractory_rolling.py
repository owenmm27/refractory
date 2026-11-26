"""Test 9c - When did the refractory period appear? Rolling 6-year windows (step 2 years), CRSP market 1926-2026.
Reports next-day variance after a 3-SD surprise (as % of the persistent state; 100% = no refractory effect),
the likelihood-ratio statistic for the refractory term, and the window's lag-1 return autocorrelation.

Usage: python3 -I src/test9c_refractory_rolling.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import fit_multi, load_market
from test9b_refractory import fit

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()


def one(args):
    start, end, x = args
    r = (x - x.mean()) * 100
    base = fit_multi(r, 2)
    ll, z, extra = fit(r, "refractory", base["params_vec"])
    xc = x - x.mean()
    return {"start": start, "end": end, "mid": start + 3, "rho": extra["rho"],
            "next_day_var_after_3sd_pct": 100 * extra["next_day_variance_after_3sd_surprise"],
            "LR": float(2 * (ll - base["loglik"])), "ret_ac1": float((xc[1:] * xc[:-1]).sum() / (xc ** 2).sum()),
            "base_corr_abs_z_lag1": float(np.corrcoef(np.abs(base["z"][:-1]), np.abs(base["z"][1:]))[0, 1])}


def main():
    mkt = load_market(ROOT / "data")
    jobs = [(y, y + 6, mkt.loc[str(y):str(y + 5)].values) for y in range(1927, 2021, 2)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "test9c_rolling.csv", index=False)
    print(df.round(3).to_string(index=False))
    print("corr(refractory strength, lag-1 return autocorr) across windows:",
          round(float(np.corrcoef(df["next_day_var_after_3sd_pct"], df["ret_ac1"])[0, 1]), 3))


if __name__ == "__main__":
    main()
