"""Test 6e - How many common factors does the leftover persistence need?

Null: real market factor + real top-k statistical factors (PCA of market residuals, k = 3, 10, 30)
      + each stock's own fitted 2-unit residual volatility * fresh i.i.d. shocks. Threshold 4.
Usage: python3 -I src/test6e_pca_null.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import load_stocks
from test6b_timescales_stocks import _fit, h_path
from test6d_sector_null import _rep

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"


def main():
    st, _ = load_stocks(ROOT / "data")
    panel = st.loc[:, st.notna().mean() >= 0.98]
    panel = panel.loc[panel.notna().mean(axis=1) >= 0.95]
    R = panel.values * 100
    T, N = R.shape
    mask = np.isfinite(R)
    mkt = np.nanmean(R, axis=1)
    idio, sys1 = np.empty_like(R), np.empty_like(R)
    for i in range(N):
        ok = mask[:, i]
        b = np.polyfit(mkt[ok], R[ok, i], 1)
        sys1[:, i] = b[1] + b[0] * mkt
        idio[:, i] = R[:, i] - sys1[:, i]
    X = np.where(mask, idio, 0.0)
    sd = X.std(0)
    U, S, Vt = np.linalg.svd(X / sd, full_matrices=False)
    var_share = S ** 2 / (S ** 2).sum()
    real = json.loads((OUT / "test6b.json").read_text())["avalanche"]["K_threshold=4"]["real"]
    res = {"real_lag1": real["lag1_slope"], "pca_var_share_cumulative": {k: float(var_share[:k].sum()) for k in (3, 10, 30)}}
    for k in (3, 10, 30):
        fitted = (U[:, :k] * S[:k]) @ Vt[:k] * sd
        resid = np.where(mask, X - fitted, np.nan)
        systematic = sys1 + fitted
        with ProcessPoolExecutor(max_workers=4) as ex:
            fits = list(ex.map(_fit, [(resid[:, i], 2) for i in range(N)], chunksize=6))
        h = np.column_stack([h_path(resid[:, i], fits[i]) for i in range(N)])
        nu = np.array([f["nu"] for f in fits])
        with ProcessPoolExecutor(max_workers=4) as ex:
            reps = pd.DataFrame(list(ex.map(_rep, [(4900 + r, 4.0, systematic, h, nu, mask) for r in range(20)])))
        v = reps["lag1_slope"]
        res[f"k={k}"] = {"lag1_mean": float(v.mean()), "p2.5": float(v.quantile(0.025)), "p97.5": float(v.quantile(0.975)),
                         "mean_firing": float(reps["mean_firing"].mean())}
        print(f"k={k} factors ({var_share[:k].sum():.1%} of residual var): null lag1 {v.mean():.3f} [{v.quantile(0.025):.3f}, {v.quantile(0.975):.3f}] vs real {real['lag1_slope']:.3f}", flush=True)
    (OUT / "test6e.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
