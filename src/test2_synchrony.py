"""Test 2 - Synchrony as a market state.

Does cross-sectional synchrony (how much the 49 industries move together) forecast next-month
volatility, tail losses or returns BEYOND what lagged volatility (and VIX) already tell you?

Usage: python3 -I src/test2_synchrony.py <project_root>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import load_industries, load_market, load_stocks, load_vix

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
H = 21  # forecast horizon / block length in trading days


def rolling_synchrony(R, window, min_units=30):
    """Daily series of (mean pairwise correlation, top-eigenvalue share, absorption ratio top N/5)."""
    vals = R.values
    n = len(vals)
    out = np.full((n, 3), np.nan)
    for t in range(window - 1, n):
        w = vals[t - window + 1:t + 1]
        keep = ~np.isnan(w).any(0)
        if keep.sum() < min_units:
            continue
        w = w[:, keep]
        if (w.std(0) == 0).any():
            continue
        c = np.corrcoef(w, rowvar=False)
        k = c.shape[0]
        out[t, 0] = (c.sum() - k) / (k * (k - 1))
        out[t, 1] = np.linalg.eigvalsh(c)[-1] / k
        ev = np.linalg.eigvalsh(np.cov(w, rowvar=False))
        out[t, 2] = ev[-max(k // 5, 1):].sum() / ev.sum()
    return pd.DataFrame(out, index=R.index, columns=["mean_corr", "eig1_share", "absorption"])


def log_rv(r, window):
    return np.log(np.sqrt((r ** 2).rolling(window).sum() * 252 / window))


def clark_west(y, p_small, p_big):
    """Clark-West (2007) test that the bigger nested model forecasts better. Returns t-stat (one-sided)."""
    f = (y - p_small) ** 2 - ((y - p_big) ** 2 - (p_small - p_big) ** 2)
    f = f[np.isfinite(f)]
    return float(f.mean() / (f.std(ddof=1) / np.sqrt(len(f))))


def expanding_oos(y, X, start):
    """Expanding-window OLS, one block ahead. Returns predictions (NaN before `start`)."""
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])
    pred = np.full(n, np.nan)
    for t in range(start, n):
        b, *_ = np.linalg.lstsq(Xc[:t], y[:t], rcond=None)
        pred[t] = Xc[t] @ b
    return pred


def r2(y, pred, bench):
    ok = np.isfinite(pred) & np.isfinite(bench)
    return float(1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - bench[ok]) ** 2))


def evaluate(df, target, base_cols, extra_cols, start=120):
    """Compare base model vs base+extra for forecasting `target`. In-sample HAC t-stats + OOS R2 gain."""
    d = df.dropna(subset=[target] + base_cols + extra_cols)
    y = d[target].values
    Xb, Xe = d[base_cols].values, d[base_cols + extra_cols].values
    ols = sm.OLS(y, sm.add_constant(Xe)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    ols_b = sm.OLS(y, sm.add_constant(Xb)).fit()
    pb, pe = expanding_oos(y, Xb, start), expanding_oos(y, Xe, start)
    mean_b = np.array([y[:t].mean() if t >= start else np.nan for t in range(len(y))])
    names = ["const"] + base_cols + extra_cols
    return {"n_blocks": int(len(y)), "sample": [str(d.index[0].date()), str(d.index[-1].date())],
            "insample_r2_base": float(ols_b.rsquared), "insample_r2_full": float(ols.rsquared),
            "tstats_full": {k: float(v) for k, v in zip(names, ols.tvalues)},
            "coefs_full": {k: float(v) for k, v in zip(names, ols.params)},
            "oos_r2_base": r2(y, pb, mean_b), "oos_r2_full": r2(y, pe, mean_b),
            "oos_r2_gain": r2(y, pe, mean_b) - r2(y, pb, mean_b),
            "clark_west_t": clark_west(y, pb, pe)}


def main():
    OUT.mkdir(exist_ok=True)
    ind = load_industries(ROOT / "data")
    mkt = load_market(ROOT / "data")
    vix = load_vix(ROOT / "data")
    res = {}

    sync = rolling_synchrony(ind, 126)
    sync63 = rolling_synchrony(ind, 63)["mean_corr"].rename("mean_corr_63")
    daily = pd.concat([mkt, sync, sync63], axis=1)
    daily["lrv_5"] = log_rv(mkt, 5)
    daily["lrv_21"] = log_rv(mkt, 21)
    daily["lrv_126"] = log_rv(mkt, 126)
    daily["lrv_252"] = log_rv(mkt, 252)
    # Kritzman-style standardized shift: (15-day mean - 1-year mean) / 1-year std
    s = daily["mean_corr"]
    daily["sync_shift"] = (s.rolling(15).mean() - s.rolling(252).mean()) / s.rolling(252).std()
    daily["sync63_minus_126"] = daily["mean_corr_63"] - daily["mean_corr"]
    daily["log_vix"] = np.log(vix.reindex(daily.index))

    # targets over the NEXT H days (strictly after the feature date)
    fwd_sq = (mkt ** 2).rolling(H).sum().shift(-H)
    daily["y_lrv"] = np.log(np.sqrt(fwd_sq * 252 / H))
    daily["y_ret"] = np.log1p(mkt).rolling(H).sum().shift(-H)
    daily["y_worst_day"] = mkt.rolling(H).min().shift(-H)

    # non-overlapping blocks: sample every H-th day
    blocks = daily.iloc[::H].copy()
    blocks.to_csv(OUT / "test2_blocks.csv")
    daily[["mean_corr", "eig1_share", "absorption", "mean_corr_63", "lrv_21"]].to_csv(OUT / "test2_daily_sync.csv")

    res["synchrony_summary"] = {
        "mean": float(s.mean()), "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95)),
        "corr_with_log_vol_21": float(daily[["mean_corr", "lrv_21"]].corr().iloc[0, 1]),
        "corr_with_log_vol_126": float(daily[["mean_corr", "lrv_126"]].corr().iloc[0, 1]),
        "corr_meancorr_vs_eig1": float(daily[["mean_corr", "eig1_share"]].corr().iloc[0, 1]),
        "corr_meancorr_vs_absorption": float(daily[["mean_corr", "absorption"]].corr().iloc[0, 1]),
        "block_autocorr_lag1": float(blocks["mean_corr"].autocorr(1)),
        "top_dates": [str(d.date()) for d in s.resample("YE").max().nlargest(8).index]}

    base = ["lrv_5", "lrv_21", "lrv_126", "lrv_252"]
    # --- 1. volatility forecasting, full history
    res["vol_forecast_full"] = {
        "sync_level": evaluate(blocks, "y_lrv", base, ["mean_corr"]),
        "sync_level_and_shift": evaluate(blocks, "y_lrv", base, ["mean_corr", "sync_shift"]),
        "eig1_share": evaluate(blocks, "y_lrv", base, ["eig1_share"]),
        "absorption": evaluate(blocks, "y_lrv", base, ["absorption"]),
        "sync_alone_no_vol": evaluate(blocks, "y_lrv", [], ["mean_corr"]) if False else None}
    # synchrony on its own (no volatility controls) for reference
    d = blocks.dropna(subset=["y_lrv", "mean_corr"])
    res["vol_forecast_full"]["sync_alone_insample_r2"] = float(sm.OLS(d["y_lrv"], sm.add_constant(d[["mean_corr"]])).fit().rsquared)
    del res["vol_forecast_full"]["sync_alone_no_vol"]

    # --- 2. volatility forecasting, by era and with VIX
    for name, a, b in [("1926-1969", "1926", "1969"), ("1970-1999", "1970", "1999"), ("2000-2026", "2000", "2026")]:
        sub = blocks.loc[a:b + "-12-31"]
        res.setdefault("vol_forecast_by_era", {})[name] = evaluate(sub, "y_lrv", base, ["mean_corr"], start=60)
    post90 = blocks.loc["1990":]
    res["vol_forecast_with_vix_1990on"] = {
        "vix_added_to_har": evaluate(post90, "y_lrv", base, ["log_vix"], start=60),
        "sync_added_to_har_plus_vix": evaluate(post90, "y_lrv", base + ["log_vix"], ["mean_corr"], start=60)}

    # --- 3. returns and tail losses
    res["return_forecast"] = {
        "sync_level_and_shift": evaluate(blocks, "y_ret", ["lrv_21", "lrv_126"], ["mean_corr", "sync_shift"]),
        "sync_shift_only": evaluate(blocks, "y_ret", [], ["sync_shift"])}
    res["worst_day_forecast"] = evaluate(blocks, "y_worst_day", base, ["mean_corr", "sync_shift"])

    # 5% quantile regression of next-block return (in-sample, suggestive only)
    d = blocks.dropna(subset=["y_ret", "lrv_21", "lrv_126", "mean_corr", "sync_shift"])
    qr = sm.QuantReg(d["y_ret"], sm.add_constant(d[["lrv_21", "lrv_126", "mean_corr", "sync_shift"]])).fit(q=0.05)
    res["quantile_reg_5pct_insample"] = {"coef": {k: float(v) for k, v in qr.params.items()},
                                         "t": {k: float(v) for k, v in qr.tvalues.items()}}

    # --- 4. double sort: within volatility terciles, does synchrony still separate next-month outcomes?
    d = blocks.dropna(subset=["y_lrv", "y_ret", "lrv_21", "mean_corr"]).copy()
    d["vol_tercile"] = pd.qcut(d["lrv_21"], 3, labels=["low vol", "mid vol", "high vol"])
    d["sync_tercile"] = d.groupby("vol_tercile", observed=True)["mean_corr"].transform(
        lambda x: pd.qcut(x, 3, labels=["low sync", "mid sync", "high sync"]))
    crash_cut = d["y_ret"].quantile(0.05)
    d["crash"] = d["y_ret"] <= crash_cut
    d["fwd_vol_ann"] = np.exp(d["y_lrv"])
    g = d.groupby(["vol_tercile", "sync_tercile"], observed=True)
    tab = pd.DataFrame({"n": g.size(), "next_vol_ann": g["fwd_vol_ann"].median(),
                        "next_ret_mean": g["y_ret"].mean(), "crash_freq": g["crash"].mean()})
    res["double_sort"] = {f"{a} / {b}": {k: float(v) for k, v in row.items()} for (a, b), row in tab.iterrows()}
    res["double_sort_crash_cutoff_return"] = float(crash_cut)
    tab.to_csv(OUT / "test2_double_sort.csv")

    # --- 5. what did synchrony look like BEFORE the worst blocks vs DURING them?
    d = blocks.dropna(subset=["y_ret", "mean_corr"]).copy()
    d["sync_pct"] = d["mean_corr"].rank(pct=True)
    d["vol_pct"] = d["lrv_21"].rank(pct=True)
    worst = d.nsmallest(25, "y_ret")
    after = d["sync_pct"].shift(-1).reindex(worst.index)
    res["worst_25_blocks"] = {"sync_percentile_before": float(worst["sync_pct"].mean()),
                              "sync_percentile_after": float(after.mean()),
                              "vol_percentile_before": float(worst["vol_pct"].mean()),
                              "dates": [str(x.date()) for x in worst.index[:10]]}

    # --- 6. robustness: individual stocks instead of industries (1990+, survivorship-biased)
    st, spx = load_stocks(ROOT / "data")
    wide = st.loc[:, st.notna().sum() >= 0.98 * len(st)]  # stocks with (near) full history
    rows = []
    idx = np.arange(126, len(wide), H)
    for t in idx:
        w = wide.values[t - 126:t]
        keep = ~np.isnan(w).any(0)
        c = np.corrcoef(w[:, keep], rowvar=False)
        k = c.shape[0]
        rows.append((wide.index[t - 1], (c.sum() - k) / (k * (k - 1))))
    stock_sync = pd.Series(dict(rows), name="stock_mean_corr")
    sb = pd.DataFrame({"stock_mean_corr": stock_sync})
    sr = spx.fillna(0)
    for w in (5, 21, 126, 252):
        sb[f"lrv_{w}"] = log_rv(sr, w).reindex(sb.index)
    sb["y_lrv"] = np.log(np.sqrt((sr ** 2).rolling(H).sum().shift(-H) * 252 / H)).reindex(sb.index)
    sb["log_vix"] = np.log(vix.reindex(sb.index))
    res["stocks_1990on"] = {"n_stocks": int(wide.shape[1]),
                            "sync_added_to_har": evaluate(sb, "y_lrv", base, ["stock_mean_corr"], start=60),
                            "sync_added_to_har_plus_vix": evaluate(sb, "y_lrv", base + ["log_vix"], ["stock_mean_corr"], start=60)}
    sb.to_csv(OUT / "test2_stock_sync.csv")

    (OUT / "test2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
