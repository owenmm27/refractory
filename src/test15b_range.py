"""Test 15b - A model-free look at the refractory period, with daily high-low ranges.

Every refractory number before this test is "variance relative to what a fitted volatility model expected",
measured on close-to-close returns. Here no volatility model is used:
  * a shock day t: its close-to-close move is >= 2 (or 3) times the trailing 21-day RMS move;
  * the next day (t+1) is compared with the day after (t+2), the same distance from any other event. In any model
    where a shock raises volatility and the effect then decays (GARCH, EGARCH, Beta-t, HAR, component models...),
    day t+1 is at least as volatile as day t+2, so every ratio below is >= 1 without a refractory period;
  * three measures, ratio of t+1 to t+2 (sums over events):
      net move   = close-to-close squared return (what the volatility models see);
      activity   = intraday high-low range, Parkinson (1980) variance (ln H/L)^2 / (4 ln 2);
      efficiency = net move per unit of activity (r^2 / range variance): how much of the day's travel shows up
                   in its close-to-close change. It falls when prices swing and then come back.
  * a robust version that big crisis days can't dominate: how often the next day is CALMER than the day after
    (|r_t+1| < |r_t+2|, or range_t+1 < range_t+2). Without a refractory period this is at most about 50%; the null
    is checked by simulating each market's own Beta-t-EGARCH fit (Test 15) with no refractory term;
  * also: SPY with the overnight gap and the trading session separated; individual stocks after their own big
    non-earnings moves on calm market days; all 15 indices by era, compared with the model-based dip (Test 15).
95% intervals: bootstrap over calendar months (events in the same month move together).

Usage: python3 -I src/test15b_range.py <project_root>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import test14_intl as T14

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OHLC = ROOT / "data" / "ohlc"
KS = np.arange(-3, 7)
# first date with reliable high/low data on Yahoo (>= 99% of days with high > low and open/close inside the range)
H_L_START = {"GSPC": "1962-01-01", "FTSE": "1984-01-01", "DAX": "1994-01-01", "CAC": "1990-03-01", "SMI": "1990-11-09",
             "AEX": "1992-10-12", "N225": "1989-01-01", "HSI": "1990-01-01", "ASX200": "2001-01-01", "TSX": "2007-01-01",
             "KOSPI": "1996-12-11", "TAIEX": "1997-07-02", "SENSEX": "1997-07-01", "BOVESPA": "1993-04-27",
             "IPC": "1994-01-01", "SPY": "1993-01-29"}
ERAS_US = {"1962-1989": ("1962", "1989"), "1990-1999": ("1990", "1999"), "2000-2012": ("2000", "2012"),
           "2013-2026": ("2013", "2026")}
ERAS_INTL = {"pre-2000": (None, "1999"), "2000-2012": ("2000", "2012"), "2013-2026": ("2013", "2026")}
RNG = np.random.default_rng(7)
NBOOT = 2000


# ----------------------------------------------------------------------------- data
def ohlc_frame(code):
    """Trading days (holiday stubs and glitches dropped as in Test 14) with r, range variance P, gap, oc (pct)."""
    d = pd.read_csv(OHLC / f"{code}.csv", parse_dates=["date"]).set_index("date")
    d = d[(d.index >= H_L_START[code]) & (d.index < "2026-11-25") & (d["close"] > 0)]
    out = pd.DataFrame({"r": np.log(d["close"]).diff() * 100, "gap": np.log(d["open"] / d["close"].shift(1)) * 100,
                        "oc": np.log(d["close"] / d["open"]) * 100,
                        "P": (np.log(d["high"] / d["low"]) * 100) ** 2 / (4 * np.log(2))}, index=d.index).iloc[1:]
    inside = (d["high"] >= d[["open", "close"]].max(axis=1) * 0.9999) & (d["low"] <= d[["open", "close"]].min(axis=1) * 1.0001)
    out["P"] = out["P"].where((d["high"] > d["low"]) & inside)
    return out[(out["r"] != 0) & (out["r"].abs() < 40)]


def shock_size(r, win=21, minp=None):
    rms = np.sqrt(pd.Series(r ** 2).rolling(win, min_periods=minp or win).mean().shift(1).values)
    return np.abs(r) / rms


def lead(a, k):
    a = np.asarray(a, float)
    out = np.full(len(a), np.nan)
    if k >= 0:
        out[:len(a) - k] = a[k:]
    else:
        out[-k:] = a[:k]
    return out


# ----------------------------------------------------------------------------- statistics
def boot(arrays, months, stat):
    """stat(sums) on the full sample and on month-cluster bootstrap resamples. arrays: {name: per-event values}."""
    df = pd.DataFrame(arrays)
    df["m"] = months
    g = df.groupby("m").sum()
    full = stat({c: g[c].sum() for c in arrays})
    M = len(g)
    w = RNG.multinomial(M, np.ones(M) / M, size=NBOOT)
    bs = stat({c: w @ g[c].values for c in arrays})
    return {"est": float(full), "lo": float(np.nanquantile(bs, 0.025)), "hi": float(np.nanquantile(bs, 0.975)),
            "n": int(len(df)), "n_months": int(M)}


def next_vs_following(df, sel):
    """t+1 vs t+2 ratios for the events in boolean mask `sel` (needs r and P on both days)."""
    r1, r2, P1, P2 = (lead(df[c].values, k) for c, k in (("r", 1), ("r", 2), ("P", 1), ("P", 2)))
    ok = sel & np.isfinite(r1) & np.isfinite(r2) & np.isfinite(P1) & np.isfinite(P2)
    A = {"r1": r1[ok] ** 2, "r2": r2[ok] ** 2, "P1": P1[ok], "P2": P2[ok]}
    months = df.index[ok].to_period("M").astype(str)
    return {"net_move": boot(A, months, lambda s: s["r1"] / s["r2"]),
            "activity": boot(A, months, lambda s: s["P1"] / s["P2"]),
            "efficiency": boot(A, months, lambda s: (s["r1"] / s["P1"]) / (s["r2"] / s["P2"]))}


def calmer(df, sel):
    """Share of events where the next day is calmer than the day after: close-to-close |r| and intraday range."""
    r1, r2, P1, P2 = (lead(df[c].values, k) for c, k in (("r", 1), ("r", 2), ("P", 1), ("P", 2)))
    ok = sel & np.isfinite(r1) & np.isfinite(r2)
    okp = ok & np.isfinite(P1) & np.isfinite(P2)
    out = {}
    for name, m, a, b in [("net_move", ok, np.abs(r1), np.abs(r2)), ("activity", okp, P1, P2)]:
        out[name] = boot({"c": (a[m] < b[m]).astype(float), "n": np.ones(m.sum())}, df.index[m].to_period("M").astype(str),
                         lambda s_: s_["c"] / s_["n"])
    return out


def calmer_null(r, n_sim=200, seed=0):
    """Share of 2+ SD days followed by a calmer next day (close-to-close), in Beta-t-EGARCH simulations of this
    series' own fit with no refractory term."""
    import test15_score_driven as S
    x = r - r.mean()
    om, p1, p2, k1, k2, ks, nu = S.unpack(S.fit_base(x, "betat")["x"])
    rng = np.random.default_rng(seed)
    vals = np.empty(n_sim)
    for i in range(n_sim):
        eps = rng.standard_t(nu, len(x) + 1000) * np.sqrt((nu - 2) / nu)
        y = S.betat_sim(eps, om, p1, p2, k1, k2, ks, nu)[1000:]
        xs = shock_size(y)
        a, b = np.abs(lead(y, 1)), np.abs(lead(y, 2))
        ok = np.isfinite(xs) & (xs >= 2) & np.isfinite(b)
        vals[i] = np.mean(a[ok] < b[ok])
    return vals


def profile(df, sel, col):
    """Mean of col at t+k over events, divided by its mean over the 21 days before the event."""
    v = df[col].values ** 2 if col == "r" else df[col].values
    pre = pd.Series(v).rolling(21, min_periods=15).mean().shift(1).values
    out = {}
    for k in KS:
        x = lead(v, k)
        ok = sel & np.isfinite(x) & np.isfinite(pre)
        out[int(k)] = boot({"x": x[ok], "pre": pre[ok]}, df.index[ok].to_period("M").astype(str), lambda s: s["x"] / s["pre"])
    return out


def short(b):
    return {k: {m: round(v["est"], 3) for m, v in b.items()} for k, b in b.items()} if isinstance(b, dict) else b


# ----------------------------------------------------------------------------- pieces
def us():
    df = ohlc_frame("GSPC")
    x = shock_size(df["r"].values)
    out = {}
    for era, (a, b) in ERAS_US.items():
        inera = (df.index >= a) & (df.index <= f"{b}-12-31") & np.isfinite(x)
        out[era] = {"2+": next_vs_following(df, inera & (x >= 2)), "3+": next_vs_following(df, inera & (x >= 3)),
                    "0-1": next_vs_following(df, inera & (x < 1)),
                    "calmer_2+": calmer(df, inera & (x >= 2)), "calmer_0-1": calmer(df, inera & (x < 1))}
        if era == "2000-2012":
            out[era]["profile_2+"] = {"net_move": profile(df, inera & (x >= 2), "r"), "activity": profile(df, inera & (x >= 2), "P")}
        o = out[era]["2+"]
        print(f"S&P 500 {era} after 2+ SD days (n={o['net_move']['n']}): " + "  ".join(
            f"{m} {v['est']:.2f} ({v['lo']:.2f}-{v['hi']:.2f})" for m, v in o.items())
              + f" | calmer next day: net {out[era]['calmer_2+']['net_move']['est']:.3f} activity {out[era]['calmer_2+']['activity']['est']:.3f}", flush=True)
    return out


def spy():
    """Overnight gap vs trading session on the next day (SPY, 2000-2012 and 2013-2026)."""
    df = ohlc_frame("SPY")
    x = shock_size(df["r"].values)
    out = {}
    for era, (a, b) in [("2000-2012", ("2000", "2012-12-31")), ("2013-2026", ("2013", "2026-12-31"))]:
        inera = (df.index >= a) & (df.index <= b) & np.isfinite(x)
        row = {}
        for lab, s in [("after 0-1 SD day", inera & (x < 1)), ("after 2+ SD day", inera & (x >= 2))]:
            for k in (1, 2):
                g, oc, P = (lead(df[c].values, k)[s] for c in ("gap", "oc", "P"))
                ok = np.isfinite(g) & np.isfinite(oc) & np.isfinite(P)
                g, oc, P = g[ok], oc[ok], P[ok]
                row[f"{lab}, t+{k}"] = {"close_to_close_r2": float(np.mean((g + oc) ** 2)), "overnight_gap2": float(np.mean(g ** 2)),
                                       "session2": float(np.mean(oc ** 2)), "2cov_gap_session": float(2 * np.mean(g * oc)),
                                       "range_P": float(np.mean(P)), "session2_over_range": float(np.mean(oc ** 2) / np.mean(P)),
                                       "corr_gap_session": float(np.corrcoef(g, oc)[0, 1]), "n": int(ok.sum())}
        row["ratios_2+"] = next_vs_following(df, inera & (x >= 2))
        for lab, sel in [("2+", inera & (x >= 2)), ("0-1", inera & (x < 1))]:
            comp = {}
            for c, f in [("gap", np.square), ("oc", np.square), ("r", np.square), ("P", lambda v: v)]:
                a, b = f(lead(df[c].values, 1)), f(lead(df[c].values, 2))
                ok = sel & np.isfinite(a) & np.isfinite(b)
                comp[c] = boot({"a": a[ok], "b": b[ok]}, df.index[ok].to_period("M").astype(str), lambda s_: s_["a"] / s_["b"])
            row[f"component_ratios_{lab}"] = comp
        out[era] = row
        print(f"SPY {era}: " + " | ".join(f"{k}: r2 {v['close_to_close_r2']:.2f} gap2 {v['overnight_gap2']:.2f} "
                                         f"session2 {v['session2']:.2f} 2cov {v['2cov_gap_session']:+.2f} P {v['range_P']:.2f}"
                                         for k, v in row.items() if k.startswith("after 2+")), flush=True)
    return out


def stocks():
    """Stocks after their OWN big move (market residual) on a calm market day, earnings days excluded; 2005-2026."""
    import test12_brain_model as T12
    panel, m, _, E = T12.load_panel()
    hi = pd.read_csv(OHLC / "stocks_high.csv", index_col=0, parse_dates=True).reindex(index=panel.index, columns=panel.columns)
    lo = pd.read_csv(OHLC / "stocks_low.csv", index_col=0, parse_dates=True).reindex(index=panel.index, columns=panel.columns)
    keep = panel.index >= "2005-01-01"
    panel, m, E, hi, lo = panel.loc[keep], m.loc[keep], E[keep], hi.loc[keep], lo.loc[keep]
    xm = np.nan_to_num(shock_size(m.values), nan=9.0)
    Ed = pd.DataFrame(E, index=panel.index, columns=panel.columns)
    near_earn = (Ed.rolling(5, min_periods=1).max().shift(-2).fillna(0) > 0).values      # earnings within t-2..t+2
    with np.errstate(divide="ignore", invalid="ignore"):
        Pw = ((np.log(hi / lo) * 100) ** 2 / (4 * np.log(2))).where(hi > lo)
    parts = {"own": [], "earn": []}
    for j, sym in enumerate(panel.columns):
        r = panel[sym].values
        ok = np.isfinite(r)
        beta, alpha = np.polyfit(m.values[ok], r[ok], 1)
        e = r - alpha - beta * m.values
        x = shock_size(e, minp=18)
        df = pd.DataFrame({"r": r, "P": Pw[sym].values}, index=panel.index)
        # outcome uses the stock's raw close-to-close move and raw range (both include the market's part)
        r1, r2, P1, P2 = (lead(df[c].values, k) for c, k in (("r", 1), ("r", 2), ("P", 1), ("P", 2)))
        fin = np.isfinite(x) & np.isfinite(r1) & np.isfinite(r2) & np.isfinite(P1) & np.isfinite(P2)
        for key, s in [("own", fin & (x >= 2) & (xm < 1.5) & ~near_earn[:, j]), ("earn", fin & (x >= 2) & (E[:, j] > 0))]:
            parts[key].append(pd.DataFrame({"r1": r1[s] ** 2, "r2": r2[s] ** 2, "P1": P1[s], "P2": P2[s],
                                            "c_net": (np.abs(r1[s]) < np.abs(r2[s])).astype(float),
                                            "c_act": (P1[s] < P2[s]).astype(float), "one": 1.0,
                                            "era": np.where(panel.index[s] >= "2013-01-01", "2013-2026", "2005-2012"),
                                            "m": panel.index[s].to_period("M").astype(str)}))
    out = {}
    for key, frames in parts.items():
        ev = pd.concat(frames)
        A = {c: ev[c].values for c in ("r1", "r2", "P1", "P2")}
        out[key] = {"net_move": boot(A, ev["m"].values, lambda s: s["r1"] / s["r2"]),
                    "activity": boot(A, ev["m"].values, lambda s: s["P1"] / s["P2"]),
                    "efficiency": boot(A, ev["m"].values, lambda s: (s["r1"] / s["P1"]) / (s["r2"] / s["P2"])),
                    "calmer_net_move": boot({"c": ev["c_net"].values, "n": ev["one"].values}, ev["m"].values, lambda s: s["c"] / s["n"]),
                    "calmer_activity": boot({"c": ev["c_act"].values, "n": ev["one"].values}, ev["m"].values, lambda s: s["c"] / s["n"])}
        for era in ("2005-2012", "2013-2026"):
            e = ev[ev["era"] == era]
            out[f"{key}_{era}"] = {f"calmer_{nm}": boot({"c": e[c].values, "n": e["one"].values}, e["m"].values, lambda s: s["c"] / s["n"])
                                   for nm, c in (("net_move", "c_net"), ("activity", "c_act"))}
        print(f"stocks, {key} 2+ SD days (n={out[key]['net_move']['n']}): " + "  ".join(
            f"{k} {v['est']:.2f} ({v['lo']:.2f}-{v['hi']:.2f})" for k, v in out[key].items()), flush=True)
    return out


def intl():
    rows, pooled, nulls = [], [], {}
    for code in T14.NAMES:
        df = ohlc_frame(code)
        x = shock_size(df["r"].values)
        for era, (a, b) in ERAS_INTL.items():
            inera = np.isfinite(x) & (df.index <= f"{b}-12-31") & ((df.index >= a) if a else True)
            if inera.sum() < 1500:
                continue
            o = next_vs_following(df, inera & (x >= 2)) | {f"calmer_{k}": v for k, v in calmer(df, inera & (x >= 2)).items()}
            row = {"code": code, "era": era, "start": str(df.index[inera][0].date()), "n_events": o["net_move"]["n"]}
            for meas, v in o.items():
                row |= {meas: v["est"], f"{meas}_lo": v["lo"], f"{meas}_hi": v["hi"]}
            nl = calmer_null(df["r"].values[inera], seed=len(rows))
            nulls[(code, era)] = nl
            row |= {"calmer_null_median": float(np.median(nl)), "calmer_null_p05": float(np.quantile(nl, 0.05)),
                    "calmer_null_p95": float(np.quantile(nl, 0.95)), "calmer_null_share_ge_real": float(np.mean(nl >= row["calmer_net_move"]))}
            rows.append(row)
            r1, r2 = np.abs(lead(df["r"].values, 1)), np.abs(lead(df["r"].values, 2))
            sel = inera & (x >= 2) & np.isfinite(r2)
            pooled.append(pd.DataFrame({"era": era, "code": code, "c": (r1[sel] < r2[sel]).astype(float), "n": 1.0,
                                        "m": df.index[sel].to_period("M").astype(str)}))
        print(code, [(r["era"], round(r["calmer_net_move"], 3), round(r["calmer_null_median"], 3)) for r in rows if r["code"] == code], flush=True)
    pooled = pd.concat(pooled)
    pool = {}
    for era in ERAS_INTL:
        p = pooled[pooled["era"] == era]
        b = boot({"c": p["c"].values, "n": p["n"].values}, p["m"].values, lambda s_: s_["c"] / s_["n"])
        cells = [(c, e) for (c, e) in nulls if e == era]
        w = np.array([(p["code"] == c).sum() for c, _ in cells], float)
        nl = np.array([nulls[k] for k in cells])           # cells x sims: event-weighted pooled null per draw
        pn = (w[:, None] * nl).sum(0) / w.sum()
        pool[era] = {"n_markets": len(cells), "n_events": int(len(p)), "calmer_net_move": b,
                     "null_median": float(np.median(pn)), "null_p05": float(np.quantile(pn, 0.05)), "null_p95": float(np.quantile(pn, 0.95)),
                     "null_max": float(pn.max())}
        print(f"pooled {era}: calmer {b['est']:.3f} ({b['lo']:.3f}-{b['hi']:.3f}) vs null {pool[era]['null_median']:.3f} "
              f"({pool[era]['null_p05']:.3f}-{pool[era]['null_p95']:.3f})", flush=True)
    return pd.DataFrame(rows), pool


def main():
    res = {"us": us(), "spy": spy(), "stocks": stocks()}
    di, pool = intl()
    res["intl_pooled"] = pool
    di["name"] = di["code"].map(T14.NAMES)
    t15 = pd.read_csv(ROOT / "results" / "test15_score_driven.csv")
    t15 = t15[t15["name"] != "CRSP market"].rename(columns={"name": "code"})
    di = di.merge(t15[["code", "era", "sq_next_day", "betat_next_day", "betat_LR"]], on=["code", "era"], how="left")
    di.to_csv(ROOT / "results" / "test15b_range_intl.csv", index=False)
    summ = {}
    for era in ERAS_INTL:
        s = di[di["era"] == era]
        summ[era] = {"n_markets": len(s)}
        for meas in ("net_move", "activity", "efficiency"):        # ratios: 1 = flat, < 1 = notch at t+1
            summ[era] |= {f"{meas}_median": float(s[meas].median()), f"{meas}_n_below_1": int((s[meas] < 1).sum()),
                          f"{meas}_n_ci_below_1": int((s[f"{meas}_hi"] < 1).sum()), f"{meas}_n_ci_above_1": int((s[f"{meas}_lo"] > 1).sum())}
        for meas in ("calmer_net_move", "calmer_activity"):          # shares: 0.5 = flat, > 0.5 = notch at t+1
            summ[era] |= {f"{meas}_median": float(s[meas].median()), f"{meas}_n_above_half": int((s[meas] > 0.5).sum()),
                          f"{meas}_n_ci_above_half": int((s[f"{meas}_lo"] > 0.5).sum())}
        summ[era]["calmer_net_move_n_above_null_p95"] = int((s["calmer_net_move"] > s["calmer_null_p95"]).sum())
    ok = di.dropna(subset=["betat_next_day"])
    summ["across_cells_vs_model_dip"] = {"n_cells": len(ok)} | {
        f"{meas}_{kind}": float(ok[[meas, "betat_next_day"]].corr(kind).iloc[0, 1])
        for meas in ("net_move", "activity", "efficiency", "calmer_net_move", "calmer_activity") for kind in ("pearson", "spearman")}
    res["intl_summary"] = summ
    (ROOT / "results" / "test15b.json").write_text(json.dumps(res, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(di[["code", "era", "n_events", "net_move", "activity", "efficiency", "calmer_net_move", "calmer_activity",
              "calmer_null_median", "calmer_null_p95", "sq_next_day", "betat_next_day"]].round(3).to_string())
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
