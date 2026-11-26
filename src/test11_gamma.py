"""Test 11 - Does dealer gamma create the index's refractory period?

Test 10 put the refractory period at the index level. A leading candidate is index-option hedging: when dealers
are long gamma they trade against price moves, which should quiet the index. If that is the mechanism, the
refractory damping after a big surprise should be stronger when dealers are long gamma.

Model (2-unit volatility state, Student-t; S&P 500, May 2011 - Nov 2026):
    log h_t = log level_t - (rho0 + rho_g*g + rho_v*v + rho_dn*dn) * (s_{t-1} - mean s) - gam_g*g - gam_v*v
    g  = GEX (SqueezeMetrics dealer-gamma estimate) at the close of t-1, as a rolling 252-day percentile, centered
    v  = log VIX at the close of t-1, centered      dn = 1 if yesterday's return was negative, centered
The gam terms let gamma and VIX shift the variance level on all days, so the rho terms measure only the change
in the day-after-a-surprise damping. Robustness: gamma at t-2 (before the surprise), the negative-gamma sign,
sub-periods. Model-light: next-day z^2 after a big surprise by gamma / VIX group, for the index and (placebo)
stocks' idiosyncratic moves.

Usage: python3 -I src/test11_gamma.py <project_root>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.signal import lfilter
from statsmodels.tools.numdiff import approx_hess3

from common import _unpack_multi, fit_multi, load_stocks, load_vix
from test9_scheduled_news import t_ll

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
OUT = ROOT / "results"
rng = np.random.default_rng(11)


# ----------------------------------------------------------------------------- model
def path(p, r, XI, XL):
    omega, a, b, delta, nu = _unpack_multi(p[:7], 2)
    kI, kL = XI.shape[1], XL.shape[1]
    rho0, rI, gL = p[7], p[8:8 + kI], p[8 + kI:8 + kI + kL]
    v = np.mean(r ** 2)
    sq = r ** 2 * (1 + delta * (r < 0)) / (1 + delta / 2)
    u = np.r_[v, sq[:-1]]
    level = np.full(len(r), omega)
    for ak, bk in zip(a, b):
        level = level + lfilter([ak], [1.0, -bk], u, zi=[bk * ak / (1 - bk) * v])[0]
    s = np.abs(r) / np.sqrt(level)
    sm_ = s.mean()
    ds = np.r_[0.0, s[:-1] - sm_]
    h = level * np.exp(-(rho0 + XI @ rI) * ds - XL @ gL)
    return h, nu, sm_


class Model:
    def __init__(self, r, XI, XL, names_I, names_L):
        self.r, self.XI, self.XL = r, XI, XL
        self.names = ["rho0"] + [f"rho_{n}" for n in names_I] + [f"gam_{n}" for n in names_L]

    def nll_full(self, p):
        h, nu, _ = path(p, self.r, self.XI, self.XL)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -t_ll(self.r, h, nu).sum()

    def fit(self, active, p_start):
        """active: names of free extra parameters (others fixed at 0). p_start: full vector to start from."""
        act = np.r_[np.ones(7, bool), [n in active for n in self.names]]
        p_full = p_start.copy()
        p_full[~act] = 0.0

        def nll(q):
            pp = p_full.copy()
            pp[act] = q
            return self.nll_full(pp)
        res = minimize(nll, p_full[act], method="L-BFGS-B")
        p_full[act] = res.x
        out = {"ll": float(-res.fun), "p": p_full, "active": list(active)}
        # Wald SEs for the extra parameters from the numerical Hessian
        try:
            H = approx_hess3(res.x, nll)
            se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))[7:]
        except Exception:
            se = np.full(act.sum() - 7, np.nan)
        extra = [n for n in self.names if n in active]
        out["coef"] = {n: float(p_full[7 + self.names.index(n)]) for n in extra}
        out["se"] = {n: float(s) for n, s in zip(extra, se)}
        out["s_mean"] = float(path(p_full, self.r, self.XI, self.XL)[2])
        return out


def base_start(r, n_extra):
    f = fit_multi(r, 2)
    return np.r_[np.asarray(f["params_vec"], float), np.zeros(n_extra)]


# ----------------------------------------------------------------------------- data
def build():
    _, spx = load_stocks(ROOT / "data")
    gx = pd.read_csv(ROOT / "data" / "gex" / "DIX.csv", parse_dates=["date"]).set_index("date")["gex"]
    vix = load_vix(ROOT / "data")
    df = pd.DataFrame({"r": spx * 100}).dropna()
    gpct = gx.rolling(252, min_periods=120).rank(pct=True)
    df["g1"] = gpct.shift(1).reindex(df.index)                  # gamma at the previous close
    df["g2"] = gpct.shift(2).reindex(df.index)                  # gamma the close before (pre-surprise)
    df["neg1"] = (gx.shift(1) < 0).astype(float).reindex(df.index)
    df["v1"] = np.log(vix).shift(1).reindex(df.index)
    df = df.dropna()
    df["dn1"] = (df["r"].shift(1) < 0).astype(float)
    df = df.iloc[1:]
    df["r"] = df["r"] - df["r"].mean()
    return df


def cent(x):
    return x - x.mean()


def implied(fit, gq, vq=0.0):
    """Next-day variance after a 3-SD surprise (vs the same day with no surprise), at centered gamma gq, VIX vq."""
    c = fit["coef"]
    rho = c.get("rho0", 0) + c.get("rho_g", 0) * gq + c.get("rho_v", 0) * vq
    return float(np.exp(-rho * (3 - fit["s_mean"])))


def run_models(df, gcol="g1", label=""):
    r = df["r"].values
    g, v, dn = cent(df[gcol].values), cent(df["v1"].values), cent(df["dn1"].values)
    XI = np.column_stack([g, v, dn])
    XL = np.column_stack([g, v])
    M = Model(r, XI, XL, ["g", "v", "dn"], ["g", "v"])
    p0 = base_start(r, len(M.names))
    fits = {"M0_base": M.fit([], p0)}
    fits["M1_refractory"] = M.fit(["rho0"], fits["M0_base"]["p"] + np.r_[np.zeros(7), 0.2, np.zeros(len(M.names) - 1)])
    fits["M2_plus_level"] = M.fit(["rho0", "gam_g", "gam_v"], fits["M1_refractory"]["p"])
    fits["M3_plus_gamma_interaction"] = M.fit(["rho0", "gam_g", "gam_v", "rho_g"], fits["M2_plus_level"]["p"])
    fits["M4_full"] = M.fit(["rho0", "gam_g", "gam_v", "rho_g", "rho_v", "rho_dn"], fits["M3_plus_gamma_interaction"]["p"])
    fits["M4_minus_rho_g"] = M.fit(["rho0", "gam_g", "gam_v", "rho_v", "rho_dn"], fits["M4_full"]["p"])
    fits["M4_minus_rho_v"] = M.fit(["rho0", "gam_g", "gam_v", "rho_g", "rho_dn"], fits["M4_full"]["p"])
    ll = {k: f["ll"] for k, f in fits.items()}
    gq = np.quantile(df[gcol], [0.1, 0.9]) - df[gcol].mean()
    vq = np.quantile(df["v1"], [0.1, 0.9]) - df["v1"].mean()
    out = {"n": len(r), "label": label,
           "fits": {k: {"ll": f["ll"], "coef": f["coef"], "se": f["se"]} for k, f in fits.items()},
           "LR": {"refractory_vs_base": 2 * (ll["M1_refractory"] - ll["M0_base"]),
                  "gamma_interaction_alone (M3 vs M2)": 2 * (ll["M3_plus_gamma_interaction"] - ll["M2_plus_level"]),
                  "gamma_interaction_given_VIX_and_sign (M4 vs M4-rho_g)": 2 * (ll["M4_full"] - ll["M4_minus_rho_g"]),
                  "VIX_interaction_given_gamma (M4 vs M4-rho_v)": 2 * (ll["M4_full"] - ll["M4_minus_rho_v"]),
                  "gamma_and_VIX_levels (M2 vs M1)": 2 * (ll["M2_plus_level"] - ll["M1_refractory"])},
           "corr_gamma_logVIX": float(np.corrcoef(df[gcol], df["v1"])[0, 1]),
           "next_day_after_3sd": {
               "M3_low_gamma_p10": implied(fits["M3_plus_gamma_interaction"], gq[0]),
               "M3_high_gamma_p90": implied(fits["M3_plus_gamma_interaction"], gq[1]),
               "M4_low_gamma_p10": implied(fits["M4_full"], gq[0]), "M4_high_gamma_p90": implied(fits["M4_full"], gq[1]),
               "M4_low_VIX_p10": implied(fits["M4_full"], 0, vq[0]), "M4_high_VIX_p90": implied(fits["M4_full"], 0, vq[1]),
               "M1_average": implied(fits["M1_refractory"], 0)}}
    print(label, json.dumps({"LR": out["LR"], "next": out["next_day_after_3sd"],
                             "M4": out["fits"]["M4_full"]["coef"], "M4_se": out["fits"]["M4_full"]["se"]}, default=float), flush=True)
    return out, fits["M0_base"]


# ----------------------------------------------------------------------------- model-light
def boot_group_means(y, groups, valid, labels, n_boot=1000, block=21):
    T = len(y)
    nb = int(np.ceil(T / block))
    starts = np.arange(0, T - block + 1)
    point = {lab: float(np.mean(y[valid & (groups == lab)])) for lab in labels}
    n = {lab: int(np.sum(valid & (groups == lab))) for lab in labels}
    B = {lab: [] for lab in labels}
    diff = []
    for _ in range(n_boot):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]
        yy, gg, vv = y[idx], groups[idx], valid[idx]
        m = {lab: np.mean(yy[vv & (gg == lab)]) if np.any(vv & (gg == lab)) else np.nan for lab in labels}
        for lab in labels:
            B[lab].append(m[lab])
        diff.append(m[labels[-1]] - m[labels[0]])
    ci = {lab: [float(np.nanpercentile(B[lab], 2.5)), float(np.nanpercentile(B[lab], 97.5))] for lab in labels}
    d = np.array(diff)
    return {"mean": point, "ci95": ci, "n": n,
            f"{labels[-1]}_minus_{labels[0]}": {"point": point[labels[-1]] - point[labels[0]],
                                                  "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}}


def boot_ratio(y, groups, big, calm, labels, n_boot=1000, block=21):
    """Refractory ratio per group: mean next-day z^2 after a big day / after a calm day (nets out level effects)."""
    T = len(y)
    nb = int(np.ceil(T / block))
    starts = np.arange(0, T - block + 1)

    def rat(yy, gg, bb, cc):
        return {lab: np.mean(yy[bb & (gg == lab)]) / np.mean(yy[cc & (gg == lab)]) for lab in labels}
    point = rat(y, groups, big, calm)
    B = []
    for _ in range(n_boot):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]
        B.append(rat(y[idx], groups[idx], big[idx], calm[idx]))
    ci = {lab: [float(np.nanpercentile([b[lab] for b in B], 2.5)), float(np.nanpercentile([b[lab] for b in B], 97.5))] for lab in labels}
    d = np.array([b[labels[-1]] - b[labels[0]] for b in B])
    return {"ratio": {k: float(v) for k, v in point.items()}, "ci95": ci,
            f"{labels[-1]}_minus_{labels[0]}": {"point": float(point[labels[-1]] - point[labels[0]]),
                                                  "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}}


def model_light(df, z, thr=1.0):
    z2 = z ** 2
    prev_big = np.r_[False, np.abs(z[:-1]) > thr]
    prev_calm = np.r_[False, np.abs(z[:-1]) <= thr]
    gt = pd.qcut(df["g1"], 3, labels=["low", "mid", "high"]).astype(str).values
    vt = pd.qcut(df["v1"], 3, labels=["low", "mid", "high"]).astype(str).values
    gm = np.where(df["g1"] > df["g1"].median(), "high", "low")
    vm = np.where(df["v1"] > df["v1"].median(), "highVIX", "lowVIX")
    out = {"threshold_abs_z_yesterday": thr,
           "after_big_by_gamma_tercile": boot_group_means(z2, gt, prev_big, ["low", "mid", "high"]),
           "after_calm_by_gamma_tercile": boot_group_means(z2, gt, prev_calm, ["low", "mid", "high"]),
           "after_big_by_VIX_tercile": boot_group_means(z2, vt, prev_big, ["low", "mid", "high"]),
           "after_calm_by_VIX_tercile": boot_group_means(z2, vt, prev_calm, ["low", "mid", "high"]),
           "ratio_big_over_calm_by_gamma_tercile": boot_ratio(z2, gt, prev_big, prev_calm, ["low", "mid", "high"]),
           "ratio_big_over_calm_by_VIX_tercile": boot_ratio(z2, vt, prev_big, prev_calm, ["low", "mid", "high"])}
    for vlab in ("lowVIX", "highVIX"):          # disentangle: gamma within VIX halves, VIX within gamma halves
        sel = vm == vlab
        out[f"ratio_by_gamma_half_within_{vlab}"] = boot_ratio(z2, np.where(sel, gm, "x"), prev_big, prev_calm, ["low", "high"])
    for glab in ("low", "high"):
        sel = gm == glab
        out[f"ratio_by_VIX_half_within_{glab}_gamma"] = boot_ratio(z2, np.where(sel, vm, "x"), prev_big, prev_calm, ["lowVIX", "highVIX"])
    for vlab in ("lowVIX", "highVIX"):
        out[f"after_big_by_gamma_half_within_{vlab}"] = boot_group_means(z2, gm, prev_big & (vm == vlab), ["low", "high"])
        out[f"after_calm_by_gamma_half_within_{vlab}"] = boot_group_means(z2, gm, prev_calm & (vm == vlab), ["low", "high"])
    # HAC regression on days after a big surprise: z_t^2 on gamma percentile and log VIX (standardized)
    sel = prev_big
    X = sm.add_constant(np.column_stack([(df["g1"] - df["g1"].mean()) / df["g1"].std(),
                                         (df["v1"] - df["v1"].mean()) / df["v1"].std()])[sel])
    reg = sm.OLS(z2[sel], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out["reg_after_big_z2_on_gamma_and_VIX_hac"] = {"coef": dict(zip(["const", "gamma_sd", "logVIX_sd"], reg.params.tolist())),
                                                     "t": dict(zip(["const", "gamma_sd", "logVIX_sd"], reg.tvalues.tolist())),
                                                     "n": int(sel.sum())}
    return out


def stock_placebo(df, thr=1.0):
    """Index vs stocks' idiosyncratic next-day z^2 after a big index day, by gamma tercile (Test 10 base residuals)."""
    d = np.load(OUT / "cache" / "test10_z.npz", allow_pickle=True)   # our own cache file
    dates = pd.DatetimeIndex(d["dates_2000"])
    syms = list(d["symbols"])
    zi = pd.Series(d["raw|INDEX ^GSPC"].astype(float), index=dates)
    Zs = pd.DataFrame(np.column_stack([d[f"idio|STOCK {s}"] for s in syms]).astype(float), index=dates)
    common = df.index.intersection(dates)
    zi, Zs, g = zi.loc[common].values, Zs.loc[common].values, df.loc[common, "g1"]
    gt = pd.qcut(g, 3, labels=["low", "mid", "high"]).astype(str).values
    prev_big = np.r_[False, np.abs(zi[:-1]) > thr]
    valid = Zs != 0
    idio_day = np.where(valid, Zs ** 2, 0).sum(1) / np.maximum(valid.sum(1), 1)   # cross-stock mean z^2 each day
    vt = pd.qcut(df.loc[common, "v1"], 3, labels=["low", "mid", "high"]).astype(str).values
    calm = np.r_[False, np.abs(zi[:-1]) <= thr]
    L = ["low", "mid", "high"]
    return {"index_ratio_by_gamma": boot_ratio(zi ** 2, gt, prev_big, calm, L),
            "index_ratio_by_VIX": boot_ratio(zi ** 2, vt, prev_big, calm, L),
            "idio_ratio_by_gamma": boot_ratio(idio_day, gt, prev_big, calm, L),
            "index_after_calm_by_gamma": boot_group_means(zi ** 2, gt, calm, L),
            "idio_after_calm_by_gamma": boot_group_means(idio_day, gt, calm, L),
            "index_after_big_by_gamma": boot_group_means(zi ** 2, gt, prev_big, ["low", "mid", "high"]),
            "idio_after_big_index_day_by_gamma": boot_group_means(idio_day, gt, prev_big, ["low", "mid", "high"]),
            "idio_after_calm_index_day_by_gamma": boot_group_means(idio_day, gt, ~prev_big, ["low", "mid", "high"])}


def main():
    df = build()
    res = {"sample": [str(df.index[0].date()), str(df.index[-1].date())], "n": len(df),
           "share_neg_gamma_days": float(df["neg1"].mean())}
    main_out, base = run_models(df, "g1", "main: gamma at t-1")
    res["main"] = main_out
    res["robust_gamma_t_minus_2"] = run_models(df, "g2", "gamma at t-2")[0]
    df_neg = df.copy()
    df_neg["gneg"] = 1 - df_neg["neg1"]          # 1 = long gamma, 0 = short gamma (sign only)
    res["robust_gamma_sign"] = run_models(df_neg, "gneg", "gamma sign")[0]
    for a, b in [("2011", "2019"), ("2020", "2026")]:
        res[f"subperiod_{a}_{b}"] = run_models(df.loc[a:b], "g1", f"{a}-{b}")[0]
    z = df["r"].values / np.sqrt(path(base["p"], df["r"].values, np.zeros((len(df), 0)), np.zeros((len(df), 0)))[0])
    for thr in (1.0, 2.0):
        res[f"model_light_thr{thr:g}"] = model_light(df, z, thr)
    ml = res["model_light_thr1"]
    for k in ("after_big_by_gamma_tercile", "after_calm_by_gamma_tercile", "after_big_by_VIX_tercile",
              "after_big_by_gamma_half_within_lowVIX", "after_big_by_gamma_half_within_highVIX"):
        print(k, {g: round(v, 3) for g, v in ml[k]["mean"].items()}, ml[k]["n"], flush=True)
    print("reg", ml["reg_after_big_z2_on_gamma_and_VIX_hac"], flush=True)
    res["stock_placebo_thr1"] = stock_placebo(df, 1.0)
    for k, v in res["stock_placebo_thr1"].items():
        vals = v.get("mean", v.get("ratio"))
        print(k, {g: round(x, 3) for g, x in vals.items()}, {g: [round(c, 2) for c in x] for g, x in v["ci95"].items()}, flush=True)
    for k in ("ratio_big_over_calm_by_gamma_tercile", "ratio_big_over_calm_by_VIX_tercile"):
        v = ml[k]
        print("2011-fit", k, {g: round(x, 3) for g, x in v["ratio"].items()}, {g: [round(c, 2) for c in x] for g, x in v["ci95"].items()}, flush=True)
    (OUT / "test11.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
