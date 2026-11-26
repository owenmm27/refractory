"""Shared loaders and a small GJR-GARCH(1,1)-t implementation (no external deps beyond numpy/scipy/pandas)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import betaln, gammaln


# ----------------------------------------------------------------------------- loaders
def _french_section(path, header_marker):
    """Read one block of a Ken French CSV: rows after `header_marker` until the first blank line."""
    lines = Path(path).read_text(errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if header_marker in l) if header_marker else None
    if start is None:
        start = next(i for i, l in enumerate(lines) if l.startswith(",")) - 1
    hdr = start + 1
    cols = [c.strip() for c in lines[hdr].split(",")][1:]
    rows = []
    for l in lines[hdr + 1:]:
        if not l.strip() or not l.strip()[0].isdigit():
            break
        rows.append(l.split(","))
    arr = np.array(rows)
    idx = pd.to_datetime(arr[:, 0], format="%Y%m%d")
    df = pd.DataFrame(arr[:, 1:].astype(float), index=idx, columns=cols)
    return df


def load_industries(data_dir):
    """49 value-weighted industry portfolios, daily simple returns in decimal. Missing -> NaN."""
    df = _french_section(Path(data_dir) / "french" / "49_Industry_Portfolios_Daily.csv",
                         "Average Value Weighted Returns -- Daily")
    df = df.where(df > -99)
    return df / 100.0


def load_market(data_dir):
    """CRSP value-weighted market total return (Mkt-RF + RF), daily, decimal."""
    df = _french_section(Path(data_dir) / "french" / "F-F_Research_Data_Factors_daily.csv", None)
    return ((df["Mkt-RF"] + df["RF"]) / 100.0).rename("mkt")


def load_vix(data_dir):
    df = pd.read_csv(Path(data_dir) / "vix" / "VIX_History.csv")
    df.index = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    return df["CLOSE"].astype(float).rename("vix")


def load_stocks(data_dir):
    """Log returns for current S&P 500 members (survivorship-biased) and the index itself."""
    px = pd.read_csv(Path(data_dir) / "yahoo" / "sp500_adjclose.csv", index_col=0, parse_dates=True)
    px = px.where(px > 0)
    ret = np.log(px).diff()
    ret = ret.where(ret.abs() < 1.0)  # drop obvious data glitches (>170% daily moves)
    spx = ret.pop("^GSPC").rename("spx")
    return ret.iloc[1:], spx.iloc[1:]


# ----------------------------------------------------------------------------- GARCH
def gjr_variance(r, omega, alpha, gamma, beta, h0=None):
    """h_t = omega + (alpha + gamma*1[r_{t-1}<0]) r_{t-1}^2 + beta h_{t-1}, computed as an IIR filter."""
    r = np.asarray(r, float)
    if h0 is None:
        h0 = r.var()
    u = np.empty_like(r)
    u[0] = h0
    u[1:] = omega + (alpha + gamma * (r[:-1] < 0)) * r[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], u)


def _t_loglik(r, h, nu):
    # log G((nu+1)/2) - log G(nu/2) written as log G(1/2) - log B(nu/2, 1/2): identical, but stays accurate as
    # nu -> infinity (thin-tailed samples), where the difference of two huge gammaln values loses all precision
    return (gammaln(0.5) - betaln(nu / 2, 0.5) - 0.5 * np.log(np.pi * (nu - 2))
            - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(r ** 2 / (h * (nu - 2))))


def fit_gjr(r, asym=True):
    """Fit GJR-GARCH(1,1) with Student-t innovations by maximum likelihood.

    `r` should be demeaned returns in PERCENT. Returns dict with params, conditional variance h,
    and standardized residuals z = r / sqrt(h).
    """
    r = np.asarray(r, float)
    v = r.var()

    def unpack(p):
        omega = np.exp(p[0])
        alpha = 1 / (1 + np.exp(-p[1])) * 0.5
        gamma = (1 / (1 + np.exp(-p[2])) * 0.5) if asym else 0.0
        # persistence = alpha + gamma/2 + beta, constrained to (0, 0.9999)
        pers = 1 / (1 + np.exp(-p[3])) * 0.9999
        beta = max(pers - alpha - gamma / 2, 1e-6)
        nu = 2.05 + np.exp(p[4])
        return omega, alpha, gamma, beta, nu

    def nll(p):
        omega, alpha, gamma, beta, nu = unpack(p)
        h = gjr_variance(r, omega, alpha, gamma, beta, v)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -_t_loglik(r, h, nu).sum()

    best = None
    for a0, pers0 in [(0.05, 0.98), (0.10, 0.95), (0.03, 0.995)]:
        g0 = 0.05 if asym else 0.01
        p0 = np.array([np.log(v * (1 - pers0)), np.log(a0 / (0.5 - a0)), np.log(g0 / (0.5 - g0)),
                       np.log(pers0 / 0.9999 / (1 - pers0 / 0.9999)), np.log(6.0)])
        res = minimize(nll, p0, method="Nelder-Mead",
                       options={"maxiter": 4000, "maxfev": 6000, "xatol": 1e-5, "fatol": 1e-6})
        res = minimize(nll, res.x, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, gamma, beta, nu = unpack(best.x)
    h = gjr_variance(r, omega, alpha, gamma, beta, v)
    return dict(omega=omega, alpha=alpha, gamma=gamma, beta=beta, nu=nu,
                persistence=alpha + gamma / 2 + beta, loglik=-best.fun, h=h, z=r / np.sqrt(h))


def simulate_gjr(params, T, n_paths, rng, burn=1000):
    """Simulate n_paths independent GJR-GARCH-t series of length T (percent units, zero mean)."""
    omega, alpha, gamma, beta, nu = (np.asarray(params[k], float) for k in ("omega", "alpha", "gamma", "beta", "nu"))
    nu_b = np.broadcast_to(nu, (n_paths,))
    eps = rng.standard_t(nu_b, size=(T + burn, n_paths)) * np.sqrt((nu_b - 2) / nu_b)
    pers = alpha + gamma / 2 + beta
    h = np.broadcast_to(omega / np.maximum(1 - pers, 1e-4), (n_paths,)).copy()
    out = np.empty((T + burn, n_paths))
    hs = np.empty((T + burn, n_paths))
    for t in range(T + burn):
        hs[t] = h
        r = np.sqrt(h) * eps[t]
        out[t] = r
        h = omega + (alpha + gamma * (r < 0)) * r ** 2 + beta * h
    return out[burn:], hs[burn:]


# ----------------------------------------------------------------------------- K-timescale volatility model
# A bank of K leaky integrators ("units") all fed by yesterday's squared shock, with an asymmetric
# input (down moves count 1+delta times as much):
#     u_t   = r_{t-1}^2 * (1 + delta * 1[r_{t-1} < 0]) / (1 + delta / 2)
#     c_k,t = a_k * u_t + b_k * c_k,t-1          (unit k decays at rate b_k)
#     h_t   = omega + sum_k c_k,t
# Unit k's loading is w_k = a_k / (1 - b_k); W = sum_k w_k is the loop gain (stationary iff W < 1).
# K = 1 is a GJR-style GARCH(1,1); K = 2 is a fast + slow (component) volatility model.
def multi_components(r, omega, a, b, delta, h0=None):
    from scipy.signal import lfilter
    r = np.asarray(r, float)
    v = r.var() if h0 is None else h0
    u = np.empty_like(r)
    u[0] = v
    u[1:] = r[:-1] ** 2 * (1 + delta * (r[:-1] < 0)) / (1 + delta / 2)
    comps = []
    for ak, bk in zip(a, b):
        wk = ak / (1 - bk)
        y, _ = lfilter([ak], [1.0, -bk], u, zi=[bk * wk * v])
        comps.append(y)
    return np.array(comps)


def multi_variance(r, omega, a, b, delta, h0=None):
    return omega + multi_components(r, omega, a, b, delta, h0).sum(0)


def _unpack_multi(p, K):
    sig = lambda x: 1 / (1 + np.exp(-x))
    omega = np.exp(p[0])
    b = []
    prev = 0.0
    for k in range(K):
        lo = prev
        bk = lo + (0.99995 - lo) * sig(p[1 + k])
        b.append(bk)
        prev = bk
    b = np.array(b)
    W = sig(p[1 + K]) * 0.9999
    logits = np.concatenate([[0.0], p[2 + K:2 + K + (K - 1)]])
    share = np.exp(logits - logits.max())
    share /= share.sum()
    w = W * share
    a = w * (1 - b)
    delta = np.exp(p[1 + K + K])
    # cap nu at ~1000 (already indistinguishable from normal): beyond that the residual term underflows
    nu = 2.05 + np.exp(np.minimum(p[2 + K + K], np.log(1000.0)))
    return omega, a, b, delta, nu


def fit_multi(r, K=2, starts=None):
    """ML fit of the K-timescale model with Student-t shocks. r: demeaned returns in percent."""
    r = np.asarray(r, float)
    v = r.var()

    def nll(p):
        omega, a, b, delta, nu = _unpack_multi(p, K)
        h = multi_variance(r, omega, a, b, delta, v)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        return -_t_loglik(r, h, nu).sum()

    logit = lambda x: np.log(x / (1 - x))
    if starts is None:
        starts = {1: [(0.90,), (0.95,), (0.80,)],
                  2: [(0.80, 0.99), (0.60, 0.98), (0.90, 0.997)],
                  3: [(0.50, 0.93, 0.995), (0.30, 0.85, 0.99), (0.70, 0.97, 0.999)]}[K]
    best = None
    for bs in starts:
        pb, prev = [], 0.0
        for bk in bs:
            pb.append(logit((bk - prev) / (0.99995 - prev)))
            prev = bk
        W0 = 0.985
        p0 = np.concatenate([[np.log(v * (1 - W0))], pb, [logit(W0 / 0.9999)], np.zeros(K - 1), [np.log(0.5)], [np.log(5.0)]])
        res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 3000 * K, "maxfev": 4000 * K, "xatol": 1e-5, "fatol": 1e-6})
        res = minimize(nll, res.x, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    omega, a, b, delta, nu = _unpack_multi(best.x, K)
    comps = multi_components(r, omega, a, b, delta, v)
    h = omega + comps.sum(0)
    M = np.diag(b) + np.outer(a, np.ones(K))
    eig = np.sort(np.abs(np.linalg.eigvals(M)))
    return dict(K=K, omega=omega, a=a, b=b, delta=delta, nu=nu, w=a / (1 - b), W=float((a / (1 - b)).sum()),
                eigs=eig, half_lives=np.log(0.5) / np.log(eig), loglik=-best.fun, n_params=1 + 2 * K + 2,
                h=h, comps=comps, z=r / np.sqrt(h), params_vec=best.x)


def multi_forecast(fit, r, horizon):
    """Sum of expected variances over the next `horizon` days, made at each date t (uses r up to t)."""
    a, b, omega = fit["a"], fit["b"], fit["omega"]
    K = len(a)
    r = np.asarray(r, float)
    rr = np.append(r, 0.0)
    comps = multi_components(rr, omega, a, b, fit["delta"], fit.get("h0"))  # comps[:, t+1] known at t
    c = comps[:, 1:].T.copy()  # row t: state for day t+1
    M = np.diag(b) + np.outer(a, np.ones(K))
    total = np.zeros(len(r))
    for _ in range(horizon):
        total += omega + c.sum(1)
        c = c @ M.T + a * omega
    return total


def simulate_multi(params, T, n_paths, rng, burn=1500):
    """Simulate the K-timescale model. params: dict with omega (n,), a (n,K), b (n,K), delta (n,), nu (n,)."""
    omega = np.broadcast_to(np.asarray(params["omega"], float), (n_paths,))
    a = np.broadcast_to(np.atleast_2d(params["a"]), (n_paths, np.atleast_2d(params["a"]).shape[1]))
    b = np.broadcast_to(np.atleast_2d(params["b"]), a.shape)
    delta = np.broadcast_to(np.asarray(params["delta"], float), (n_paths,))
    nu = np.broadcast_to(np.asarray(params["nu"], float), (n_paths,))
    W = (a / (1 - b)).sum(1)
    hbar = omega / np.maximum(1 - W, 1e-4)
    c = (a / (1 - b)) * hbar[:, None]
    eps = rng.standard_t(nu, size=(T + burn, n_paths)) * np.sqrt((nu - 2) / nu)
    out = np.empty((T + burn, n_paths))
    for t in range(T + burn):
        h = omega + c.sum(1)
        x = np.sqrt(h) * eps[t]
        out[t] = x
        u = x ** 2 * (1 + delta * (x < 0)) / (1 + delta / 2)
        c = a * u[:, None] + b * c
    return out[burn:]


# ----------------------------------------------------------------------------- power-input variant
# Same bank of leaky integrators, but driven by |r|^d instead of r^2, and the state sets sigma^d:
#     u_t = |r_{t-1}|^d * asym,   c_k,t = a_k u_t + b_k c_k,t-1,   sigma_t^d = omega + sum_k c_k,t
# d = 2 recovers the model above; d < 2 makes the response to big shocks saturate (concave in |r|).
def power_variance(r, omega, a, b, delta, d, h0=None):
    from scipy.signal import lfilter
    r = np.asarray(r, float)
    v = (r.var() if h0 is None else h0) ** (d / 2)
    u = np.empty_like(r)
    u[0] = v
    u[1:] = np.abs(r[:-1]) ** d * (1 + delta * (r[:-1] < 0)) / (1 + delta / 2)
    y = np.full(len(r), omega)
    for ak, bk in zip(a, b):
        wk = ak / (1 - bk)
        y = y + lfilter([ak], [1.0, -bk], u, zi=[bk * wk * v])[0]
    return np.maximum(y, 1e-12) ** (2 / d)


def fit_power(r, K=2, d_fixed=None):
    r = np.asarray(r, float)
    v = r.var()
    sig = lambda x: 1 / (1 + np.exp(-x))

    def unpack(p):
        omega, a, b, delta, nu = _unpack_multi(p[:-1] if d_fixed is None else p, K)
        d = d_fixed if d_fixed is not None else 0.5 + 2.0 * sig(p[-1])
        return omega, a, b, delta, nu, d

    def nll(p):
        omega, a, b, delta, nu, d = unpack(p)
        h = power_variance(r, omega, a, b, delta, d, v)
        if not np.all(np.isfinite(h)) or h.min() <= 0:
            return 1e12
        # Student-t log-likelihood written out here to avoid a forward reference
        return -(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
                 - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(r ** 2 / (h * (nu - 2)))).sum()

    logit = lambda x: np.log(x / (1 - x))
    best = None
    bstarts = {1: [(0.9,), (0.95,)], 2: [(0.80, 0.99), (0.60, 0.985), (0.9, 0.997)]}[K]
    dstarts = [1.0, 1.5] if d_fixed is None else [d_fixed]
    for bs in bstarts:
        for d0 in dstarts:
            pb, prev = [], 0.0
            for bk in bs:
                pb.append(logit((bk - prev) / (0.99995 - prev)))
                prev = bk
            W0 = 0.97
            p0 = [np.log(v ** (d0 / 2) * (1 - W0))] + pb + [logit(W0 / 0.9999)] + [0.0] * (K - 1) + [np.log(0.5), np.log(5.0)]
            if d_fixed is None:
                p0 = p0 + [logit((d0 - 0.5) / 2.0)]
            res = minimize(nll, np.array(p0), method="Nelder-Mead", options={"maxiter": 5000, "maxfev": 7000, "xatol": 1e-5, "fatol": 1e-6})
            res = minimize(nll, res.x, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
    omega, a, b, delta, nu, d = unpack(best.x)
    h = power_variance(r, omega, a, b, delta, d, v)
    M = np.diag(b) + np.outer(a, np.ones(K))
    eig = np.sort(np.abs(np.linalg.eigvals(M)))
    return dict(K=K, omega=omega, a=a, b=b, delta=delta, nu=nu, d=d, loglik=-best.fun, h=h, z=r / np.sqrt(h),
                half_lives=np.log(0.5) / np.log(eig), n_params=1 + 2 * K + 2 + (d_fixed is None))


def simulate_power(params, T, rng, burn=1500):
    omega, a, b, delta, nu, d = (params[k] for k in ("omega", "a", "b", "delta", "nu", "d"))
    a, b = np.asarray(a), np.asarray(b)
    W = (a / (1 - b)).sum()
    c = (a / (1 - b)) * omega / max(1 - W, 1e-4)
    eps = rng.standard_t(nu, T + burn) * np.sqrt((nu - 2) / nu)
    out = np.empty(T + burn)
    for t in range(T + burn):
        sd = max(omega + c.sum(), 1e-12) ** (1 / d)
        x = sd * eps[t]
        out[t] = x
        c = a * (abs(x) ** d * (1 + delta * (x < 0)) / (1 + delta / 2)) + b * c
    return out[burn:]


# ----------------------------------------------------------------------------- small stats helpers
def acf(x, nlags):
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = (x ** 2).sum()
    return np.array([1.0] + [(x[k:] * x[:-k]).sum() / denom for k in range(1, nlags + 1)])


def ljung_box(x, lags):
    from scipy.stats import chi2
    n = len(x)
    rho = acf(x, lags)[1:]
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return q, chi2.sf(q, lags)


def hill(x, frac=0.025):
    """Hill tail-index estimate using the largest `frac` of |x|."""
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    k = max(int(len(a) * frac), 10)
    return 1.0 / np.mean(np.log(a[:k] / a[k]))
