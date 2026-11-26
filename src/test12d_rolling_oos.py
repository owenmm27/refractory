"""Test 12d - Do the market unit's pieces help forecasting when re-estimated as data arrive?

Test 12b froze every parameter at its 2000-2012 value and found the refractory and reversal pieces HURT the
2013-2026 forecasts, while Test 10 showed the refractory effect exists in every era but at different strengths.
Here each year Y = 2013..2026 is forecast one day ahead with parameters fitted on 2000..Y-1 (expanding window),
so a drifting strength can be tracked. Models: two-unit state; + refractory; + refractory + volatility-keyed reversal.

Usage: python3 -I src/test12d_rolling_oos.py <project_root>
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import test12_brain_model as B
import test12c_model_v2 as C
from test12b_oos import fit_market_vec, hac_t, mkt_path_fixed
from test9_scheduled_news import t_ll

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()


def year_job(args):
    year, dates, m = args
    ntr = int(np.searchsorted(dates, np.datetime64(f"{year}-01-01")))
    nte = int(np.searchsorted(dates, np.datetime64(f"{year + 1}-01-01")))
    mc = m[:nte] - m[:ntr].mean()
    out = {"year": year, "rho": None}
    for name, (refr, mean) in {"two_unit": (False, False), "refractory": (True, False)}.items():
        p, sbar, v = fit_market_vec(mc[:ntr], 2, refr, mean)
        e, h, nu, _ = mkt_path_fixed(p, mc, 2, refr, mean, sbar, v)
        out[name] = t_ll(e, h, nu)[ntr:nte]
        if refr:
            out["rho"] = float(p[7])
    f2 = C.fit_market2(mc[:ntr])
    lr = C.regime(mc)
    e2 = mc - B.mkt_mean(mc, f2["a1"], f2["a2"]) * np.exp(f2["c"] * np.minimum(lr - f2["Rbar"], f2["Rmax"]))
    d_ = f2["delta"]
    lvl = B.filt(np.r_[f2["vbar"], (e2 ** 2 * (1 + d_ * (e2 < 0)) / (1 + d_ / 2))[:-1]], f2["a"], f2["b"], f2["vbar"], f2["omega"])
    s2 = np.abs(e2) / np.sqrt(lvl)
    h2 = lvl * np.exp(-f2["rho"] * (np.r_[f2["sbar"], s2[:-1]] - f2["sbar"]))
    out["refractory_plus_reversal_v2"] = t_ll(e2, h2, f2["nu"])[ntr:nte]
    out["c"] = f2["c"]
    return out


def main():
    panel, m_s, _, _ = B.load_panel()
    dates, m = panel.index.values, m_s.values
    jobs = [(y, dates, m) for y in range(2013, 2027)]
    with ProcessPoolExecutor(max_workers=3) as ex:
        outs = list(ex.map(year_job, jobs))
    names = ["two_unit", "refractory", "refractory_plus_reversal_v2"]
    ll = {k: np.concatenate([o[k] for o in outs]) for k in names}
    res = {"loglik_test_2013_2026": {k: float(v.sum()) for k, v in ll.items()},
           "gain_vs_two_unit": {k: {"loglik_gain": float((ll[k] - ll["two_unit"]).sum()), "hac_t": hac_t(ll[k] - ll["two_unit"])}
                                for k in names[1:]},
           "by_year": {o["year"]: {"refractory_gain": float((o["refractory"] - o["two_unit"]).sum()),
                                   "v2_gain": float((o["refractory_plus_reversal_v2"] - o["two_unit"]).sum()),
                                   "rho_fitted": o["rho"], "c_fitted": o["c"]} for o in outs}}
    print(json.dumps(res, indent=1))
    (ROOT / "results" / "test12d.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
