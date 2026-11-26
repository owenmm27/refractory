"""Charts for Test 6 (two-timescale state). Usage: python3 -I src/figures_timescales.py <project_root>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figures as F  # reuse palette, rcParams and helpers (renders the base figures again, harmless)

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
RES = ROOT / "results"
FIG = RES / "figures"
INK, INK2, MUTED, GRID, BASE, SURF = F.INK, F.INK2, F.MUTED, F.GRID, F.BASE, F.SURF
BLUE, ORANGE = F.BLUE, F.ORANGE

# ----------------------------------------------------------------------------- fig 5: the two units over time
d = np.load(RES / "test6a.npz")
dates = pd.to_datetime(d["dates"])
ann = lambda v: np.sqrt(np.maximum(v, 0) * 252)          # daily %^2 variance -> annualized % vol
m = pd.DataFrame({"fast": d["fast"], "slow": d["slow"]}, index=dates).resample("W").mean()
fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.07, hspace=0.32)
for ax, col, c, lab in [(axes[0], "slow", BLUE, "Slow unit: half-life about a year (regimes)"),
                        (axes[1], "fast", ORANGE, "Fast unit: half-life about two weeks (bursts)")]:
    ax.plot(m.index, ann(m[col]), color=c, lw=1.4 if col == "slow" else 0.9)
    ax.set_title(lab, loc="left", fontsize=10.5, color=INK, pad=4)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50, 100], ["2%", "5%", "10%", "20%", "50%", "100%"])
    ax.set_ylim(1.5, 160)
for x, t in [("1932-06-01", "1930s"), ("1987-10-23", "1987"), ("2008-11-15", "2008"), ("2020-03-20", "2020")]:
    ts = pd.Timestamp(x)
    axes[1].annotate(t, (ts, ann(m["fast"].asof(ts))), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9, color=INK2)
axes[1].set_xlim(pd.Timestamp("1926-06-01"), pd.Timestamp("2027-06-01"))
fig.text(0.015, 0.43, "Volatility each unit contributes (annualized)", rotation=90, va="center", color=INK2, fontsize=10.5)
F.title(fig, "The market's volatility state is two units: a fast one and a slow one",
        "Fitted two-timescale model, US market 1926–2026, weekly averages. Both units are driven by the same daily shocks.")
fig.savefig(FIG / "fig5_two_units.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------- fig 6: closing the persistence gap
b6 = json.loads((RES / "test6b.json").read_text())["avalanche"]["K_threshold=4"]
c6 = json.loads((RES / "test6c.json").read_text())["stocks_population_unit"]["K_threshold=4"]["realvol_K2_plus_population"]
d6 = json.loads((RES / "test6d.json").read_text())["K_threshold=4"]
e6 = json.loads((RES / "test6e.json").read_text())
real = b6["real"]["lag1_slope"]
rows = [("Market factor + 1-unit stock volatility", b6["realvol_K1"]["lag1_slope"]),
        ("Market factor + 2-unit stock volatility", b6["realvol_K2"]["lag1_slope"]),
        ("2-unit + population feedback unit", c6["lag1_slope"]),
        ("2-unit + real sector factors (11)", d6["lag1_slope"]),
        ("2-unit + top 3 statistical factors", {"mean": e6["k=3"]["lag1_mean"], "p2.5": e6["k=3"]["p2.5"], "p97.5": e6["k=3"]["p97.5"]}),
        ("2-unit + top 10 statistical factors", {"mean": e6["k=10"]["lag1_mean"], "p2.5": e6["k=10"]["p2.5"], "p97.5": e6["k=10"]["p97.5"]}),
        ("2-unit + top 30 statistical factors", {"mean": e6["k=30"]["lag1_mean"], "p2.5": e6["k=30"]["p2.5"], "p97.5": e6["k=30"]["p97.5"]})]
fig, ax = plt.subplots(figsize=(9.5, 4.9))
fig.subplots_adjust(left=0.38, right=0.95, top=0.78, bottom=0.13)
ax.axvline(real, color=INK, lw=1.4, zorder=2)
ax.text(real - 0.002, -0.75, "real stocks", color=INK, fontsize=9.5, ha="right", va="center")
for y, (lab, v) in enumerate(rows):
    ax.plot([v["p2.5"], v["p97.5"]], [y, y], color=BLUE, lw=3, solid_capstyle="round", zorder=3)
    ax.plot([v["mean"]], [y], "o", ms=6.5, color=BLUE, mec=SURF, mew=1.2, zorder=4)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_ylim(len(rows) - 0.4, -1.0)
ax.set_xlim(0.42, 0.55)
ax.set_xlabel("Day-to-day persistence of collective firing (lag-1 slope; 95% range over 20 runs)")
ax.grid(axis="y", visible=False)
F.title(fig, "No single hidden unit explains the extra persistence; it is spread across many factors",
        "237 stocks, 1990–2026, firing threshold 4 SD. All nulls keep the real market factor; lower rows add more real structure.")
fig.savefig(FIG / "fig6_leftover_persistence.png", dpi=200)
plt.close(fig)
print("saved fig5, fig6")
