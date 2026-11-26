"""Charts for the four tests. Usage: python3 -I src/figures.py <project_root>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_market, load_stocks

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# reference palette (light mode), validated: slots 1-3 pass all checks; aqua needs direct labels (done below)
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"], "font.size": 10.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
})


def title(fig, main, sub):
    fig.text(0.06, 0.965, main, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.905, sub, fontsize=10, color=INK2, va="top")


def label_end(ax, x, y, text, color, dy=0, dx=4):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK2, fontsize=9.5, va="center",
                path_effects=None)
    ax.plot([x], [y], "o", ms=4, color=color, mec=SURF, mew=1.2, zorder=5)


# ----------------------------------------------------------------------------- fig 1: memory
d = np.load(RES / "test1.npz")
lags = np.arange(1, 251)
fig, ax = plt.subplots(figsize=(8.4, 4.8))
fig.subplots_adjust(left=0.08, right=0.8, top=0.8, bottom=0.12)
band = 1.96 / np.sqrt(26338)
ax.axhspan(-band, band, color=GRID, alpha=0.7, lw=0, zorder=0)
ax.axhline(0, color=BASE, lw=0.8)
series = [(d["acf_abs"][1:], ORANGE, "Size of daily move"), (d["acf_r"][1:], BLUE, "Direction of daily move"),
          (d["acf_absz"][1:], AQUA, "Size, after volatility model")]
for y, c, lab in series:
    ax.plot(lags, y, color=c, label=lab)
offs = {"Size of daily move": 0, "Direction of daily move": -9, "Size, after volatility model": 9}
for y, c, lab in series:
    label_end(ax, lags[-1], y[-1], lab, c, dy=offs[lab])
ax.set_xscale("log")
ax.set_xlim(1, 250)
ax.set_ylim(-0.07, 0.36)
ax.set_xticks([1, 5, 21, 63, 250], ["1 day", "1 week", "1 month", "1 quarter", "1 year"])
ax.set_ylabel("Autocorrelation")
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
ax.text(120, -band - 0.004, "noise band", color=MUTED, fontsize=8.5, va="top", ha="center")
title(fig, "The market's memory is for size, not direction",
      "US market, daily, 1926–2026. Correlation of today's move with the move N days earlier.")
fig.savefig(FIG / "fig1_memory.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------- fig 2: synchrony event study
ev = pd.read_csv(RES / "test2_event_study.csv")
fig, ax = plt.subplots(figsize=(8.4, 4.8))
fig.subplots_adjust(left=0.08, right=0.8, top=0.8, bottom=0.13)
ax.axvspan(0, 1, color=GRID, alpha=0.8, lw=0, zorder=0)
ax.text(0.5, 0.36, "crash\nmonth", color=INK2, fontsize=9, ha="center", va="bottom")
ax.axhline(0.5, color=BASE, lw=0.8)
ax.plot(ev["k"], ev["vol_pct"], color=ORANGE, label="Volatility (trailing month)")
ax.plot(ev["k"], ev["sync_pct"], color=BLUE, label="Synchrony (industries moving together)")
label_end(ax, 6, ev["vol_pct"].iloc[-1], "Volatility", ORANGE, dy=6)
label_end(ax, 6, ev["sync_pct"].iloc[-1], "Synchrony", BLUE, dy=-6)
ax.set_xlim(-6.3, 6)
ax.set_ylim(0.35, 1.0)
ax.set_xticks(range(-6, 7), [f"{k:+d}" if k else "0" for k in range(-6, 7)])
ax.set_xlabel("Month boundaries relative to the crash month (0 = its first day; 21-trading-day blocks)")
ax.set_ylabel("Percentile vs all months (0.5 = typical)")
ax.legend(loc="upper left", fontsize=9.5)
title(fig, "Synchrony follows crashes; it doesn't warn of them",
      "Average state at the start of each month, around the 25 worst months of 1926–2026.")
fig.savefig(FIG / "fig2_synchrony_event.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------- fig 3: avalanches
d3 = np.load(RES / "test3_dists.npz")


def ccdf(x):
    xs = np.sort(x)
    return xs, 1 - np.arange(len(xs)) / len(xs)


fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.13, wspace=0.25)
sets = [("Real stocks", INK, "real", 2.2), ("Null: real volatility, random shocks", BLUE, "rv", 1.6),
        ("Null: one simulated volatility driver", ORANGE, "gjr", 1.6), ("Null: days shuffled", AQUA, "shuf", 1.6)]
for ax, kind, xl in [(axes[0], "s", "Avalanche size (total stock-firings)"), (axes[1], "d", "Avalanche duration (trading days)")]:
    for lab, c, key, lw in sets:
        x, y = ccdf(d3[f"{kind}_{key}"])
        ax.plot(x, y, color=c, lw=lw, label=lab, zorder=4 if key == "real" else 3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xl)
axes[0].set_ylabel("Share of avalanches at least this big")
axes[1].set_xlim(0.9, 160)
axes[0].legend(loc="lower left", fontsize=9)
title(fig, "Market avalanches look critical, but a shared volatility state reproduces them",
      "237 S&P 500 stocks, 1990–2026. A stock \"fires\" on a move beyond 4 robust SDs; an avalanche is a run of days with firing.\n"
      "The largest real avalanche is the COVID crash (Feb–May 2020), the second is autumn 2008.")
fig.savefig(FIG / "fig3_avalanches.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------- fig 4: feedback balance
roll = pd.read_csv(RES / "test4_rolling.csv", index_col=0, parse_dates=True)["ac1_252"]
_, spx = load_stocks(ROOT / "data")


def ac1(x):
    x = x - x.mean()
    return (x[1:] * x[:-1]).sum() / (x ** 2).sum()


spx_roll = np.expm1(spx).rolling(252).apply(ac1, raw=True)
fig, ax = plt.subplots(figsize=(10, 4.8))
fig.subplots_adjust(left=0.07, right=0.86, top=0.8, bottom=0.1)
ax.axhline(0, color=BASE, lw=1)
r1 = roll.resample("ME").last()
r2 = spx_roll.resample("ME").last()
ax.plot(r1.index, r1.values, color=BLUE, label="Whole US market (CRSP)")
ax.plot(r2.index, r2.values, color=ORANGE, label="S&P 500 (large, liquid stocks only)")
label_end(ax, r1.index[-1], r1.dropna().iloc[-1], "Whole market", BLUE, dy=-9)
label_end(ax, r2.index[-1], r2.dropna().iloc[-1], "S&P 500", ORANGE, dy=9)
ax.text(pd.Timestamp("1931-01-01"), 0.42, "Positive: today tends to continue yesterday (trend)", color=INK2, fontsize=9)
ax.text(pd.Timestamp("1931-01-01"), -0.36, "Negative: today tends to reverse yesterday (mean reversion)", color=INK2, fontsize=9)
ax.set_ylim(-0.42, 0.47)
ax.set_xlim(pd.Timestamp("1927-01-01"), pd.Timestamp("2027-06-01"))
ax.set_ylabel("Lag-1 autocorrelation, trailing year")
ax.legend(loc="upper right", fontsize=9.5, bbox_to_anchor=(1.0, 1.04))
title(fig, "The market's feedback flipped from trend-following to self-correcting",
      "Correlation of each day's return with the previous day's, measured over a rolling one-year window.")
fig.savefig(FIG / "fig4_feedback.png", dpi=200)
plt.close(fig)
print("saved:", sorted(p.name for p in FIG.glob("*.png")))
