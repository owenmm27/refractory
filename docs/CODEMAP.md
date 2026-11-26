# Code map

Every script in `src/`, what it reads and writes, what it depends on, and roughly how long it takes. Run any script from the repo root as `python3 -I src/<script>.py . [args]`. `reproduce.sh` runs them in this order.

Runtimes are on the 8 GB laptop this was built on, with single-threaded BLAS. Blank means under about 2 minutes or not recorded.

## Shared code

| File | What it provides |
|---|---|
| `common.py` | Data loaders: `load_market` (CRSP value-weighted market = Mkt-RF + RF from the Fama-French factors file, daily, decimal), `load_industries` (French 49 value-weighted industries), `load_vix`, `load_stocks` (Yahoo adjusted-close panel and the S&P 500 index). Volatility models: GJR-GARCH-t (`fit_gjr`, `simulate_gjr`); the K-unit leaky-integrator model used everywhere from Test 6 on (`multi_components`, `multi_variance`, `_unpack_multi`, `fit_multi`, `multi_forecast`, `simulate_multi`); a power-input variant (`fit_power`). Stats: `acf`, `ljung_box`, `hill`. **No `__main__` guard** (it only defines things, so importing is safe). |
| `figures.py` | Chart style: palette tokens (`SURF INK INK2 MUTED GRID BASE BLUE ORANGE AQUA`), matplotlib rcParams, `title(fig, main, sub)`, `label_end(...)`. Also draws Figs 1–4 **at import time** (no guard), so importing it needs `data/` and `results/test1–4` outputs. Every later `figures_*.py` does `import figures as F`. |

## Data downloaders

| Script | Fetches | Writes | Notes |
|---|---|---|---|
| `download.py` | Ken French 49 industries + Fama-French factors (zip), Cboe VIX history, S&P 500 member list (GitHub `datasets/s-and-p-500-companies`), Yahoo adjusted close + volume for members and `^GSPC` since 1990 | `data/french/`, `data/vix/VIX_History.csv`, `data/meta/constituents.csv`, `data/yahoo/sp500_{adjclose,volume}.csv` | Skips files that already exist. 4 download threads. |
| `download_intl.py` | Yahoo daily closes for 15 indices since 1960 | `data/intl/<code>.csv` | Defines `INDICES` (code → Yahoo symbol), reused by `download_ohlc.py`. Dates use each exchange's local time (`gmtoffset`). |
| `download_ohlc.py` | Yahoo open/high/low/close for the 15 indices, SPY, and the 237 panel stocks | `data/ohlc/<code>.csv`, `data/ohlc/stocks_{open,high,low,close}.csv` | Panel = members with ≥ 98% adjusted-close coverage. Prices unadjusted (only same-day ratios are used). |
| `download_earnings.py` | SEC EDGAR submissions API: 8-K filings with Item 2.02 | `data/earnings/earnings_8k_202.csv` (committed) | Optional 2nd arg: a CSV with a `symbol` column to restrict companies. Throttled ~0.15 s per request. |
| `parse_fomc.py` | Parses saved federalreserve.gov calendar pages in `data/macro/raw/` | `data/macro/fomc_dates.csv` (committed) | Raw pages aren't committed; see DATA.md. |
| (inside `test5_options.py`) | `get_spy()`: Yahoo SPY open/close/adjclose; SqueezeMetrics DIX/GEX | `data/yahoo/SPY_ohlc.csv`, `data/gex/DIX.csv` | Downloaded on first run if missing. `get_spy` is reused by Tests 9 and 9b. |

## Tests

| Script | Question | Args | Writes (in `results/`) | Imports from | Time |
|---|---|---|---|---|---|
| `test1_predictability.py` | Is direction or size predictable? Hidden state beyond GJR-GARCH? (ACF, OOS R², gradient boosting, mutual information, BDS with bootstrap null) | | `test1.json`, `test1.npz` | common | ~5 min |
| `test2_synchrony.py` | Is synchrony (mean correlation, eigenvalue share, absorption ratio) a warning sign? Event study around the 25 worst months | | `test2.json`, `test2_{blocks,daily_sync,double_sort,stock_sync}.csv` | common | |
| `test3_avalanches.py` | Avalanches in the stock cross-section vs nulls (shuffle, iid, GJR, real-vol-path factor model); branching ratio, power-law fits | `K` threshold in robust SDs (default 4; also run 3 and 5) | `test3.json` (K=4), `test3_K3.json`, `test3_K5.json`, `test3_dists*.npz`, `test3_firing_counts.csv` | common | ~1 min |
| `test4_balance.py` | Trend vs reversal: lag-1 autocorrelation, variance ratios, by era and volatility regime | | `test4.json`, `test4_rolling.csv` | common | |
| `test5_options.py` | Did options hedging cause the flip? (expiry cycle, overnight split, GEX) | | `test5.json` | common | |
| `test6a_timescales_market.py` | Two-timescale (K=2) volatility state for the market; K=1/2/3 LR tests; residual BDS | | `test6a.json`, `test6a.npz` | common | ~10 min |
| `test6b_timescales_stocks.py` | Same for 237 stocks; does it explain leftover avalanche persistence? Cross-stock spillover | | `test6b.json` | common, test3 | ~10 min |
| `test6c_closure.py` | Closing Test 6a/6b leftovers (power input; population unit) | | `test6c.json` | common, test3, test6b | |
| `test6d_sector_null.py` | Is leftover persistence carried by sector factors? | | `test6d.json` | common, test3, test6b | |
| `test6e_pca_null.py` | How many statistical factors does it need? | | `test6e.json` | common, test6b, test6d | |
| `test7_contagion.py` | Directed (one-way, lagged) links among stocks vs common factors; synthetic validation | | `test7.json`, `test7_real.npz`, `test7_leadership_magnitude.csv`, **`cache/z_2unit.npz`** | common, test6b | ~10 min |
| `test8_earnings.py` | Is the remaining structure the earnings calendar? | | `test8.json` | common, test7 | |
| `test9_scheduled_news.py` | Do Fed/CPI/jobs days explain the post-2000 overreaction? | | `test9.json` | common, test5 | |
| `test9b_refractory.py` | **Discovery of the refractory term**: refractory vs delayed-absorption variants, CRSP 2000–26, SPY, CRSP 1926–99 placebo | | `test9b.json` | common, test5, test9 | |
| `test9c_refractory_rolling.py` | When did it appear? Rolling 6-year windows 1926–2026 | | `test9c_rolling.csv`, `test9c.log` | common, test9b | |
| `test10_refractory_where.py` | Where does it live? Index, GICS sectors, French 49, 237 stocks (raw and idiosyncratic), own vs market surprise, with a simulated-null bias check | optional `n_stocks` limit | `test10.json`, `test10_raw.json`, `test10_dilution.csv`, **`cache/test10_z.npz`** | common, test8, test9 | ~80 min |
| `test10b_modelfree.py` | Model-light: today's z² by yesterday's surprise bin, by level; cross-level (market → stocks) | | `test10b.json` | common | |
| `test11_gamma.py` | Does dealer gamma (GEX) create the dip? Controls for VIX | | `test11.json` | common, test9 | |
| `test12_brain_model.py` | **The generative "market brain" model v1** (market, 11 sector and 237 stock units), fit, simulate, ~25-pattern scorecard, ablations | `n_paths` (default 20) | `test12.json`, `test12_scorecard.csv`, **`cache/test12_fit.pkl`** | common, test3, test4, test8, test9 | long |
| `test12c_model_v2.py` | v2: stress-keyed reversal + shared volatility gain on stocks | `n_paths` (20) | `test12c.json`, **`cache/test12c_fit.pkl`** | common, test9, test12 | long |
| `test12b_oos.py` | Out of sample: fit 2000–2012, forecast 2013–2026 | | `test12b.json` | common, test9, test12, test12c | |
| `test12d_rolling_oos.py` | Rolling re-estimation forecast check | | `test12d.json` | test9, test12, test12b, test12c | |
| `test13_bursts.py` | Why do collective bursts persist? Repeat vs new firers, aftershocks | `n_paths` (10) | `test13a.json` | test12, test12c | |
| `test13b_model_v3.py` | v3: two-day earnings bump; robust burst-persistence scorecard | `n_paths` (20) | `test13b.json`, **`cache/test13_fit.pkl`** | common, test9, test12, test12c, test13 | long |
| `test14_intl.py` | Market unit in 15 indices by era: two-speed state, refractory term, reversal, stress-keyed reversal | | `test14.json`, `test14_intl.csv` | common, test4, test10, test12c | |
| `test14b_era_vs_stress.py` | Is the era pattern structural or just stress? | | `test14b.json`, `test14b_era_vs_stress.csv` | test11, test12c, test14 | |
| `test15_score_driven.py` | **Hardening 1**: refractory term on EGARCH and Beta-t-EGARCH baselines; simulation null | `n_sim` (40) | `test15.json`, `test15_score_driven.csv` | common, test10, test14 | ~2 min |
| `test15b_range.py` | **Hardening 2**: model-free next-day vs day-after checks with highs/lows; SPY gap/session split; stocks; 15 indices by era | | `test15b.json`, `test15b_range_intl.csv` | test12, test14, test15 | ~10 min |

Logs (`results/testN.log`, `test15_log.txt`, `test15b_log.txt`) were captured from stdout by shell redirection, not written by the scripts; `reproduce.sh` prints to the terminal instead. Some include warnings from optimizer overflows, which are expected (bad parameter values return a large penalty).

### Known gap: three result files have no script

These were written by one-off analysis code on 21 Nov 2026, not by anything in `src/`. They're committed, so the figures still draw, but a from-scratch run won't regenerate them:

| File | Used by | What it holds |
|---|---|---|
| `test2_event_study.csv` | Fig 2 (`figures.py`) | Synchrony and volatility percentiles around the 25 worst months |
| `test7_followup.json` | Fig 7 (`figures_contagion.py`) | Test 7 follow-up: one-way share of lagged links for magnitude vs direction |
| `test8_event_study.json` | Fig 8 (`figures_earnings.py`) | Peer volatility after earnings surprises, by lag and peer group |

Fixing this means folding that logic into `test2_synchrony.py`, `test7_contagion.py` and `test8_earnings.py`.

## Charts

| Script | Draws | From |
|---|---|---|
| `figures.py` | fig1_memory, fig2_synchrony_event, fig3_avalanches, fig4_feedback | test1.npz, test2_event_study.csv, test3_dists.npz, test4_rolling.csv, plus `data/` |
| `figures_timescales.py` | fig5_two_units, fig6_leftover_persistence | test6a.npz, test6b–e.json |
| `figures_contagion.py` | fig7_contagion | test7.json, test7_followup.json |
| `figures_earnings.py` | fig8_earnings | test8.json, test8_event_study.json |
| `figures_refractory.py` | fig9_refractory | test9c_rolling.csv |
| `figures_where.py` | fig10_where_refractory | test10_dilution.csv, test10b.json |
| `figures_gamma.py` | fig11_gamma | test11.json |
| `figures_brain_model.py` | fig12_brain_model | test12.json, test12c.json |
| `figures_bursts.py` | fig13_bursts | test13b.json |
| `figures_intl.py` | fig14_intl | test14_intl.csv |
| `figures_hardening.py` | fig15_hardening | test15_score_driven.csv, test15b_range_intl.csv, test15b.json |
| `figures_hero.py` | hero_refractory (README) | test15b.json only (no `data/` needed, doesn't import `figures`) |

Fig9 and Fig14 use the original squared-return model; their post-2012 values are inflated (see Test 15). Fig14 carries a note saying so in WRITEUP.md.

## Cache files (`results/cache/`, gitignored)

| File | Written by | Read by | Contents |
|---|---|---|---|
| `z_2unit.npz` | test7 | test8; `reproduce.sh` (symbol list for the earnings download) | Two-unit model residuals for the panel stocks, with `symbols` |
| `test10_z.npz` | test10 | test10b, test11 | Base-model residuals for every Test 10 series |
| `test12_fit.pkl` | test12 | test12c, test13, test13b | Fitted v1 model parameters (market, sectors, stocks) |
| `test12c_fit.pkl` | test12c | test13, test13b | Fitted v2 additions (reversal, gain dynamics, refitted stocks) |
| `test13_fit.pkl` | test13b | test13b | v3 stock refits |

Scripts reuse a cache file if it exists, so delete the relevant `.pkl` to force a refit.

## Dependency order (short form)

download → 1, 2, 3, 4, 5 → 6a–6e → 7 (cache z_2unit) → 8 → 9, 9b, 9c → 10 (cache test10_z) → 10b, 11 → 12 (cache) → 12c (cache) → 12b, 12d, 13 → 13b → 14 → 14b → 15 → 15b → figures.
