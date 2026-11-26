# Methods reference

The models, statistics and conventions the tests share, with pointers to the code. Numbers quoted here are illustrative; the authoritative values are in FINDINGS.md and `results/`.

## Brain terms and what they mean here

| Brain term | Market measurement | Where |
|---|---|---|
| State / memory | The volatility state: a fast and a slow leaky integrator of past squared shocks | Tests 1, 6 |
| Arousal | The overall level of that state | Tests 6, 12 |
| Firing | A stock's move beyond K robust SDs (K = 4 by default) | Tests 3, 6, 13 |
| Avalanche | A run of consecutive days with at least one stock firing; size = total firings, duration = days | Test 3 |
| Branching ratio | How many firings one day's firings "cause" the next day (multistep-regression estimator) | Test 3 |
| Synchrony | Mean pairwise correlation, top-eigenvalue share, absorption ratio (rolling) | Test 2 |
| Excitation / inhibition | Trend (positive autocorrelation, variance ratio > 1) vs reversal (< 1) | Tests 4, 12, 14 |
| Refractory period | The day after a big surprise is quieter than the volatility state implies, then the state resumes | Tests 9–15 |
| Synaptic contagion | Directed (one-way, lagged) links between stocks | Tests 7, 8 |
| Neuromodulation | A shared gain on all stocks' own volatility | Tests 12c, 13 |

## Data conventions

- Returns are daily, in percent, demeaned within the fitting sample. The US market is the CRSP value-weighted return (Fama-French Mkt-RF + RF, via `common.load_market`).
- **Stock panel:** the 237 current S&P 500 members with ≥ 98% price coverage since 1990 (survivorship bias acknowledged). `test12_brain_model.load_panel()` returns the panel, market, GICS sectors and earnings-day matrix.
- **Idiosyncratic return:** residual on the S&P 500 (Test 10), or on market plus sector factor (Tests 12–13).
- **Earnings reaction day:** the filing date if the 8-K Item 2.02 was accepted before 4 pm New York time on a trading day, otherwise the next trading day (`test8_earnings.reaction_days`).
- **Eras.** Most tests: before 2000, 2000–2012, 2013–2026 (`test14_intl.ERAS`, `test15b_range.ERAS_INTL`). Test 10 uses 1990–1999 / 2000–2012 / 2013–2026, since the stock data start in 1990. Test 15b's US index split is 1962–1989 / 1990–1999 / 2000–2012 / 2013–2026 (`ERAS_US`).
- International indices: Yahoo price indices, exact repeat closes and |r| > 40% dropped (`test14_intl.load`). Highs and lows only from reliable start dates (`test15b_range.H_L_START`).

## The volatility state (the "two-unit" model)

Defined in `common.multi_components` and fitted by `common.fit_multi` (Student-t likelihood):

```
u_t   = r_{t-1}^2 * (1 + δ·1[r_{t-1} < 0]) / (1 + δ/2)      asymmetric input: down moves count more
c_k,t = a_k · u_t + b_k · c_k,t-1                           unit k leaks at rate b_k (k = 1 fast, 2 slow)
h_t   = ω + Σ_k c_k,t                                       conditional variance
```

- K = 1 is GJR-GARCH(1,1); K = 2 is the fast + slow model used from Test 6 on (half-lives about 2 weeks and about a year).
- Loop gain W = Σ a_k / (1 − b_k) < 1 for stationarity. Parameters are unconstrained in `_unpack_multi` (ordered b's, logistic W and shares, ν = 2.05 + exp(·) capped near 1000).
- Standardized residual z_t = r_t / √h_t. If the model is right, z² averages 1 whatever happened yesterday.

## The refractory term

Introduced in `test9b_refractory.py`, generalized in `test10_refractory_where.gpath`:

```
s_t   = |r_t| / √level_t                          yesterday's surprise relative to the state
h_t   = level_t · exp(−ρ · (s_{t-1} − s̄))         a one-day transient; it never feeds the state
```

- **Headline metric:** next-day variance after a 3-SD surprise, as % of the state's forecast = exp(−ρ·(3 − s̄)). 100% means no refractory effect; ~55% (squared-return model) or ~61% (Beta-t) in 2000–2012.
- Test 10 adds an own-surprise and a market-surprise term (ρ_own, ρ_mkt) and an earnings-day bump κ kept out of the persistent state.
- Tests 11 and 14b let ρ depend on covariates (gamma percentile, log VIX, down-day dummy, a stress gauge, era dummies), with matching level shifts so ρ measures only the day-after change.
- Significance: likelihood ratio vs the nested base fit. χ²(1) cut-offs 3.84 (5%) and 10.83 (0.1%).
- **Known bias:** a squared-return state overreacts to big days, which by itself makes the next day look quiet. Test 15 measured this: the squared-return model reports a spurious ~83% on no-effect simulations. Estimates from Tests 9–14 are upper bounds on the effect's size.

## Robust baselines (Test 15)

`test15_score_driven.py`, numba kernels `betat_state` and `egarch_state`, 7 parameters each like the squared-return model:

- **EGARCH:** log-variance state fed by |z| − E|z| plus a sign term θz, in two components.
- **Beta-t-EGARCH** (Harvey & Chakravarty 2008): state fed by the bounded Student-t score u = (ν+1)·r² / ((ν−2)·e^λ + r²) − 1, with leverage k*·sgn(−r)·(u+1) in the fast unit. An extreme day moves the state only a limited amount.
- Both: λ_t = ω + c1_t + c2_t; log h_t = λ_t − ρ·(s_{t−1} − s̄), s = |r| / e^{λ/2}.
- **Simulation null:** simulate the fitted Beta-t base (no refractory term, `betat_sim`), refit every model + refractory term, and see what each reports.

## Model-free check (Test 15b)

`test15b_range.py`:

- **Shock day t:** |r_t| ≥ 2 (or 3) × trailing 21-day RMS move (`shock_size`).
- **Compare day t+1 with day t+2.** In any model where a shock raises volatility and the effect then decays, t+1 is at least as volatile as t+2. So ratios t+1 / t+2 ≥ 1 and the "calmer share" ≤ ~50% without a refractory effect.
- **Measures** (ratio of sums over events): net move = r²; activity = Parkinson range variance (ln H/L)² / (4 ln 2); efficiency = net move / activity.
- **Calmer share:** fraction of events with |r_{t+1}| < |r_{t+2}| (robust to crisis days). Null: 200 simulations of each market's own Beta-t fit (`calmer_null`).
- **SPY split:** overnight gap (close→open) and session (open→close), plus their covariance and range.
- **95% intervals:** bootstrap over calendar months (`boot`, multinomial weights on month sums).

## The market-brain model (Tests 12–13)

| Version | Script | Adds |
|---|---|---|
| v1 | `test12_brain_model.py` | Market unit (two-unit state, refractory term, short and medium reversal); 11 sector units (own state + persistent market spillover + one-day response to yesterday's market surprise); 237 stock units (same, plus earnings-day bump) |
| v2 | `test12c_model_v2.py` | Reversal that strengthens with the volatility level; a shared multiplicative gain G_t on every stock's own variance (log G = β·log market level + fast and slow AR(1) parts + noise) |
| v3 | `test13b_model_v3.py` | A second-day earnings bump |

Each layer is fitted by Student-t maximum likelihood on 2000–2026, then simulated (20 histories per variant). A scorecard of ~25–30 patterns from earlier tests is computed identically on real and simulated data. Ablations refit without one piece at a time.

## Other statistics

- **Variance ratio** VR(q) = Var(q-day sums) / (q · Var(daily)) (`test4_balance.variance_ratio`). Above 1 = trend, below 1 = reversal.
- **Lag-1 autocorrelation** `test4_balance.ac1`; by regime with `pair_ac1` (regime known at t−1).
- **Firing:** |r| > K × robust scale, scale = 1.4826 × trailing 252-day median |r| (`test3_avalanches.firing_counts`).
- **Branching ratio:** multistep regression (Wilting & Priesemann 2018), `test3_avalanches.mr_branching`.
- **Overreaction signature:** correlation of |z| on consecutive days after a plain two-unit fit. Negative means the state overshoots.
- **BDS:** test of i.i.d. residuals, compared with a simulate-and-refit bootstrap null (Tests 1, 6a).
- **Directed structure:** split the matrix of lag-1 cross-correlations into symmetric and antisymmetric (one-way) parts, and keep the structure that repeats across the two halves of the sample (`test7_contagion`).
- **GEX / DIX:** SqueezeMetrics' dealer-gamma and dark-pool indices, used as a rolling one-year percentile (Test 11).

## Inference conventions

- Nested maximum-likelihood fits (one start exactly at the base solution), so likelihood ratios are ≥ 0.
- Nulls are built by **simulate-and-refit** wherever an estimator might be biased (Tests 1, 10, 15, 15b). Test 3's nulls simulate independent stocks under several volatility models.
- Bootstraps respect dependence: 21-day blocks (Tests 4, 10b) or calendar months (Test 15b).
- Simulations use fixed seeds. Test 15 was rerun in Nov 2026 and reproduced exactly; other tests haven't been rechecked.
- Many tests were run. Treat single p-values around 0.02 as leads, not results.
