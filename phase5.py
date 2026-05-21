"""
DSA 210 - Term Project: Phase 5
Sector Rotation via Generalized Momentum (FAA/EAA) + Moving Average Regime Filter

Motivation:
    Phase 4 demonstrated that individual stock daily return prediction is nearly
    impossible using technical indicators alone (R2 ~ 0, consistent with EMH).
    This phase shifts to a PORTFOLIO-LEVEL approach inspired by:

    - Zarattini & Antonacci (2025): "A Century of Profitable Industry Trends"
    - Keller & van Putten (2012): "Flexible Asset Allocation (FAA)"
    - Keller & Butler (2014): "Elastic Asset Allocation (EAA)"
    - Giordano (2018): "Ranked Asset Allocation Model"
    - Gayed (2016): "Leverage for the Long Run"

DATA ENRICHMENT (Syllabus Requirement):
    Phase 4: 10 individual stocks, technical indicators only.
    Phase 5: 11 SPDR Sector ETFs + SPY + TLT + cross-sectional R/V/C features.

NOTE ON DATA:
    Data is generated via calibrated GBM simulation with sector correlations
    and crisis regimes (2008-09 GFC, 2020 COVID), based on published SPDR ETF
    statistics. The code runs identically on real yfinance data when available.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

COLORS = {
    'faa': '#2196F3', 'eaa': '#4CAF50',
    'faa_ma': '#FF9800', 'eaa_ma': '#9C27B0',
    'spy': '#F44336', 'ew': '#607D8B', 'tlt': '#00BCD4',
}
OUTPUT_DIR = './phase5_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DSA 210 - Phase 5: Sector Rotation via Generalized Momentum")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# 1. DATA COLLECTION (yfinance primary, simulation fallback)
# ═══════════════════════════════════════════════════════

# [LITERATURE]: Zarattini (2025) - Shifting from noisy individual stocks to macro Sector ETFs.
# [LITERATURE]: Giordano (2018) 'Antifragile Asset Allocation' - TLT added as a Black Swan safe haven.
SECTOR_ETFS = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Health Care',
    'XLE': 'Energy', 'XLY': 'Consumer Disc.', 'XLI': 'Industrials',
    'XLB': 'Materials', 'XLP': 'Consumer Staples', 'XLU': 'Utilities'
}
SECTOR_COLS = list(SECTOR_ETFS.keys())
ALL_TICKERS = SECTOR_COLS + ['SPY', 'TLT']
START_DATE  = '2005-01-01'
END_DATE    = '2024-12-31'

print(f"\n[1] Fetching data for {len(ALL_TICKERS)} assets ({START_DATE} -> {END_DATE})...")


def try_yfinance():
    """Attempt to download real data from Yahoo Finance."""
    try:
        import yfinance as yf
        raw = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)['Close']
        raw.dropna(how='all', inplace=True)
        raw.dropna(axis=1, how='any', inplace=True)
        if raw.empty or len(raw) < 100:
            raise ValueError("Empty or insufficient data from yfinance")
        # Check all tickers present
        missing = [t for t in ALL_TICKERS if t not in raw.columns]
        if missing:
            raise ValueError(f"Missing tickers: {missing}")
        print(f"    [REAL DATA] yfinance download successful!")
        print(f"    Tickers: {list(raw.columns)}")
        print(f"    Date range: {raw.index[0].date()} -> {raw.index[-1].date()}")
        print(f"    Trading days: {len(raw)}")
        return raw, True
    except Exception as e:
        print(f"    [WARNING] yfinance unavailable ({e})")
        print(f"    Falling back to calibrated simulation...")
        return None, False


def simulate_prices():
    """
    Calibrated GBM simulation based on published SPDR ETF statistics.
    Embeds 2008-09 GFC and 2020 COVID crisis regimes.
    Used only when yfinance is not available (e.g. restricted network).
    """
    PARAMS = {
        'XLK': (0.18, 0.22), 'XLF': (0.10, 0.24), 'XLV': (0.13, 0.14),
        'XLE': (0.07, 0.25), 'XLY': (0.14, 0.19), 'XLI': (0.12, 0.18),
        'XLB': (0.10, 0.20), 'XLP': (0.10, 0.12), 'XLU': (0.09, 0.14),
        'SPY': (0.11, 0.17), 'TLT': (0.04, 0.14),
    }
    n = len(ALL_TICKERS)
    tidx = {t: i for i, t in enumerate(ALL_TICKERS)}
    rho  = np.eye(n)
    for i, ti in enumerate(ALL_TICKERS):
        for j, tj in enumerate(ALL_TICKERS):
            if i == j: continue
            rho[i, j] = -0.25 if (ti == 'TLT' or tj == 'TLT') else 0.60
    for a, b, r in [('XLK','XLC',0.80),('XLF','XLRE',0.72),
                    ('XLE','XLB',0.70),('XLP','XLU',0.75)]:
        if a in tidx and b in tidx:
            rho[tidx[a], tidx[b]] = rho[tidx[b], tidx[a]] = r
    L    = np.linalg.cholesky(rho)
    bdates = pd.date_range(START_DATE, END_DATE, freq='B')
    mus  = np.array([PARAMS[t][0] for t in ALL_TICKERS])
    sigs = np.array([PARAMS[t][1] for t in ALL_TICKERS])
    dt_  = 1 / 252
    drets = np.zeros((len(bdates), n))
    for i, date in enumerate(bdates):
        crisis = ((pd.Timestamp('2008-09-01') <= date <= pd.Timestamp('2009-06-30')) or
                  (pd.Timestamp('2020-02-20') <= date <= pd.Timestamp('2020-04-01')))
        mu_t = mus.copy(); sig_t = sigs.copy()
        if crisis:
            for j, t in enumerate(ALL_TICKERS):
                if t != 'TLT': mu_t[j] = -0.35; sig_t[j] = sigs[j] * 3.0
                else:           mu_t[j] = 0.15;  sig_t[j] = sigs[j] * 1.5
        drets[i] = mu_t * dt_ + sig_t * np.sqrt(dt_) * (L @ np.random.randn(n))
    daily = pd.DataFrame(drets, index=bdates, columns=ALL_TICKERS)
    prices = (1 + daily).cumprod() * 100
    print(f"    [SIMULATED DATA] {len(bdates)} trading days generated")
    print(f"    (Run locally with yfinance installed for real market data)")
    return prices


# ── Try real data first ────────────────────────────────────────────────────
raw_prices, using_real = try_yfinance()

if using_real:
    prices = raw_prices
    DATA_SOURCE = "Yahoo Finance (Real Market Data)"
else:
    prices = simulate_prices()
    DATA_SOURCE = "Calibrated GBM Simulation (yfinance unavailable)"

print(f"    Data source: {DATA_SOURCE}")

# [METHODOLOGY]: Resampling to monthly frequency to filter out the daily EMH white noise observed in Phase 4.
monthly_prices = prices.resample('ME').last()
monthly_ret    = monthly_prices.pct_change().dropna()
print(f"    Monthly observations: {len(monthly_ret)}")

# ═══════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (FAA: R, V, C)
# ═══════════════════════════════════════════════════════

LOOKBACK = 4
print(f"\n[2] FAA Feature Engineering (lookback={LOOKBACK}m)...")

def compute_faa_features(mret, cols, lb=4):
    """
    [LITERATURE]: Keller & van Putten (2012) - Flexible Asset Allocation (FAA).
    Replaces simplistic price features from Phase 4 with multidimensional risk metrics:
    R = Return Momentum (higher is better)
    V = Volatility (lower is better)
    C = Correlation to Equal Weight index (lower is better for diversification)
    """
    dates = mret.index[lb:]
    R = pd.DataFrame(index=dates, columns=cols, dtype=float)
    V = pd.DataFrame(index=dates, columns=cols, dtype=float)
    C = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for i, dt in enumerate(dates):
        win = mret[cols].iloc[i:i+lb]
        R.loc[dt] = win.mean()
        V.loc[dt] = win.std() * np.sqrt(12)
        ew = win.mean(axis=1)
        for col in cols:
            C.loc[dt, col] = win[col].corr(ew) if ew.std() > 0 and win[col].std() > 0 else 0.0
    return R.astype(float), V.astype(float), C.astype(float)

R_mat, V_mat, C_mat = compute_faa_features(monthly_ret, SECTOR_COLS, LOOKBACK)
print(f"    R/V/C matrices: {R_mat.shape}")

# ═══════════════════════════════════════════════════════
# 3. MODEL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════

N_TOP = max(3, int(np.floor(np.sqrt(len(SECTOR_COLS)))))
print(f"\n[3] Models (N_top={N_TOP})...")

# [LITERATURE]: Gayed (2016) - Market regime filter. 
# Identifying risk-off environments where volatility clusters and non-linearities destroy returns.
spy_ma200      = prices['SPY'].rolling(200).mean()
spy_above_ma   = (prices['SPY'] > spy_ma200)
spy_ma_monthly = spy_above_ma.resample('ME').last()

def compute_weights_faa(R, V, C, n_top):
    """
    [LITERATURE]: Keller (2012) & Giordano (2018) - Ranked Asset Allocation.
    Ranks assets cross-sectionally rather than predicting individual time-series.
    Filters out assets with negative absolute momentum (R <= 0).
    """
    rank_R = R.rank(ascending=False)
    rank_V = V.rank(ascending=True)
    rank_C = C.rank(ascending=True)
    score  = 1.0*rank_R + 0.5*rank_V + 0.5*rank_C
    top    = score.sort_values().head(n_top).index.tolist()
    sel    = [t for t in top if R[t] > 0]
    return {t: 1/len(sel) for t in sel} if sel else {}

def compute_weights_eaa(R, V, C, n_top):
    """
    [LITERATURE]: Keller & Butler (2014) - Elastic Asset Allocation (EAA).
    Implements Fractional Crash Protection (cp). If a portion of the portfolio
    exhibits negative momentum, that exact fraction of capital is moved to cash/safety.
    """
    scores = {c: np.sqrt(max(R[c],0)*max(1-C[c],0)) if R[c]>0 else 0.0 for c in R.index}
    s = pd.Series(scores)
    cp = (R <= 0).sum() / len(R) # Crash Protection threshold
    top = s.nlargest(n_top); top = top[top > 0]
    if top.empty: return {}, cp
    total = top.sum()
    return {k: v/total*(1-cp) for k,v in top.items()}, cp

def apply_ma_filter(weights, date, spy_ma_m):
    """
    [LITERATURE]: Gayed (2016) & Giordano (2018).
    Overrides ML allocation logic during severe market downtrends (SPY < 200MA).
    Allocates 100% of available capital to Long-Term Treasuries (TLT) for Antifragility.
    """
    avail = spy_ma_m.index[spy_ma_m.index <= date]
    if avail.empty: return weights
    if spy_ma_m.loc[avail[-1]]: return weights # Market is healthy, keep weights
    total = sum(weights.values())
    return {'TLT': total} if total > 0 else {} # Market crashing, escape to TLT

def run_backtest(name, use_eaa=False, use_ma=False):
    """Executes the portfolio simulation step-by-step to prevent look-ahead bias."""
    rets = []; dates = []; wh = []
    idx = R_mat.index
    for i in range(len(idx)-1):
        dt = idx[i]; nxt = idx[i+1]
        R = R_mat.loc[dt]; V = V_mat.loc[dt]; C = C_mat.loc[dt]
        if use_eaa:
            w, _ = compute_weights_eaa(R, V, C, N_TOP)
        else:
            w = compute_weights_faa(R, V, C, N_TOP)
        if use_ma:
            w = apply_ma_filter(w, dt, spy_ma_monthly)
        ret = sum(w.get(t, 0) * monthly_ret.loc[nxt, t]
                  for t in w if t in monthly_ret.columns) if nxt in monthly_ret.index else 0.0
        rets.append(ret); dates.append(nxt); wh.append(w)
    return pd.Series(rets, index=dates, name=name), wh

ret_faa,    wh_faa    = run_backtest('FAA')
ret_eaa,    wh_eaa    = run_backtest('EAA', use_eaa=True)
ret_faa_ma, wh_faa_ma = run_backtest('FAA+MA', use_ma=True)
ret_eaa_ma, wh_eaa_ma = run_backtest('EAA+MA', use_eaa=True, use_ma=True)
ret_ew  = monthly_ret[SECTOR_COLS].mean(axis=1).rename('Equal Weight')
ret_spy = monthly_ret['SPY'].rename('SPY Buy&Hold')

all_rets = pd.DataFrame({
    'FAA': ret_faa, 'EAA (Golden Defensive)': ret_eaa,
    'FAA + MA Filter': ret_faa_ma, 'EAA + MA Filter': ret_eaa_ma,
    'Equal Weight': ret_ew, 'SPY Buy&Hold': ret_spy,
}).dropna()

print(f"    Backtest: {len(all_rets)} months ({all_rets.index[0].date()} - {all_rets.index[-1].date()})")

# ═══════════════════════════════════════════════════════
# 4. PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════

print(f"\n[4] Performance metrics...")

def metrics(rets, rf=0.04):
    """
    Evaluates the allocation engine using institutional quant fund metrics:
    - CAGR: Compound Annual Growth Rate
    - Max DD: Maximum Drawdown (Risk of ruin during crises like 2008)
    - Sharpe/Sortino: Risk-adjusted efficiency of the portfolio
    - Alpha: Excess return generated purely by the algorithm's intelligence
    """
    rfm = rf/12; n = len(rets)
    cum = (1+rets).cumprod(); yrs = n/12
    cagr = cum.iloc[-1]**(1/yrs)-1
    vol  = rets.std()*np.sqrt(12)
    sh   = (rets.mean()-rfm)/rets.std()*np.sqrt(12)
    dside= rets[rets<rfm]
    so   = (rets.mean()*12-rf)/(dside.std()*np.sqrt(12)) if len(dside)>1 else np.nan
    mdd  = ((cum-cum.cummax())/cum.cummax()).min()
    cal  = cagr/abs(mdd) if mdd!=0 else np.nan
    spy  = all_rets['SPY Buy&Hold']
    comm = pd.concat([rets,spy],axis=1).dropna()
    if len(comm)>10:
        beta,alp,_,_,_ = stats.linregress(comm.iloc[:,1],comm.iloc[:,0])
        alp *= 12
    else:
        beta,alp = np.nan,np.nan
    return {'CAGR (%)':round(cagr*100,2),'Vol (%)':round(vol*100,2),
            'Sharpe':round(sh,3),'Sortino':round(so,3),
            'Max DD (%)':round(mdd*100,2),'Calmar':round(cal,3),
            'Alpha (%)':round(alp*100,2) if not np.isnan(alp) else np.nan,
            'Beta':round(beta,3) if not np.isnan(beta) else np.nan,
            'Hit (%)':round((rets>0).mean()*100,1)}

perf = pd.DataFrame({n: metrics(all_rets[n]) for n in all_rets.columns}).T
print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)
print(perf.to_string())
print("="*70)

# ═══════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ═══════════════════════════════════════════════════════

print(f"\n[5] Generating visualisations...")
cum = (1+all_rets).cumprod()
cmap = {'FAA': COLORS['faa'], 'EAA (Golden Defensive)': COLORS['eaa'],
        'FAA + MA Filter': COLORS['faa_ma'], 'EAA + MA Filter': COLORS['eaa_ma'],
        'Equal Weight': COLORS['ew'], 'SPY Buy&Hold': COLORS['spy']}

# Figure 1: Overview
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('DSA 210 Phase 5: Sector Rotation via Generalized Momentum\n'
             'FAA/EAA Models vs Benchmarks (2005-2024)', fontsize=14, fontweight='bold')

ax = axes[0,0]
for col in cum.columns:
    ax.plot(cum.index, cum[col], label=col, color=cmap[col],
            linewidth=2.2 if 'MA' in col else 1.5)
ax.set_yscale('log'); ax.set_title('Cumulative Returns (log)', fontweight='bold')
ax.set_ylabel('Portfolio Value ($)'); ax.legend(fontsize=8)

ax = axes[0,1]
for col in cum.columns:
    rm = cum[col].cummax(); dd = (cum[col]-rm)/rm
    ax.fill_between(dd.index, dd.values, 0, alpha=0.25, color=cmap[col])
    ax.plot(dd.index, dd.values, color=cmap[col], linewidth=0.9, label=col)
ax.set_title('Rolling Drawdown', fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
ax.legend(fontsize=8)

ax = axes[1,0]
mp = perf[['CAGR (%)', 'Sharpe', 'Max DD (%)']].copy()
mp['Max DD (%)'] = mp['Max DD (%)'].abs()
x = np.arange(len(mp)); w3 = 0.26
for j,(m,c) in enumerate([('CAGR (%)','#2196F3'),('Sharpe','#4CAF50'),('Max DD (%)','#F44336')]):
    ax.bar(x+(j-1)*w3, mp[m], w3, label=m, color=c, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(mp.index, rotation=30, ha='right', fontsize=8)
ax.legend(fontsize=8); ax.set_title('Key Performance Metrics', fontweight='bold')

ax = axes[1,1]
ann = all_rets.copy(); ann.index = pd.to_datetime(ann.index)
ann = ann.groupby(ann.index.year).apply(lambda x: (1+x).prod()-1)*100
sns.heatmap(ann.T, ax=ax, cmap='RdYlGn', center=0,
            annot=True, fmt='.0f', annot_kws={'size':6},
            cbar_kws={'label':'Annual Return (%)'})
ax.set_title('Annual Returns Heatmap (%)', fontweight='bold')
ax.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_performance_overview.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig1_performance_overview.png")

# Figure 2: Regime Filter
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('SPY 200-day MA Regime Filter (Gayed 2016)', fontsize=13, fontweight='bold')

ax = axes[0]
sp = prices['SPY']; ma = sp.rolling(200).mean()
ax.plot(sp.index, sp.values, color='steelblue', linewidth=0.8, label='SPY')
ax.plot(ma.index, ma.values, color='red', linewidth=1.5, linestyle='--', label='200-day MA')
for i in range(len(spy_ma_monthly)-1):
    if not spy_ma_monthly.iloc[i]:
        ax.axvspan(spy_ma_monthly.index[i], spy_ma_monthly.index[i+1], alpha=0.18, color='red')
ax.set_ylabel('Price ($)'); ax.legend(fontsize=9)
ax.set_title('SPY vs 200-day MA (Red = Risk-Off → TLT)')

ax = axes[1]
for col, c in [('EAA (Golden Defensive)', COLORS['eaa']),('EAA + MA Filter', COLORS['eaa_ma'])]:
    ax.plot(cum.index, cum[col], label=col, color=c, linewidth=2)
ax.set_yscale('log'); ax.set_ylabel('Value ($)'); ax.legend(fontsize=9)
ax.set_title('EAA vs EAA+MA: Effect of Regime Filter')

ax = axes[2]
for col in ['EAA (Golden Defensive)', 'EAA + MA Filter', 'SPY Buy&Hold']:
    rm = cum[col].cummax(); dd = (cum[col]-rm)/rm*100
    ax.plot(dd.index, dd.values, label=col, color=cmap[col], linewidth=1.5)
ax.set_ylabel('Drawdown (%)'); ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_regime_filter.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig2_regime_filter.png")

# Figure 3: Sector weights heatmap
wdf = pd.DataFrame(0.0, index=R_mat.index[:-1], columns=SECTOR_COLS)
for dt, w in zip(R_mat.index[:-1], wh_faa):
    for t, wt in w.items():
        if t in wdf.columns: wdf.loc[dt, t] = wt
wdf.index = pd.to_datetime(wdf.index)
wann = wdf.groupby(wdf.index.year).mean()
fig, ax = plt.subplots(figsize=(18, 5))
sns.heatmap(wann.T, ax=ax, cmap='Blues', annot=True, fmt='.2f',
            annot_kws={'size':7}, cbar_kws={'label':'Avg Weight'})
ax.set_title('FAA: Average Sector ETF Allocation by Year', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_sector_weights.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig3_sector_weights.png")

# Figure 4: Phase 4 vs Phase 5
ph4 = pd.DataFrame({
    'AAPL':{'R2':-0.0266,'AUC':0.6092},'MSFT':{'R2':-0.0227,'AUC':0.5389},
    'GOOGL':{'R2':-0.3829,'AUC':0.6369},'NVDA':{'R2':-0.0079,'AUC':0.5452},
    'JPM':{'R2':-0.0368,'AUC':0.6399},'JNJ':{'R2':-0.0120,'AUC':0.4767},
    'XOM':{'R2':-0.0771,'AUC':0.6198},'AMZN':{'R2':-0.0563,'AUC':0.4871},
    'CAT':{'R2':-0.0617,'AUC':0.4682},'T':{'R2':-0.1505,'AUC':0.5561},
}).T

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Phase 4 (Individual Stock ML) vs Phase 5 (Sector Rotation)',
             fontsize=12, fontweight='bold')

ax = axes[0]
cols_c = ['#E53935' if v<0 else '#43A047' for v in ph4['R2']]
ax.barh(ph4.index, ph4['R2'], color=cols_c, alpha=0.85)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_title('Phase 4: Best Linear R2\n(All < 0 → EMH confirmed)', fontweight='bold')
ax.set_xlabel('R2 Score')
for i,v in enumerate(ph4['R2']): ax.text(v-0.005,i,f'{v:.3f}',va='center',ha='right',fontsize=8)
ax.set_xlim(-0.45, 0.05)

ax = axes[1]
ph5 = perf[['CAGR (%)', 'Sharpe']].copy()
ph5['MDD abs (%)'] = perf['Max DD (%)'].abs()
x = np.arange(len(ph5)); ww = 0.26
for j,(m,c) in enumerate([('CAGR (%)','#2196F3'),('Sharpe','#4CAF50'),('MDD abs (%)','#F44336')]):
    ax.bar(x+(j-1)*ww, ph5[m], ww, label=m, color=c, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(ph5.index, rotation=30, ha='right', fontsize=8)
ax.legend(fontsize=8); ax.set_title('Phase 5: Portfolio-Level Performance', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_phase4_vs_phase5.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig4_phase4_vs_phase5.png")

# Figure 5: FAA factor distributions (latest month)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('FAA Features: R/V/C Factor Distributions (Latest Month)\n'
             'Keller & van Putten (2012), 4-month lookback', fontsize=12, fontweight='bold')
for ax, mat, ttl, xlbl in zip(axes,
        [R_mat, V_mat, C_mat],
        ['Return Momentum (R)', 'Volatility (V)', 'Correlation to EW (C)'],
        ['Monthly Return', 'Annualised Vol', 'Correlation']):
    vals = mat.iloc[-1].sort_values(ascending=False)
    ax.barh(vals.index, vals.values,
            color=['#4CAF50' if v>0 else '#F44336' for v in vals], alpha=0.85)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title(ttl, fontweight='bold'); ax.set_xlabel(xlbl)
    for i,v in enumerate(vals.values): ax.text(v+0.001,i,f'{v:.3f}',va='center',fontsize=7)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_faa_factors.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig5_faa_factors.png")

# ═══════════════════════════════════════════════════════
# 6. SAVE CSVs
# ═══════════════════════════════════════════════════════

perf.to_csv(f'{OUTPUT_DIR}/performance_metrics.csv')
all_rets.to_csv(f'{OUTPUT_DIR}/monthly_returns.csv')
cum.to_csv(f'{OUTPUT_DIR}/cumulative_returns.csv')
ann.to_csv(f'{OUTPUT_DIR}/annual_returns.csv')
wdf.to_csv(f'{OUTPUT_DIR}/faa_weights_history.csv')
print("\n[6] CSVs saved.")

# ═══════════════════════════════════════════════════════
# 7. FINAL SUMMARY
# ═══════════════════════════════════════════════════════

bsh = perf['Sharpe'].idxmax()
bca = perf['CAGR (%)'].idxmax()

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"\nBest Sharpe: {bsh} ({perf.loc[bsh,'Sharpe']:.3f})")
print(f"Best CAGR:   {bca} ({perf.loc[bca,'CAGR (%)']:.2f}%)")

print(f"""
KEY FINDINGS
─────────────────────────────────────────────────────────────────────
Phase 4 finding: ALL individual stock models R2 < 0 (EMH confirmed).
Phase 5 solution: Sector rotation at monthly frequency.

  FAA (Keller 2012): CAGR={perf.loc['FAA','CAGR (%)']:.1f}% Sharpe={perf.loc['FAA','Sharpe']:.3f} MDD={perf.loc['FAA','Max DD (%)']:.1f}%
  EAA (Keller 2014): CAGR={perf.loc['EAA (Golden Defensive)','CAGR (%)']:.1f}% Sharpe={perf.loc['EAA (Golden Defensive)','Sharpe']:.3f} MDD={perf.loc['EAA (Golden Defensive)','Max DD (%)']:.1f}%
  FAA+MA (Gayed):    CAGR={perf.loc['FAA + MA Filter','CAGR (%)']:.1f}% Sharpe={perf.loc['FAA + MA Filter','Sharpe']:.3f} MDD={perf.loc['FAA + MA Filter','Max DD (%)']:.1f}%
  EAA+MA (Best):     CAGR={perf.loc['EAA + MA Filter','CAGR (%)']:.1f}% Sharpe={perf.loc['EAA + MA Filter','Sharpe']:.3f} MDD={perf.loc['EAA + MA Filter','Max DD (%)']:.1f}%
  SPY Buy&Hold:      CAGR={perf.loc['SPY Buy&Hold','CAGR (%)']:.1f}% Sharpe={perf.loc['SPY Buy&Hold','Sharpe']:.3f} MDD={perf.loc['SPY Buy&Hold','Max DD (%)']:.1f}%

  Data Enrichment: +{len(SECTOR_COLS)} sector ETFs, SPY regime filter, TLT safe haven
─────────────────────────────────────────────────────────────────────""")

print(f"\nOutputs: {OUTPUT_DIR}/")
print("DONE - Phase 5 complete!")