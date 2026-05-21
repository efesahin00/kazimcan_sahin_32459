import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf

TICKER = "GOOG"

df = yf.download(TICKER, period="500d")
df.columns = df.columns.get_level_values(0)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.index.name = "Date"

print(f"=== Data Quality Pipeline: {TICKER} ===")
print(f"Raw data: {len(df)} trading days")

violations = {}
violations['low_gt_open']     = int((df['Low']   > df['Open']).sum())
violations['low_gt_close']    = int((df['Low']   > df['Close']).sum())
violations['open_gt_high']    = int((df['Open']  > df['High']).sum())
violations['close_gt_high']   = int((df['Close'] > df['High']).sum())
violations['negative_volume'] = int((df['Volume'] < 0).sum())
total_v = sum(violations.values())
print(f"\nStructural violations: {total_v}")

full_index = pd.bdate_range(df.index[0], df.index[-1])
df_full = df.reindex(full_index)
n_missing = int(df_full['Close'].isna().sum())
print(f"Missing business days: {n_missing}")

for col in ['Open', 'High', 'Low', 'Close']:
    df_full[col] = df_full[col].ffill()
df_full['Volume'] = df_full['Volume'].fillna(0)

df_full['log_return'] = np.log(df_full['Close'] / df_full['Close'].shift(1))

returns = df_full['log_return'].dropna()
rm   = returns.rolling(63, min_periods=20).median()
dev  = (returns - rm).abs()
rmad = dev.rolling(63, min_periods=20).median()
mz   = (returns - rm) / (1.4826 * rmad)
outliers   = mz.abs() > 3.5
n_outliers = int(outliers.sum())
print(f"Outliers flagged: {n_outliers}")

n  = len(df_full)
nr = len(returns)
validity     = (n - total_v) / n
completeness = (n - n_missing) / n
outlier_sc   = (nr - n_outliers) / nr
quality      = validity * completeness * outlier_sc

grade = 'A' if quality >= 0.99 else 'B' if quality >= 0.95 else 'C' if quality >= 0.90 else 'F'
print(f"\n{'='*40}")
print(f"QUALITY GRADE: {grade} ({quality:.3f})")
print(f"  Validity:     {validity:.4f}")
print(f"  Completeness: {completeness:.4f}")
print(f"  Outlier:      {outlier_sc:.4f}")
print(f"{'='*40}")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(df_full.index, df_full['Close'], color='#26804a', linewidth=1)
if n_outliers > 0:
    outlier_idx = returns[outliers].index
    axes[0].scatter(outlier_idx, df_full.loc[outlier_idx, 'Close'],
                    color='red', s=30, zorder=5, label=f'{n_outliers} outliers')
    axes[0].legend()
axes[0].set_ylabel('Close Price')
axes[0].set_title(f'{TICKER} - Clean Pipeline Output (Grade: {grade})')
axes[0].grid(True, alpha=0.3)

axes[1].bar(returns.index, returns, color='#339b5e', alpha=0.5, width=1)
axes[1].set_ylabel('Log Return')
axes[1].axhline(y=0, color='white', linewidth=0.5)
axes[1].grid(True, alpha=0.3)

vol = returns.rolling(20).std() * np.sqrt(252)
axes[2].plot(vol.index, vol, color='#56b67d', linewidth=1)
axes[2].set_ylabel('20d Ann. Vol')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot1_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
os.system('open plot1_pipeline.png')

print("\n=== EDA: Descriptive Statistics ===")
print(f"Mean:     {returns.mean():.6f}")
print(f"Median:   {returns.median():.6f}")
print(f"Std:      {returns.std():.6f}")
print(f"Skewness: {returns.skew():.4f}  {'(left-skewed)' if returns.skew() < 0 else '(right-skewed)'}")
print(f"Kurtosis: {returns.kurt():.4f}  {'(fat tails)' if returns.kurt() > 0 else '(thin tails)'}")
print(f"Min:      {returns.min():.6f}")
print(f"Max:      {returns.max():.6f}")

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle(f'{TICKER} — EDA: Return Distribution', fontsize=13)

mu, sigma = returns.mean(), returns.std()
axes2[0].hist(returns, bins=50, density=True,
              color='#339b5e', alpha=0.7, edgecolor='white', linewidth=0.3)
x = np.linspace(returns.min(), returns.max(), 300)
axes2[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
axes2[0].axvline(mu,               color='gold', lw=1.5, linestyle='--', label=f'Mean={mu:.4f}')
axes2[0].axvline(returns.median(), color='cyan', lw=1.5, linestyle=':',  label='Median')
axes2[0].set_xlabel('Log Return')
axes2[0].set_ylabel('Density')
axes2[0].set_title('Histogram + Normal Fit')
axes2[0].legend(fontsize=8)

axes2[1].boxplot(returns, vert=True, patch_artist=True,
                 boxprops=dict(facecolor='#339b5e', alpha=0.7),
                 medianprops=dict(color='gold', linewidth=2),
                 flierprops=dict(marker='o', color='red', markersize=3, alpha=0.5))
axes2[1].set_ylabel('Log Return')
axes2[1].set_title('Box-Plot (Outlier Detection)')
axes2[1].set_xticks([])

axes2[2].scatter(returns.values[:-1], returns.values[1:],
                 alpha=0.3, s=8, color='#26804a')
axes2[2].set_xlabel('Return (t)')
axes2[2].set_ylabel('Return (t+1)')
axes2[2].set_title('Return(t) vs Return(t+1)')
axes2[2].axhline(0, color='white', lw=0.5)
axes2[2].axvline(0, color='white', lw=0.5)

plt.tight_layout()
plt.savefig('plot2_eda.png', dpi=150, bbox_inches='tight')
plt.close()
os.system('open plot2_eda.png')

ALPHA = 0.05

print(f"\n{'='*50}")
print("  HYPOTHESIS TESTING")
print(f"{'='*50}")

print("\n── Test 1: One-Sample t-test ──")
print("  H0: Mean log return = 0")
print("  H1: Mean log return != 0")

t1, p1 = stats.ttest_1samp(returns, popmean=0)
print(f"\n  n           = {len(returns)}")
print(f"  Sample mean = {returns.mean():.6f}")
print(f"  t-statistic = {t1:.4f}")
print(f"  p-value     = {p1:.6f}")
if p1 < ALPHA:
    print(f"\n  -> REJECT H0  (p={p1:.4f} < alpha={ALPHA})")
else:
    print(f"\n  -> FAIL TO REJECT H0  (p={p1:.4f} >= alpha={ALPHA})")

print("\n── Test 2: Two-Sample t-test ──")
print("  H0: Mean return (1st half) = Mean return (2nd half)")
print("  H1: They differ")

mid         = len(returns) // 2
first_half  = returns.iloc[:mid]
second_half = returns.iloc[mid:]

lev_stat, lev_p = stats.levene(first_half, second_half)
equal_var = lev_p > 0.05
t2, p2 = stats.ttest_ind(first_half, second_half, equal_var=equal_var)
print(f"\n  Levene p-value = {lev_p:.4f} -> equal_var = {equal_var}")
print(f"  n(1st half) = {len(first_half)},  mean = {first_half.mean():.6f}")
print(f"  n(2nd half) = {len(second_half)}, mean = {second_half.mean():.6f}")
print(f"  t-statistic = {t2:.4f}")
print(f"  p-value     = {p2:.6f}")
if p2 < ALPHA:
    print(f"\n  -> REJECT H0  (p={p2:.4f} < alpha={ALPHA})")
else:
    print(f"\n  -> FAIL TO REJECT H0  (p={p2:.4f} >= alpha={ALPHA})")

print("\n── Test 3: One-Way ANOVA ──")
print("  H0: Mean return is equal across 3 time periods")
print("  H1: At least one period differs")

third   = len(returns) // 3
period1 = returns.iloc[:third]
period2 = returns.iloc[third:2*third]
period3 = returns.iloc[2*third:]

f_stat, p3 = stats.f_oneway(period1, period2, period3)
print(f"\n  Period 1: n={len(period1)}, mean={period1.mean():.6f}")
print(f"  Period 2: n={len(period2)}, mean={period2.mean():.6f}")
print(f"  Period 3: n={len(period3)}, mean={period3.mean():.6f}")
print(f"  F-statistic = {f_stat:.4f}")
print(f"  p-value     = {p3:.6f}")
if p3 < ALPHA:
    print(f"\n  -> REJECT H0  (p={p3:.4f} < alpha={ALPHA})")
else:
    print(f"\n  -> FAIL TO REJECT H0  (p={p3:.4f} >= alpha={ALPHA})")

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle(f'{TICKER} — Hypothesis Tests (alpha={ALPHA})', fontsize=13)

df_t   = len(returns) - 1
x_t    = np.linspace(-5, 5, 500)
y_t    = stats.t.pdf(x_t, df_t)
t_crit = stats.t.ppf(1 - ALPHA / 2, df_t)

axes3[0].fill_between(x_t, y_t, where=(x_t < -t_crit), color='red', alpha=0.4, label='Rejection region')
axes3[0].fill_between(x_t, y_t, where=(x_t >  t_crit), color='red', alpha=0.4)
axes3[0].fill_between(x_t, y_t, where=(x_t >= -t_crit) & (x_t <= t_crit), color='#339b5e', alpha=0.4, label='Fail to reject')
axes3[0].axvline(t1, color='gold', lw=2, label=f't={t1:.2f}, p={p1:.4f}')
axes3[0].set_title(f'Test 1: One-Sample t-test\n{"REJECT H0" if p1 < ALPHA else "FAIL TO REJECT"}')
axes3[0].set_xlabel('t-statistic')
axes3[0].legend(fontsize=8)

axes3[1].hist(first_half,  bins=40, density=True, alpha=0.6, color='#26804a', label=f'1st half (mean={first_half.mean():.4f})')
axes3[1].hist(second_half, bins=40, density=True, alpha=0.6, color='#f44336', label=f'2nd half (mean={second_half.mean():.4f})')
axes3[1].set_title(f'Test 2: Two-Sample t-test\n{"REJECT H0" if p2 < ALPHA else "FAIL TO REJECT"}')
axes3[1].set_xlabel('Log Return')
axes3[1].legend(fontsize=8)

axes3[2].boxplot([period1, period2, period3],
                 tick_labels=['Period 1', 'Period 2', 'Period 3'],
                 patch_artist=True,
                 boxprops=dict(facecolor='#339b5e', alpha=0.6),
                 medianprops=dict(color='gold', linewidth=2),
                 flierprops=dict(marker='o', color='red', markersize=3, alpha=0.5))
axes3[2].set_title(f'Test 3: One-Way ANOVA\n{"REJECT H0" if p3 < ALPHA else "FAIL TO REJECT"}')
axes3[2].set_ylabel('Log Return')
axes3[2].axhline(0, color='white', lw=0.8, linestyle='--')

plt.tight_layout()
plt.savefig('plot3_hypothesis.png', dpi=150, bbox_inches='tight')
plt.close()
os.system('open plot3_hypothesis.png')

print(f"\n{'='*50}")
print("  SUMMARY")
print(f"{'='*50}")
print(f"  Ticker : {TICKER}  |  Quality Grade: {grade}")
print(f"  Test 1 (One-Sample t)  : p={p1:.4f} -> {'REJECT H0' if p1 < ALPHA else 'FAIL TO REJECT'}")
print(f"  Test 2 (Two-Sample t)  : p={p2:.4f} -> {'REJECT H0' if p2 < ALPHA else 'FAIL TO REJECT'}")
print(f"  Test 3 (One-Way ANOVA) : p={p3:.4f} -> {'REJECT H0' if p3 < ALPHA else 'FAIL TO REJECT'}")
print(f"{'='*50}")