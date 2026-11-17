###
## This is a guided solution to the final exercise
## of the chapter on Pandas. It is, in essence, an
## exploration of a temperatures dataset.
###

# Import typical libraries
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

### Part 1: Importing and exploring the dataset

# Dataset URL
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'

# Verify that the dataset URL exists
import requests
resp = requests.get(url)
if resp.status_code != 200:
    print('Error: URL does not exist.')
    exit(1)
else:
    print('Dataset: average daily minimum temperature in Melbourne (1981–1990) exists.')
    print('Loading data ...\n')

# Load the CSV into a DataFrame with parsed dates and Date as index
try:
    df = pd.read_csv(url, header=0, parse_dates=['Date'], index_col='Date')
except Exception as e:
    print(f'Error loading data: {e}')
    exit(1)

# Display structure of the dataset
print('First 5 rows:')
print(df.head(), '\n')
print('Last 5 rows:')
print(df.tail(), '\n')

# Summary statistics
print('Summary statistics:')
print(df.describe(), '\n')

# Plot the full dataset
df.plot()
plt.title('Daily Minimum Temperatures in Melbourne')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()

### Part 2 — Validating and cleaning

# Ensure Date index is well-formed and sorted
# Does not fix it, just checks and reports issues
if not isinstance(df.index, pd.DatetimeIndex):
    raise TypeError('Index must be a DatetimeIndex.')
df = df.sort_index() # Default (rows) sorting is ascending

# (Optional) ensure index is timezone-naive 
# Not needed here as data is already naive. "Naive" means
# it does not contain timezone information. Example of
# non naive: 2020-01-01 00:00:00+00:00. Corresponding
# example of naive: 2020-01-01 00:00:00.
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# 5) Check for duplicated dates and remove them (keep first)
dup_count = df.index.duplicated(keep='first').sum()
if dup_count > 0:
    print(f'Removing {dup_count} duplicated timestamps.')
    df = df[~df.index.duplicated(keep='first')] # ~ means negation
else:
    print('No duplicated timestamps found.')

# 6) Reindex to a daily frequency and report missing dates created
# If there are missing days, these will be filled with NaNs
df = df.asfreq('D')
missing_days = df.isna().sum(numeric_only=True)
total_missing = int(missing_days.sum())
if total_missing > 0:
    # Show just a preview of missing dates
    na_preview = df[df.isna().any(axis=1)].index[:10]
    print(f'Missing daily rows introduced by asfreq: {total_missing}')
    if len(na_preview) > 0:
        print('First few missing dates:', list(na_preview))
else:
    print('No missing daily rows after aligning to a daily grid.')

# 7) Handle missing values via time-based interpolation
# This is cool: you are trying to do something about missing data
# But can be dangerous as in introduces values that were not measured.
# Also, appreciate how easy is to do complicated things with Pandas.
if total_missing > 0:
    print('Interpolating missing values linearly on the time axis...')
    # Interpolate numeric columns based on the DatetimeIndex
    df2 = df.interpolate(method='time', limit_direction='both')
    # Re-check
    remaining_missing = int(df2.isna().sum(numeric_only=True).sum())
    print(f'Missing values after interpolation: {remaining_missing}')
else:
    print('No interpolation needed — dataset is complete on the daily grid.')
# show groups of three rows around the missing dates, before and after the fix
if total_missing > 0:
    for dt in na_preview:
        print('\nBefore interpolation:')
        print(df.loc[dt - pd.Timedelta(days=1): dt + pd.Timedelta(days=1)])
        print('After interpolation:')
        print(df2.loc[dt - pd.Timedelta(days=1): dt + pd.Timedelta(days=1)])

### Part 3 — Feature engineering

# 8) Add calendar columns. These work because index is a DatetimeIndex
df['year']  = df.index.year
df['month'] = df.index.month
df['day']   = df.index.day
df['doy']   = df.index.dayofyear

# Sanity check for expected temperature column
# It's just that anything we do after this uses
# this column, so we want to fail fast if it's not there
if 'Temp' not in df.columns:
    raise KeyError('Expected column "Temp" not found in DataFrame.')

# 9) Climatology: mean temperature for each day-of-year across all years
# Using transform keeps alignment with original rows
df['clim'] = df.groupby('doy')['Temp'].transform('mean')

# 10) Daily anomaly: observed minus climatology
df['anom'] = df['Temp'] - df['clim']

# Preview
print('Climatology/anomaly preview:')
print(df[['Temp', 'clim', 'anom']].head(), '\n')

### Part 4 — Rolling statistics and yearly extremes

# 11) Rolling means (centred; sensible min_periods)
df['roll7']  = df['Temp'].rolling(window=7,  min_periods=1, center=True).mean()
df['roll30'] = df['Temp'].rolling(window=30, min_periods=1, center=True).mean()

# 12) Yearly extremes: min/max/mean + dates of coldest/warmest day
yr_extremes = df.groupby(df.index.year)['Temp'].agg(['min', 'max', 'mean'])
yr_extremes['coldest_day'] = df.groupby(df.index.year)['Temp'].idxmin()
yr_extremes['warmest_day'] = df.groupby(df.index.year)['Temp'].idxmax()

print('Yearly extremes (min/max/mean with dates):')
print(yr_extremes, '\n')

### Part 5 — Detecting special events (cold spells)

# Parameters
threshold = 5.0   # °C
k = 3             # minimum consecutive days

# Boolean flag for cold days
is_cold = df['Temp'] < threshold

# Identify consecutive runs where the condition is constant
segments = pd.DataFrame({'is_cold': is_cold})
segments['grp'] = (segments['is_cold'] != segments['is_cold'].shift()).cumsum()

# Summarise each run
runs = segments.groupby('grp').agg(
    start=('is_cold', lambda s: s.index[0]),
    end=('is_cold',   lambda s: s.index[-1]),
    length=('is_cold','size'),
    is_cold=('is_cold','first')
)

# Keep only cold runs of length >= k
cold_spells = runs[runs['is_cold'] & (runs['length'] >= k)].drop(columns='is_cold').reset_index(drop=True)

print(f'Cold spells (Temp < {threshold}°C for ≥ {k} consecutive days): {len(cold_spells)} found')
print(cold_spells.head(), '\n')

### Part 6 — Aggregations and pivots (with extra explanations)

# --- Monthly summaries --------------------------------------------------------
# We have a daily DatetimeIndex, so .resample('MS') groups by "Month Start".
#   - 'MS' means bins anchored at the first calendar day of each month.
#   - Resampling uses the INDEX, so it is ideal for time-based aggregation.
#   - This differs from groupby('month'): groupby('month') mixes all Januaries
#     across years together; resample keeps each (year, month) separate.
#
# We aggregate only the temperature column to avoid dtype issues.
monthly = df['Temp'].resample('MS').agg(['mean', 'min', 'max', 'std', 'count'])

# Add a convenience "range" column as (max - min)
monthly['range'] = monthly['max'] - monthly['min']

print('Monthly summaries (first 6 rows):')
print(monthly.head(6), '\n')

# --- Month × Year pivot of average daily minimum temperatures -----------------
# Goal: a 12 (months) × N_years (1981..1990) table with *monthly* averages.
# Strategy:
#   - We already created 'year' and 'month' in Part 3 from the DatetimeIndex.
#   - We keep the data at daily granularity and ask pivot_table to compute the
#     **mean temperature** for each (month, year) combination.
#
# Why pivot_table (instead of pivot)?
#   - pivot requires unique (index, column) pairs.
#   - pivot_table allows an aggregation function when there are multiple rows
#     for the same (month, year) (here: all daily rows within that month).
#
# The result has:
#   - index: months 1..12
#   - columns: years 1981..1990
#   - values: monthly average of daily minima (°C)
pivot = df.pivot_table(
    index='month',
    columns='year',
    values='Temp',
    aggfunc='mean'
)

# Ensure months are in natural order (1..12) even if the index got shuffled
pivot = pivot.reindex(index=range(1, 12 + 1))

print('Month × Year pivot (head shown; it has 12 rows):')
print(pivot.head(), '\n')

# --- Visualise the pivot as a heatmap using Matplotlib (no seaborn) ----------
# Notes:
#   - We do not set a specific colour map to keep defaults simple.
#   - imshow expects a 2D numeric array; we pass pivot.values.
#   - origin='lower' so month 1 appears at the bottom.
#   - aspect='auto' lets Matplotlib stretch to fit the figure nicely.
fig = plt.figure(figsize=(8, 5))
plt.imshow(pivot.values, aspect='auto', origin='lower')
cbar = plt.colorbar()
cbar.set_label('Mean daily minimum (°C)')

# Ticks: months on y-axis, years on x-axis
plt.yticks(
    ticks=np.arange(0, 12),
    labels=[str(m) for m in range(1, 13)]
)
plt.xticks(
    ticks=np.arange(pivot.shape[1]),
    labels=list(pivot.columns),
    rotation=45
)

plt.title('Mean Daily Minimum Temperature — Months × Years')
plt.xlabel('Year')
plt.ylabel('Month')
plt.tight_layout()
plt.show()

### Part 7 — Anomalies and distributions

# 18) Plot anomalies with a 30-day rolling mean and a zero reference line
if 'anom' not in df.columns:
    raise KeyError('Column "anom" not found. Make sure Part 3 created anomalies.')

fig = plt.figure(figsize=(9, 4))
plt.plot(df.index, df['anom'], label='Daily anomaly')
plt.plot(df.index, df['anom'].rolling(30, min_periods=1, center=True).mean(),
         label='30-day mean')
plt.axhline(0.0, linestyle='--')
plt.title('Daily Anomalies vs Climatology')
plt.xlabel('Date')
plt.ylabel('Temp anomaly (°C)')
plt.legend()
plt.tight_layout()
plt.show()

# 19) Per-month distributions of daily temperatures (describe)
by_month_desc = df.groupby('month')['Temp'].describe()
print('Per-month distribution of daily temperatures:')
print(by_month_desc, '\n')

# (Optional) also inspect anomalies by month
by_month_anom = df.groupby('month')['anom'].describe()
print('Per-month distribution of anomalies (optional):')
print(by_month_anom, '\n')

### Part 8 — Exporting results

# Daily export must include: Temp, clim, anom, roll7, roll30
required_cols = ['Temp', 'clim', 'anom', 'roll7', 'roll30']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns for DailyTemps.csv: {missing} — run Parts 3–4 first.")

# Write daily data (index is Date)
df[required_cols].to_csv('DailyTemps.csv', index_label='Date')

# Monthly summary (expects 'monthly' from Part 6)
if 'monthly' not in globals():
    raise NameError('monthly not found — run Part 6 first.')
# Ensure range column exists (harmless if already present)
if 'range' not in monthly.columns and {'max', 'min'}.issubset(monthly.columns):
    monthly['range'] = monthly['max'] - monthly['min']
monthly.to_csv('MonthlySummary.csv', index_label='MonthStart')

# Cold spells table (expects 'cold_spells' from Part 5)
if 'cold_spells' not in globals():
    raise NameError('cold_spells not found — run Part 5 first.')
cold_spells.to_csv('ColdSpells.csv', index=False)

print('Wrote files:')
print(' - DailyTemps.csv')
print(' - MonthlySummary.csv')
print(' - ColdSpells.csv')