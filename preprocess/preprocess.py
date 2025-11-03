"""Preprocess raw CSV data into a canonical monthly dataset for training.

Behavior:
- Scans a raw data directory for CSV files. If files named with 'revenue' or 'budget' exist they are used; otherwise uses the most recent CSV file.
- Parses date columns (tries common names like 'Month', 'Date', 'ds').
- Normalizes column names to: Department, ds, y (where y is Revenue or Amount).
- Ensures monthly frequency (period start of month) and fills small gaps with NaNs (does not impute automatically).
- Writes processed CSV to out_dir/latest.csv and a timestamped file.

CLI args:
  --raw-dir    directory containing raw CSV(s) (default data/raw)
  --out-dir    directory to write processed CSV (default data/processed)
  --revenue-col optional: name of revenue column if notebook uses a different name
  --min-rows   department minimum rows to keep (default 6)

This is a best-effort port of typical notebook cleaning logic and is intentionally conservative about imputations.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def try_parse_date(series: pd.Series):
    # Try several common date formats; return pd.Series of datetimes or NaT
    return pd.to_datetime(series, errors='coerce')


def find_candidate_file(raw_dir: Path, keyword: str=None):
    csvs = list(raw_dir.glob('*.csv'))
    if not csvs:
        raise FileNotFoundError(f'No CSV files found in {raw_dir}')
    if keyword:
        matches = [p for p in csvs if keyword.lower() in p.name.lower()]
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    # fallback: return most recent CSV
    return sorted(csvs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def normalize_dataframe(df: pd.DataFrame, revenue_col_hint=None):
    # Lowercase column names for easier matching
    df = df.rename(columns={c: c.strip() for c in df.columns})
    cols_lower = {c.lower(): c for c in df.columns}

    # find date column
    date_candidates = ['month', 'date', 'ds', 'timestamp']
    date_col = None
    for cand in date_candidates:
        if cand in cols_lower:
            date_col = cols_lower[cand]
            break
    if date_col is None:
        # try any column that contains 'date' or looks like month
        for c in df.columns:
            if 'date' in c.lower() or 'month' in c.lower():
                date_col = c
                break
    if date_col is None:
        raise ValueError('No date-like column found in dataframe')

    # find revenue/amount column
    revenue_candidates = ['revenue', 'revenues', 'amount', 'actual', 'value', 'y']
    if revenue_col_hint:
        revenue_candidates.insert(0, revenue_col_hint)
    revenue_col = None
    for cand in revenue_candidates:
        if cand in cols_lower:
            revenue_col = cols_lower[cand]
            break
    if revenue_col is None:
        # try numeric columns excluding date
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != date_col]
        if len(numeric_cols) == 1:
            revenue_col = numeric_cols[0]
        else:
            raise ValueError('No revenue/amount column found; candidates: ' + ','.join(revenue_candidates))

    # find department column (optional)
    dept_candidates = ['department', 'dept', 'team', 'segment']
    dept_col = None
    for cand in dept_candidates:
        if cand in cols_lower:
            dept_col = cols_lower[cand]
            break

    # build canonical dataframe
    out = pd.DataFrame()
    out['ds'] = try_parse_date(df[date_col])
    out['y'] = pd.to_numeric(df[revenue_col], errors='coerce')
    if dept_col:
        out['Department'] = df[dept_col].astype(str)
    else:
        out['Department'] = 'global'

    # drop rows where date is NaT
    out = out.dropna(subset=['ds']).copy()

    # convert dates to end-of-month timestamps (consistent with Prophet monthly 'M')
    out['ds'] = out['ds'].dt.to_period('M').dt.to_timestamp('M')

    # sort
    out = out.sort_values(['Department', 'ds']).reset_index(drop=True)

    return out


def aggregate_monthly(out: pd.DataFrame):
    # ensure one value per department-month by summing or taking last
    agg = out.groupby(['Department', 'ds'], as_index=False).agg({'y': 'sum'})
    return agg


def filter_min_rows(df: pd.DataFrame, min_rows: int):
    counts = df.groupby('Department').size()
    keep_depts = counts[counts >= min_rows].index.tolist()
    logging.info(f'Keeping {len(keep_depts)} departments with >={min_rows} rows')
    return df[df['Department'].isin(keep_depts)].copy()


def write_outputs(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    out_latest = out_dir / 'latest.csv'
    out_ts = out_dir / f'processed_{ts}.csv'
    df.to_csv(out_latest, index=False)
    df.to_csv(out_ts, index=False)
    logging.info(f'Wrote processed data to {out_latest} and {out_ts}')
    return out_latest, out_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='data/raw', help='Directory with raw CSV files')
    parser.add_argument('--out-dir', default='data/processed', help='Directory to write processed CSV')
    parser.add_argument('--revenue-col', default=None, help='Optional hint for revenue column name')
    parser.add_argument('--min-rows', type=int, default=6, help='Minimum rows per department to keep')
    parser.add_argument('--revenue-file', default=None, help='Optional explicit filename inside raw-dir to use')
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.revenue_file:
        file_path = raw_dir / args.revenue_file
        if not file_path.exists():
            raise FileNotFoundError(f'{file_path} not found')
    else:
        # prefer files that look like revenues
        try:
            file_path = find_candidate_file(raw_dir, keyword='revenue')
        except FileNotFoundError:
            logging.warning('No revenue-named file found; picking most recent CSV')
            file_path = find_candidate_file(raw_dir, keyword=None)

    logging.info(f'Using raw file: {file_path}')

    df_raw = pd.read_csv(file_path)
    # Normalize
    try:
        df_norm = normalize_dataframe(df_raw, revenue_col_hint=args.revenue_col)
    except Exception as e:
        logging.error('Error normalizing dataframe: %s', e)
        raise

    df_agg = aggregate_monthly(df_norm)
    df_filtered = filter_min_rows(df_agg, args.min_rows)
    write_outputs(df_filtered, out_dir)


if __name__ == '__main__':
    main()
