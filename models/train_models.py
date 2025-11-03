"""Train Prophet models per-department and persist them.

CLI flags:
  --data-path: path to processed CSV (expects columns: Department, ds or Month, y or Revenue)
  --horizon: integer months to forecast
  --out-model-dir: model registry root directory
  --out-forecast-dir: directory to write CSV forecasts
  --min-rows: minimum rows per department to train (default 6)

This script was adapted from your notebook training cells.
"""
import argparse
import pandas as pd
import os
from pathlib import Path
from prophet import Prophet
from models.persist import save_model
from datetime import datetime


def try_parse_date(s):
    return pd.to_datetime(s, errors='coerce')


def prepare_df(df, date_col_candidates=['Month', 'ds', 'Date'], value_col_candidates=['Revenue', 'Revenues', 'y']):
    # find date column
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError('No date column found in data; expected one of: ' + ','.join(date_col_candidates))
    df[date_col] = try_parse_date(df[date_col])
    # find value column
    val_col = None
    for c in value_col_candidates:
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        raise ValueError('No revenue/value column found in data; expected one of: ' + ','.join(value_col_candidates))

    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: 'ds', val_col: 'y'})
    # If Department missing, add 'global'
    if 'Department' not in df.columns:
        df['Department'] = 'global'
    # ensure monthly period start
    df['ds'] = pd.to_datetime(df['ds']).dt.to_period('M').dt.to_timestamp('M')
    return df


def train_and_save(df_dept: pd.DataFrame, department: str, out_model_dir: str, out_forecast_dir: str, horizon: int, min_rows=6):
    if df_dept.shape[0] < min_rows:
        print(f"Skipping {department}: only {df_dept.shape[0]} rows (<{min_rows})")
        return None

    # Prophet expects columns ds and y
    df_train = df_dept[['ds', 'y']].sort_values('ds')

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    # add monthly seasonality if desired
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    m.fit(df_train)

    # create future dataframe
    future = m.make_future_dataframe(periods=horizon, freq='M')
    forecast = m.predict(future)

    # save model
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    model_path = os.path.join(out_model_dir, department, f'model_v{ts}.joblib')
    metadata = {
        'department': department,
        'trained_at': ts,
        'n_train_rows': int(df_train.shape[0]),
        'horizon_months': int(horizon)
    }
    save_model(m, model_path, metadata)
    print(f"Saved model for {department} -> {model_path}")

    # save forecast CSV (include only forecast horizon rows)
    fc_dir = Path(out_forecast_dir)
    fc_dir.mkdir(parents=True, exist_ok=True)
    forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    out_csv = fc_dir / f'revenue_forecast_{department}_{ts}.csv'
    forecast_out.to_csv(out_csv, index=False)
    print(f"Saved forecast for {department} -> {out_csv}")

    return {'department': department, 'model_path': model_path, 'forecast_path': str(out_csv)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Path to processed CSV')
    parser.add_argument('--horizon', type=int, default=3, help='Forecast horizon in months')
    parser.add_argument('--out-model-dir', default='model_registry', help='Directory to save models')
    parser.add_argument('--out-forecast-dir', default='artifacts/forecasts', help='Directory to save forecasts')
    parser.add_argument('--min-rows', type=int, default=6, help='Minimum rows per department to train')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df = prepare_df(df)

    departments = df['Department'].unique()
    manifest = []
    for dept in departments:
        df_dept = df[df['Department'] == dept].copy()
        res = train_and_save(df_dept, str(dept), args.out_model_dir, args.out_forecast_dir, args.horizon, min_rows=args.min_rows)
        if res:
            manifest.append(res)

    # write manifest
    Path('artifacts').mkdir(parents=True, exist_ok=True)
    manifest_path = Path('artifacts') / f'manifest_{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.json'
    import json
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print('Wrote manifest to', manifest_path)


if __name__ == '__main__':
    main()
