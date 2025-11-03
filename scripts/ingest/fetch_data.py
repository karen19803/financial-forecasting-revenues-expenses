"""Fetch data from source. Supports local path or s3:// bucket URI.

Usage:
  python fetch_data.py --source s3://bucket/path/konecta_revenues.csv --out-dir data/raw

If source is a directory, will copy all CSV files found.
"""
import argparse
from pathlib import Path
import shutil
import os
from urllib.parse import urlparse
from datetime import datetime


def fetch_local(path, out_dir: Path):
    p = Path(path)
    if p.is_file():
        out = out_dir / f"{p.stem}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{p.suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(p), str(out))
        return [str(out)]
    elif p.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in p.glob('*.csv'):
            out = out_dir / f"{f.stem}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{f.suffix}"
            shutil.copy2(str(f), str(out))
            copied.append(str(out))
        return copied
    else:
        raise FileNotFoundError(path)


def fetch_s3(uri, out_dir: Path):
    # lazy import boto3
    try:
        import boto3
    except ImportError:
        raise ImportError('boto3 required to fetch from s3; install with pip install boto3')
    s3 = boto3.client('s3')
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(key).name
    out_path = out_dir / f"{Path(filename).stem}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{Path(filename).suffix}"
    s3.download_file(bucket, key, str(out_path))
    return [str(out_path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Local path or s3:// URI')
    parser.add_argument('--out-dir', default='data/raw', help='Directory to place fetched files')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    parsed = urlparse(args.source)
    if parsed.scheme in ('s3',):
        files = fetch_s3(args.source, out_dir)
    else:
        files = fetch_local(args.source, out_dir)

    print('Fetched files:')
    for f in files:
        print(' -', f)


if __name__ == '__main__':
    main()
