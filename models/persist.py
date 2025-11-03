import joblib
import json
from pathlib import Path
from datetime import datetime


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_model(model, path: str, metadata: dict):
    """Save model with joblib and write metadata JSON next to it.

    Args:
        model: any picklable object
        path: full path to .joblib file
        metadata: dict containing metadata fields
    """
    ensure_parent(path)
    joblib.dump(model, path)
    meta_path = Path(path).with_suffix('.meta.json')
    metadata_out = metadata.copy()
    metadata_out.setdefault('saved_at', datetime.utcnow().isoformat() + 'Z')
    with open(meta_path, 'w') as f:
        json.dump(metadata_out, f, indent=2)


def load_model(path: str):
    return joblib.load(path)


def latest_model_for_department(registry_dir: str, department: str):
    """Return path to latest model.joblib for a department by timestamped filenames.
    Assumes path pattern: {registry_dir}/{department}/model_v{YYYYMMDDHHMMSS}.joblib
    """
    from glob import glob
    import re

    pattern = f"{registry_dir}/{department}/*.joblib"
    files = glob(pattern)
    if not files:
        return None
    # sort by file modification time
    files_sorted = sorted(files, key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return files_sorted[0]
