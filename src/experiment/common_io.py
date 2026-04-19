import csv
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float], int]:
    vals = [safe_float(v) for v in values if safe_float(v) is not None]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, var ** 0.5, n


def read_rows_csv(csv_path: str) -> List[Dict[str, Any]]:
    if not csv_path or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        try:
            return list(csv.DictReader(f))
        except Exception:
            return []


def append_rows_csv(csv_path: str, rows: List[Dict[str, Any]], preset_fieldnames: Optional[List[str]] = None) -> None:
    if not csv_path:
        return
    ensure_dir(os.path.dirname(csv_path) or '.')
    rows = rows or []
    fieldnames: List[str] = list(preset_fieldnames or [])
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            try:
                existing = csv.DictReader(f)
                existing_fieldnames = list(existing.fieldnames or [])
                for name in existing_fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
            except Exception:
                pass
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        return

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if not file_exists:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fieldnames})
        return

    old_rows = read_rows_csv(csv_path)
    old_fieldnames = list(old_rows[0].keys()) if old_rows else fieldnames
    if old_fieldnames != fieldnames:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in old_rows + rows:
                w.writerow({k: row.get(k) for k in fieldnames})
        return

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def write_rows_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not csv_path:
        return
    ensure_dir(os.path.dirname(csv_path) or '.')
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if not fieldnames:
            f.write('')
            return
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})
