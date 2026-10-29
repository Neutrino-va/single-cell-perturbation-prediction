"""Inspect the competition training parquet without modifying it."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('--out', type=Path, default=Path('数据审计结果.json'))
    args = parser.parse_args()
    path = args.file
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    parquet = pq.ParquetFile(path)
    frame = pd.read_parquet(path)
    required = ['cell_type', 'sm_name']
    missing = [name for name in required if name not in frame.columns]
    result = {'source_file': path.name, 'bytes': path.stat().st_size,
              'sha256': digest.hexdigest(), 'rows': len(frame),
              'columns': len(frame.columns), 'column_names': list(frame.columns),
              'schema': str(parquet.schema_arrow), 'missing_required_columns': missing,
              'missing_values_total': int(frame.isna().sum().sum())}
    if not missing:
        result['duplicate_cell_drug_pairs'] = int(frame.duplicated(required).sum())
        result['cell_type_counts'] = {str(k): int(v) for k, v in frame.cell_type.value_counts(dropna=False).items()}
        result['drug_count'] = int(frame.sm_name.nunique())
        result['drug_counts_by_cell_type'] = {str(k): int(v) for k, v in frame.groupby('cell_type').sm_name.nunique().items()}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('column_names', 'schema')}, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit('Required metadata columns are missing; inspect schema before proceeding.')


if __name__ == '__main__':
    main()
