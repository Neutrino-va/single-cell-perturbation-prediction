"""Fit the chosen fusion configuration on all rows and save a local artifact."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from 药物均值岭回归融合实验 import METADATA


def train(data_path, config_path):
    frame = pd.read_parquet(data_path)
    features = frame[['cell_type', 'sm_name']].astype(str)
    gene_names = [column for column in frame.columns if column not in METADATA]
    targets = frame[gene_names].to_numpy(dtype=np.float32)
    if features.duplicated().any() or not np.isfinite(targets).all():
        raise ValueError('Duplicate combinations or non-finite values in data')
    results = pd.read_csv(config_path)
    chosen = results[(results.scenario == '组合留出') & (results.seed == 42)]
    if len(chosen) != 1:
        raise ValueError('Expected exactly one combination-split configuration for seed 42')
    alpha = float(chosen.iloc[0].alpha)
    weight = float(chosen.iloc[0].ridge_weight)
    encoder = OneHotEncoder(handle_unknown='error', sparse_output=False, dtype=np.float32)
    x = encoder.fit_transform(features)
    ridge = Ridge(alpha=alpha).fit(x, targets)
    grouped = pd.DataFrame(targets).groupby(features.sm_name.reset_index(drop=True)).mean()
    return {
        'format_version': 1, 'encoder': encoder, 'ridge': ridge,
        'drug_names': grouped.index.to_numpy(), 'drug_means': grouped.to_numpy(dtype=np.float32),
        'gene_names': gene_names, 'ridge_weight': weight, 'alpha': alpha,
        'training_rows': len(frame), 'configuration_source': '组合留出 seed 42',
        'source_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }, features


def predict(model, features):
    features = features[['cell_type', 'sm_name']].astype(str)
    lookup = {name: i for i, name in enumerate(model['drug_names'])}
    missing = sorted(set(features.sm_name) - lookup.keys())
    if missing:
        raise ValueError('Unknown drugs are unsupported: ' + ', '.join(missing))
    encoded = model['encoder'].transform(features)
    ridge = model['ridge'].predict(encoded)
    drug = model['drug_means'][[lookup[name] for name in features.sm_name]]
    weight = model['ridge_weight']
    return (1 - weight) * drug + weight * ridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('数据/de_train.parquet'))
    parser.add_argument('--config', type=Path, default=Path('实验结果/药物均值岭回归融合.csv'))
    parser.add_argument('--output', type=Path, default=Path('模型/fusion-seed42.joblib'))
    args = parser.parse_args()
    model, features = train(args.data, args.config)
    expected = predict(model, features.iloc[:3])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output, compress=3)
    restored = joblib.load(args.output)
    actual = predict(restored, features.iloc[:3])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    manifest = {key: model[key] for key in ('format_version', 'ridge_weight', 'alpha', 'training_rows', 'configuration_source', 'source_sha256')}
    manifest.update({'gene_count': len(model['gene_names']), 'file_size_bytes': args.output.stat().st_size})
    args.output.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('Reload and prediction consistency check passed.')


if __name__ == '__main__':
    main()
