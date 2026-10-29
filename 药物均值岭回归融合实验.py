"""Fit and evaluate a validation-selected drug-mean/ridge ensemble."""
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

METADATA = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_data(path=Path('数据/de_train.parquet')):
    frame = pd.read_parquet(path)
    features = frame[['cell_type', 'sm_name']].astype(str)
    targets = frame.drop(columns=METADATA).to_numpy(dtype=np.float32)
    if features.duplicated().any() or not np.isfinite(targets).all():
        raise ValueError('Duplicate combinations or non-finite target values')
    return features, targets


def score(targets, predictions):
    return float(np.sqrt(np.mean((targets - predictions) ** 2, axis=1)).mean())


def drug_mean(features, targets, train, test):
    groups = pd.DataFrame(targets[train]).groupby(
        features.iloc[train].sm_name.reset_index(drop=True)
    ).mean()
    fallback = targets[train].mean(axis=0)
    return np.stack([
        groups.loc[drug].to_numpy() if drug in groups.index else fallback
        for drug in features.iloc[test].sm_name
    ]).astype(np.float32)


def encode(features, train, test):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    return encoder.fit_transform(features.iloc[train]), encoder.transform(features.iloc[test])


def evaluate(features, targets, train, validation, test):
    indices = [set(map(int, part)) for part in (train, validation, test)]
    if indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2]:
        raise ValueError('Train, validation and test indices must not overlap')
    x_train, x_validation = encode(features, train, validation)
    mean_validation = drug_mean(features, targets, train, validation)
    candidates = []
    for alpha in ALPHAS:
        ridge = Ridge(alpha=alpha).fit(x_train, targets[train])
        ridge_validation = ridge.predict(x_validation)
        for weight in WEIGHTS:
            prediction = weight * ridge_validation + (1 - weight) * mean_validation
            candidates.append((score(targets[validation], prediction), weight, alpha))

    validation_score, weight, alpha = min(candidates)
    combined = np.r_[train, validation]
    x_fit, x_test = encode(features, combined, test)
    mean_test = drug_mean(features, targets, combined, test)
    ridge_test = Ridge(alpha=alpha).fit(x_fit, targets[combined]).predict(x_test)
    prediction = weight * ridge_test + (1 - weight) * mean_test
    return {
        'n_train_refit': len(combined), 'n_test': len(test),
        'ridge_weight': weight, 'alpha': alpha,
        'validation_mrrmse': validation_score,
        'test_mrrmse': score(targets[test], prediction),
        'drug_mean_test': score(targets[test], mean_test),
        'ridge_same_alpha_test': score(targets[test], ridge_test),
    }


def main():
    features, targets = load_data()
    splits = {'组合留出': json.loads(Path('实验结果/划分.json').read_text(encoding='utf-8'))}
    splits.update(json.loads(Path('实验结果/多细胞少样本划分.json').read_text(encoding='utf-8')))
    records = []
    for scenario, scenario_splits in splits.items():
        for seed in (42, 123, 2026):
            split = scenario_splits[str(seed)]
            parts = [np.array(split[key]) for key in ('train', 'validation', 'test')]
            result = evaluate(features, targets, *parts)
            records.append({'scenario': scenario, 'seed': seed, **result})
            print(scenario, seed, 'done', flush=True)
    result = pd.DataFrame(records)
    result.to_csv('实验结果/药物均值岭回归融合.csv', index=False)
    print(result.groupby('scenario')[['test_mrrmse', 'drug_mean_test']].agg(['mean', 'std']))


if __name__ == '__main__':
    main()
