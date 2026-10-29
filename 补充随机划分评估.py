"""Explore additional fixed random splits on the same competition training data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from 药物均值岭回归融合实验 import ALPHAS, encode, evaluate, load_data, score

SEEDS = (17, 314, 2718, 3407, 2027)


def split(features, seed):
    order = np.random.default_rng(seed).permutation(len(features))
    remaining = set(order.tolist())
    test, validation = [], []
    for destination in (test, validation):
        for index in order:
            if len(destination) >= round(len(features) * 0.15):
                break
            if index not in remaining:
                continue
            rest = list(remaining - {index})
            if (features.iloc[rest].cell_type == features.iloc[index].cell_type).sum() < 2:
                continue
            if (features.iloc[rest].sm_name == features.iloc[index].sm_name).sum() < 1:
                continue
            remaining.remove(index)
            destination.append(int(index))
    if len(test) != 92 or len(validation) != 92:
        raise ValueError('Unable to construct the requested split')
    return np.array(sorted(remaining)), np.array(validation), np.array(test)


def ridge_baseline(features, targets, train, validation, test):
    x_train, x_validation = encode(features, train, validation)
    choices = []
    for alpha in ALPHAS:
        prediction = Ridge(alpha=alpha).fit(x_train, targets[train]).predict(x_validation)
        choices.append((score(targets[validation], prediction), alpha))
    _, alpha = min(choices)
    combined = np.r_[train, validation]
    x_fit, x_test = encode(features, combined, test)
    prediction = Ridge(alpha=alpha).fit(x_fit, targets[combined]).predict(x_test)
    return score(targets[test], prediction), alpha


def main():
    features, targets = load_data()
    records, saved = [], {}
    for seed in SEEDS:
        train, validation, test = split(features, seed)
        result = evaluate(features, targets, train, validation, test)
        ridge_score, ridge_alpha = ridge_baseline(features, targets, train, validation, test)
        records.append({'seed': seed, **result, 'ridge_test': ridge_score, 'ridge_alpha': ridge_alpha})
        saved[str(seed)] = {'train': train.tolist(), 'validation': validation.tolist(), 'test': test.tolist()}
        print(seed, 'done', flush=True)
    Path('实验结果/补充随机划分.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
    output = pd.DataFrame(records)
    output.to_csv('实验结果/补充随机划分评估.csv', index=False)
    print(output[['test_mrrmse', 'ridge_test', 'drug_mean_test']].agg(['mean', 'std']).round(5))


if __name__ == '__main__':
    main()
