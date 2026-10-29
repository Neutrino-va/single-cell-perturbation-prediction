"""Predict one known cell type and drug using a locally generated model."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from 训练保存模型 import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('模型/fusion-seed42.joblib'))
    parser.add_argument('--cell', required=True)
    parser.add_argument('--drug', required=True)
    parser.add_argument('--output', type=Path, default=Path('预测结果.csv'))
    args = parser.parse_args()
    model = joblib.load(args.model)
    features = pd.DataFrame([{'cell_type': args.cell, 'sm_name': args.drug}])
    values = predict(model, features)[0]
    pd.DataFrame({'gene': model['gene_names'], 'predicted_response': values}).to_csv(args.output, index=False)
    print(f'Saved {len(values)} gene predictions to {args.output}')


if __name__ == '__main__':
    main()
