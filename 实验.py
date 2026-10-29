"""Reproducible held-out cell/drug-combination comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_parquet('数据/de_train.parquet')
meta = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']
genes = [c for c in df.columns if c not in meta]
assert len(genes) == 18211 and not df.duplicated(['cell_type', 'sm_name']).any()
y = df[genes].to_numpy(dtype=np.float32)
assert np.isfinite(y).all()
X = df[['cell_type', 'sm_name']].astype(str)

def score(a, b):
    return float(np.mean(np.sqrt(np.mean((a-b)**2, axis=1))))

def split(seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df)); train = set(order.tolist()); test=[]; val=[]
    def take(dest, target):
        for i in order:
            if len(dest) >= target: break
            if i not in train: continue
            rem = list(train - {i})
            if (X.iloc[rem].cell_type == X.iloc[i].cell_type).sum() < 2: continue
            if (X.iloc[rem].sm_name == X.iloc[i].sm_name).sum() < 1: continue
            train.remove(i); dest.append(int(i))
    take(test, round(len(df)*.15)); take(val, round(len(df)*.15))
    if len(test) < round(len(df)*.15) or len(val) < round(len(df)*.15): raise ValueError('Unable to produce requested split')
    return np.array(sorted(train)),np.array(val),np.array(test)

def means(train, test, group):
    means = pd.DataFrame(y[train]).groupby(X.iloc[train][group].reset_index(drop=True)).mean()
    fallback=y[train].mean(axis=0)
    return np.stack([means.loc[k].to_numpy() if k in means.index else fallback for k in X.iloc[test][group]])

results=[]; splits={}
for seed in [42,123,2026]:
    tr,va,te=split(seed); splits[str(seed)]={'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()}
    combined=np.r_[tr,va]
    for name,pred in [('zero',np.zeros_like(y[te])),('global_mean',np.tile(y[combined].mean(0),(len(te),1))),('drug_mean',means(combined,te,'sm_name')),('cell_mean',means(combined,te,'cell_type'))]:
        results.append({'seed':seed,'model':name,'mrrmse':score(y[te],pred),'alpha':None})
    for name,cols in [('ridge_both',['cell_type','sm_name']),('ridge_cell_only',['cell_type']),('ridge_drug_only',['sm_name'])]:
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        a=enc.fit_transform(X.iloc[tr][cols]); b=enc.transform(X.iloc[va][cols]);
        candidates=[]
        for alpha in [0.01,0.1,1.,10.,100.]:
            model=Ridge(alpha=alpha).fit(a,y[tr]); candidates.append((score(y[va],model.predict(b)),alpha))
        _,alpha=min(candidates)
        combined=np.r_[tr,va]; enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        a=enc.fit_transform(X.iloc[combined][cols]); b=enc.transform(X.iloc[te][cols]);
        model=Ridge(alpha=alpha).fit(a,y[combined])
        results.append({'seed':seed,'model':name,'mrrmse':score(y[te],model.predict(b)),'alpha':alpha})
Path('实验结果').mkdir(exist_ok=True)
Path('实验结果/划分.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
pd.DataFrame(results).to_csv('实验结果/模型结果.csv',index=False)
print(pd.DataFrame(results).groupby('model').mrrmse.agg(['mean','std']).round(5).to_string())
