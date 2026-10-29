"""Exploratory comparison across target cell types on fresh fixed splits."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

df=pd.read_parquet('数据/de_train.parquet'); x=df[['cell_type','sm_name']].astype(str)
y=df.drop(columns=['cell_type','sm_name','sm_lincs_id','SMILES','control']).to_numpy(dtype=np.float32)
def score(a,b):return float(np.sqrt(np.mean((a-b)**2,axis=1)).mean())
records=[]; splits={}
for target in ['NK cells','T cells CD4+','T regulatory cells','T cells CD8+']:
    splits[target]={}
    for seed in [42,123,2026]:
        idx=np.flatnonzero(x.cell_type.to_numpy()==target); perm=np.random.default_rng(seed).permutation(idx)
        tr=np.r_[np.flatnonzero(x.cell_type.to_numpy()!=target),perm[:15]]; va=perm[15:30]; te=perm[30:]; joined=np.r_[tr,va]
        assert set(x.iloc[te].sm_name)<=set(x.iloc[tr].sm_name)
        splits[target][str(seed)]={'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()}
        means=pd.DataFrame(y[joined]).groupby(x.iloc[joined].sm_name.reset_index(drop=True)).mean()
        pred=np.stack([means.loc[k].to_numpy() for k in x.iloc[te].sm_name])
        records.append(dict(target=target,seed=seed,model='drug_mean',n_test=len(te),mrrmse=score(y[te],pred),alpha=None))
        for name,cols in [('ridge_both',['cell_type','sm_name']),('ridge_drug_only',['sm_name'])]:
            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
            a=enc.fit_transform(x.iloc[tr][cols]);b=enc.transform(x.iloc[va][cols])
            alpha=min((score(y[va],Ridge(alpha=i).fit(a,y[tr]).predict(b)),i) for i in [.01,.1,1.,10.,100.])[1]
            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
            a=enc.fit_transform(x.iloc[joined][cols]);b=enc.transform(x.iloc[te][cols])
            pred=Ridge(alpha=alpha).fit(a,y[joined]).predict(b)
            records.append(dict(target=target,seed=seed,model=name,n_test=len(te),mrrmse=score(y[te],pred),alpha=alpha))
        print(target,seed,flush=True)
Path('实验结果/多细胞少样本划分.json').write_text(json.dumps(splits,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(records).to_csv('实验结果/多细胞少样本探索.csv',index=False)
print(pd.DataFrame(records).groupby(['target','model']).mrrmse.agg(['mean','std']).to_string())
