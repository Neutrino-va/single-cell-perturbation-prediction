"""Post-hoc association only, not a test of mechanism."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
df=pd.read_parquet('数据/de_train.parquet')
X=df[['cell_type','sm_name']].astype(str)
Y=df.drop(columns=['cell_type','sm_name','sm_lincs_id','SMILES','control']).to_numpy(dtype=np.float32)
splits=json.loads(Path('实验结果/多细胞少样本划分.json').read_text(encoding='utf-8'))
records=[]
for target in splits:
    ids=splits[target]['42']; joined=np.array(ids['train']+ids['validation']); te=np.array(ids['test'])
    group=pd.DataFrame(Y[joined]).groupby(X.iloc[joined].sm_name.reset_index(drop=True)).mean()
    prediction=np.stack([group.loc[k].to_numpy() for k in X.iloc[te].sm_name])
    a=Y[te]-Y[te].mean(axis=1,keepdims=True); b=prediction-prediction.mean(axis=1,keepdims=True)
    denom=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1)); r=np.divide((a*b).sum(axis=1),denom,out=np.full(len(te),np.nan),where=denom>0)
    for index,corr in zip(te,r): records.append({'target':target,'seed':42,'row_index':int(index),'drug':X.iloc[index].sm_name,'pearson':float(corr)})
out=pd.DataFrame(records);out.to_csv('实验结果/跨细胞相关性探索.csv',index=False)
print(out.groupby('target').pearson.agg(['count','mean','median','min','max']).round(4).to_string())
