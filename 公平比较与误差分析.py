"""Additional post-hoc analysis on fixed splits; no test labels select models."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

df=pd.read_parquet('数据/de_train.parquet')
X=df[['cell_type','sm_name']].astype(str)
y=df.drop(columns=['cell_type','sm_name','sm_lincs_id','SMILES','control']).to_numpy(dtype=np.float32)
splitsets={'普通':json.loads(Path('实验结果/划分.json').read_text(encoding='utf-8')),
           '少样本':json.loads(Path('实验结果/补充划分.json').read_text(encoding='utf-8'))}
prior=pd.read_csv('实验结果/补充实验.csv')
def row_error(a,b): return np.sqrt(np.mean((a-b)**2,axis=1))
outputs=[]; paired=[]
for setting,splits in splitsets.items():
    for seed in (42,123,2026):
        tr,va,te=[np.array(splits[str(seed)][x]) for x in ('train','validation','test')]
        joined=np.r_[tr,va]
        row=prior[(prior.setting==setting)&(prior.seed==seed)&(prior.model=='mlp_32')].iloc[0]
        epoch=int(row.config.split(';')[0].split('=')[1])
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        xx=enc.fit_transform(X.iloc[joined]); xt=enc.transform(X.iloc[te])
        model=MLPRegressor(hidden_layer_sizes=(32,),activation='relu',solver='adam',batch_size=128,learning_rate_init=.001,alpha=1.,max_iter=1,random_state=seed)
        for _ in range(epoch): model.partial_fit(xx,y[joined])
        pred=model.predict(xt)
        outputs.append({'setting':setting,'seed':seed,'train_rows':len(joined),'target_nk_train':int((X.iloc[joined].cell_type=='NK cells').sum()),'selected_epoch':epoch,'mrrmse':float(row_error(y[te],pred).mean())})
        if setting=='少样本':
            means=pd.DataFrame(y[joined]).groupby(X.iloc[joined].sm_name.reset_index(drop=True)).mean()
            dm=np.stack([means.loc[k].to_numpy() for k in X.iloc[te].sm_name])
            alpha=float(prior[(prior.setting==setting)&(prior.seed==seed)&(prior.model=='ridge_both')].config.iloc[0])
            ridge=Ridge(alpha=alpha).fit(xx,y[joined]); rp=ridge.predict(xt)
            errs=row_error(y[te],dm); errs_r=row_error(y[te],rp); errs0=row_error(y[te],np.zeros_like(y[te]))
            for i,index in enumerate(te):
                paired.append({'seed':seed,'row_index':int(index),'drug':X.iloc[index].sm_name,'drug_mean_error':float(errs[i]),'ridge_error':float(errs_r[i]),'zero_error':float(errs0[i])})
        print(setting,seed,'complete',flush=True)
pd.DataFrame(outputs).to_csv('实验结果/MLP等预算重训.csv',index=False)
pd.DataFrame(paired).to_csv('实验结果/NK逐组合误差.csv',index=False)
print(pd.DataFrame(outputs).groupby('setting').mrrmse.agg(['mean','std']))
q=pd.DataFrame(paired);print(q.groupby('seed').apply(lambda d:pd.Series({'drug_mean_wins':int((d.drug_mean_error<d.ridge_error).sum()),'n':len(d),'drug_vs_zero_wins':int((d.drug_mean_error<d.zero_error).sum())}),include_groups=False))
