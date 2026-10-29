"""Few-shot held-out combinations and a compact MLP; outputs are research records."""
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
meta=['cell_type','sm_name','sm_lincs_id','SMILES','control']
Y=df.drop(columns=meta).to_numpy(dtype=np.float32)
X=df[['cell_type','sm_name']].astype(str)
saved=json.loads(Path('实验结果/划分.json').read_text(encoding='utf-8'))
def score(a,b): return float(np.sqrt(((a-b)**2).mean(axis=1)).mean())
out=[]; splitout={}; percell=[]
for seed in [42,123,2026]:
    common=saved[str(seed)]
    target='NK cells'
    idx=np.flatnonzero(X.cell_type.to_numpy()==target)
    rng=np.random.default_rng(seed); perm=rng.permutation(idx)
    # 15 target rows train, 15 validation, remainder test; other types train.
    tr=np.r_[np.flatnonzero(X.cell_type.to_numpy()!=target),perm[:15]]
    va=perm[15:30]; te=perm[30:]
    assert set(X.iloc[te].sm_name)<=set(X.iloc[tr].sm_name)
    splitout[str(seed)]={'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()}
    for setting,(tr,va,te) in {'普通':tuple(np.array(common[k]) for k in ['train','validation','test']),'少样本':(tr,va,te)}.items():
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        a=enc.fit_transform(X.iloc[tr]); b=enc.transform(X.iloc[va]); c=enc.transform(X.iloc[te])
        if setting=='少样本':
            joined=np.r_[tr,va]
            def group_mean(group):
                grouped=pd.DataFrame(Y[joined]).groupby(X.iloc[joined][group].reset_index(drop=True)).mean()
                fallback=Y[joined].mean(axis=0)
                return np.stack([grouped.loc[k].to_numpy() if k in grouped.index else fallback for k in X.iloc[te][group]])
            for name,pred in [('zero',np.zeros_like(Y[te])),('global_mean',np.broadcast_to(Y[joined].mean(0),Y[te].shape)),('drug_mean',group_mean('sm_name')),('cell_mean',group_mean('cell_type'))]:
                out.append(dict(seed=seed,setting=setting,model=name,mrrmse=score(Y[te],pred),config=''))
            for name,cols in [('ridge_both',['cell_type','sm_name']),('ridge_drug_only',['sm_name']),('ridge_cell_only',['cell_type'])]:
                e=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
                ta=e.fit_transform(X.iloc[tr][cols]); vb=e.transform(X.iloc[va][cols])
                best=min((score(Y[va],Ridge(alpha=alpha).fit(ta,Y[tr]).predict(vb)),alpha) for alpha in [.01,.1,1.,10.,100.])[1]
                e=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
                fit=e.fit_transform(X.iloc[joined][cols]); test=e.transform(X.iloc[te][cols])
                pred=Ridge(alpha=best).fit(fit,Y[joined]).predict(test)
                out.append(dict(seed=seed,setting=setting,model=name,mrrmse=score(Y[te],pred),config=str(best)))
        # One increment per epoch; use only validation for epoch selection. No test feedback.
        mlp=MLPRegressor(hidden_layer_sizes=(32,),activation='relu',solver='adam',batch_size=128,learning_rate_init=.001,alpha=1.,max_iter=1,warm_start=False,random_state=seed)
        bestval=float('inf'); bestpred=None; bestepoch=None
        for epoch in range(1,61):
            mlp.partial_fit(a,Y[tr])
            if epoch in [10,20,40,60]:
                val=score(Y[va],mlp.predict(b))
                if val<bestval: bestval=val;bestepoch=epoch;bestpred=mlp.predict(c)
        out.append(dict(seed=seed,setting=setting,model='mlp_32',mrrmse=score(Y[te],bestpred),config=f'epoch={bestepoch};val={bestval:.6f}'))
        if setting=='普通':
            e=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
            joined=np.r_[tr,va]; fit=e.fit_transform(X.iloc[joined]); test=e.transform(X.iloc[te]); pred=Ridge(alpha=10).fit(fit,Y[joined]).predict(test)
            err=np.sqrt(((Y[te]-pred)**2).mean(axis=1))
            for cell in sorted(X.iloc[te].cell_type.unique()):
                mask=(X.iloc[te].cell_type.to_numpy()==cell)
                percell.append(dict(seed=seed,cell_type=cell,n=int(mask.sum()),mrrmse=float(err[mask].mean())))
        print(seed,setting,'done',flush=True)
Path('实验结果/补充划分.json').write_text(json.dumps(splitout,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(out).to_csv('实验结果/补充实验.csv',index=False)
pd.DataFrame(percell).to_csv('实验结果/细胞分组误差.csv',index=False)
print(pd.DataFrame(out).groupby(['setting','model']).mrrmse.agg(['mean','std']).to_string())
