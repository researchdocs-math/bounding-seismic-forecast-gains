"""Post-evaluation diagnostic: known persistent probability drift versus no drift."""
from pathlib import Path
import json,concurrent.futures
import numpy as np,pandas as pd
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1]

def replicate(job):
 scale,seed=job;rng=np.random.default_rng(seed);T=194;C=20;base=np.linspace(.04,.4,C);ell=logit(base);state=np.empty(T);state[0]=rng.normal()
 for i in range(1,T):state[i]=.95*state[i-1]+np.sqrt(1-.95**2)*rng.normal()
 truth=expit(ell[None,:]+scale*state[:,None]);y=rng.binomial(1,truth);pred=np.tile(base,(T,1));est=np.zeros(T)
 for i in range(1,T):
  a=0.;past=y[max(0,i-6):i];n=len(past)
  for it in range(15):
   p=expit(ell+a);g=(past-p).sum()-4*a;h=n*np.sum(p*(1-p))+4;step=g/h;a=float(np.clip(a+step,-4,4))
   if abs(step)<1e-9:break
  pred[i]=expit(ell+a);est[i]=a
 yy=y[-72:];ref=np.tile(base,(72,1));pp=pred[-72:];ll=lambda q:float(np.mean(-yy*np.log(q)-(1-yy)*np.log1p(-q)));return dict(scale=scale,seed=seed,reference=ll(ref),corrected=ll(pp),gain=ll(ref)-ll(pp))
if __name__=='__main__':
 with concurrent.futures.ProcessPoolExecutor(8) as pool:rows=list(pool.map(replicate,[(s,84000+i) for s in [0.,.25,.5,1.] for i in range(200)]))
 d=pd.DataFrame(rows);d.to_csv(ROOT/'results/synthetic_replicates.csv',index=False);z=d.groupby('scale').gain.agg(['mean','std','count']).reset_index();z['mc_lower']=z['mean']-1.96*z['std']/np.sqrt(z['count']);z['mc_upper']=z['mean']+1.96*z['std']/np.sqrt(z['count']);z.to_csv(ROOT/'results/synthetic_summary.csv',index=False)
 # Future-label perturbation must not affect current or earlier corrections.
 from online_calibration import correct
 x=pd.read_csv(ROOT/'results/issued_japan_ge35.csv',dtype={'cell':str});dates=sorted(x.origin.unique());cut=dates[80];a,_=correct(x,'ratebank_fiveyear','own',24,16);v=x.copy();v.loc[v.origin>=cut,'y']=1-v.loc[v.origin>=cut,'y'];b,_=correct(v,'ratebank_fiveyear','own',24,16);assert np.array_equal(a[x.origin<=cut],b[x.origin<=cut]);delta=float(np.max(np.abs(a[x.origin<=cut]-b[x.origin<=cut])))
 (ROOT/'results/online_causality_check.json').write_text(json.dumps(dict(cutoff=cut,max_difference_through_current_forecast=delta,changed_future_labels=int((x.origin>=cut).sum())),indent=2));print(z.to_string(index=False))
