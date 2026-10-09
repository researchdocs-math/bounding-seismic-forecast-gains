"""Numerical validation, information cutoffs, catalog diagnostics and skill resolution."""
from pathlib import Path
import json,concurrent.futures,sys,platform,importlib.metadata
import numpy as np,pandas as pd
from scipy.optimize._numdiff import approx_derivative
from etas import *
from features import day
from models import loss
ROOT=Path(__file__).resolve().parents[1]

def numerical(co):
 cells=[c for c in json.loads((ROOT/'data/processed/cells.json').read_text()) if c['country']==co];c=max(cells,key=lambda x:x['history_count']);cid=c['cell'];p=ROOT/f'results/etas_{co}_{cid}.json'
 if not p.exists():p=ROOT/f'data/inherited_etas/etas_{co}_{cid}.json'
 data=json.loads(p.read_text());rows=[];cat=pd.read_csv(ROOT/f'data/processed/{co}_catalog.csv.gz',dtype={'cell':str});cat=cat[cat.cell==cid].sort_values('t_days');t=cat.t_days.to_numpy();m=cat.magnitude.to_numpy()
 for year in [2011,2020,2024]:
  v=data[str(year)];par=np.array(v['params']);z=np.array(v['z']);s=(t>=v['start']-365)&(t<v['end']);val,g=objective_gradient(z,par[-1],t[s],m[s],v['start'],v['end']);num=approx_derivative(lambda a:objective_gradient(a,par[-1],t[s],m[s],v['start'],v['end'])[0],z,method='3-point').ravel();err=np.max(np.abs(g-num)/(1+np.abs(num)));assert err<1e-4
  o=day(f'{year}-07-01');j=(t>=o-365)&(t<o)
  for band,lo,hi in [('primary',3.5,8.5),('band',3.5,4.5),('large',4.5,8.5)]:
   pp=[]
   for dt,dm in [(.05,.025),(.025,.0125),(.0125,.0125)]:
    q=cluster_survival(par,lo,hi,dt=dt,dm=dm);pp.append(float(forecast_from_q(par,t[j],m[j],o,q,dt=dt)))
   rows.append(dict(country=co,cell=cid,year=year,target=band,gradient_error=float(err),p_standard=pp[0],p_refined=pp[-1],probability_difference=abs(pp[-1]-pp[0])))
 return rows

def skill_bounds():
 jobs=[('alaska','predictions_alaska_ge35.csv','borrowed','selected_baseline'),('chile','persistence_predictions_chile_ge35.csv','duration','baseline'),('japan','online_predictions_japan_ge35.csv','corrected','reference')];rows=[];device=[]
 for co,filename,model,base in jobs:
  d=pd.read_csv(ROOT/'results'/filename);z=loss(d.y,d[base])-loss(d.y,d[model]);a=pd.DataFrame({'origin':d.origin,'z':z}).groupby('origin').z.mean().sort_index().to_numpy();T=len(a);L=loss(d.y,d[base]).mean()
  for block in [3,6,12]:
   rng=np.random.default_rng(740321);idx=(rng.integers(0,T,(20000,int(np.ceil(T/block)),1))+np.arange(block))%T;idx=idx.reshape(20000,-1)[:,:T];weights=np.stack([(idx==j).sum(1)/T for j in range(T)],axis=1);cpu=weights@a
   try:
    import torch
    assert torch.backends.mps.is_available();gpu=(torch.tensor(weights,dtype=torch.float32,device='mps')@torch.tensor(a,dtype=torch.float32,device='mps')).cpu().numpy();error=float(np.max(np.abs(cpu-gpu)));assert error<1e-6;backend='GPU checked against CPU'
   except (ImportError,AssertionError,RuntimeError):error=None;backend='CPU'
   upper=float(np.quantile(cpu,1-.05/9));rows.append(dict(country=co,block=block,gain=float(z.mean()),reference_loss=float(L),upper_bonferroni9=upper,relative_upper_percent=100*upper/L));device.append(dict(country=co,block=block,backend=backend,max_abs_error=error))
 pd.DataFrame(rows).to_csv(ROOT/'results/skill_resolution.csv',index=False);(ROOT/'results/gpu_verification.json').write_text(json.dumps(device,indent=2));print(pd.DataFrame(rows).to_string(index=False),flush=True)

def data_audit():
 rows=[];types=[];annual=[]
 for co in ['italy','greece','alaska','chile','japan']:
  c=pd.read_csv(ROOT/f'data/processed/{co}_catalog.csv.gz',dtype={'cell':str});f=pd.read_csv(ROOT/f'data/processed/{co}_forecasts.csv.gz',dtype={'cell':str});dates=pd.to_datetime(f.origin);te=f[dates>=pd.Timestamp('2020-01-01')];c['date']=pd.to_datetime(c.time,utc=True,format='mixed');offset=1.5 if co in ['chile','japan'] else 0;assert not c.event_id.duplicated().any()
  rows.append(dict(country=co,events=len(c),eligible_events=int(c.eligible.sum()),cells=f.cell.nunique(),origins=f.origin.nunique(),test_n=len(te),positive_primary=int(te.y_ge35.sum()),positive_band=int(te.y_band35_45.sum()),positive_large=int(te.y_ge45.sum()),catalog_threshold=3+offset,primary_threshold=3.5+offset,large_threshold=4.5+offset))
  for k,v in c.magnitude_type.value_counts().items():types.append(dict(country=co,type=k,n=int(v)))
  for yr,x in c.groupby(c.date.dt.year):
   bins=np.arange(3.,8.6,.1);counts,_=np.histogram(x.magnitude,bins);maxc=bins[np.argmax(counts)]+.05+offset;annual.append(dict(country=co,year=int(yr),events=len(x),maximum_curvature_proxy=maxc,median_magnitude=float(x.magnitude.median()+offset),fraction_small=float((x.magnitude<3.5).mean())))
  # Recompute all target labels independently from catalog interval searches.
  for cid,g in f.groupby('cell'):
   x=c[c.cell==cid].sort_values('t_days');tt=x.t_days.to_numpy();mm=x.magnitude.to_numpy()
   for target,lo,hi in [('ge35',3.5,8.5),('band35_45',3.5,4.5),('ge45',4.5,8.5)]:
    times=tt[(mm>=lo)&(mm<hi)];n=np.searchsorted(times,g.origin_day+30,'left')-np.searchsorted(times,g.origin_day,'left');assert np.array_equal(n>0,g['y_'+target].to_numpy().astype(bool));assert np.array_equal(n,g['count_'+target])
 pd.DataFrame(rows).to_csv(ROOT/'results/data_audit.csv',index=False);pd.DataFrame(types).to_csv(ROOT/'results/magnitude_types.csv',index=False);pd.DataFrame(annual).to_csv(ROOT/'results/catalog_annual_audit.csv',index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':
 data_audit()
 with concurrent.futures.ProcessPoolExecutor(5) as pool:r=list(pool.map(numerical,['italy','greece','alaska','chile','japan']))
 pd.DataFrame([x for a in r for x in a]).to_csv(ROOT/'results/numerical_checks.csv',index=False);skill_bounds()
 env={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','numba','xgboost','statsmodels']}
 try:env['torch']=importlib.metadata.version('torch')
 except importlib.metadata.PackageNotFoundError:env['torch']='not installed (optional)'
 env.update(python=sys.version,system=platform.system(),machine=platform.machine());(ROOT/'results/environment.json').write_text(json.dumps(env,indent=2))
