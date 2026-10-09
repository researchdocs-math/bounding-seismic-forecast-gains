"""Annual ETAS fits, catalog features and causally completed forecast memory."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,concurrent.futures,time
import numpy as np,pandas as pd
from scipy.special import logit
from etas import fit,compensator,cluster_survival,forecast_from_q,MC
from features import feature_vector,day,EPOCH
ROOT=Path(__file__).resolve().parents[1]
BANDS=[('ge35',3.5,8.5),('band35_45',3.5,4.5),('ge45',4.5,8.5)]

def run_cell(cell):
 co=cell['country'];cid=cell['cell'];out=ROOT/f'results/forecast_{co}_{cid}.csv';modelout=ROOT/f'results/etas_{co}_{cid}.json'
 if out.exists() and modelout.exists():return f'{co} {cid} cached'
 cat=pd.read_csv(ROOT/f'data/processed/{co}_catalog.csv.gz',dtype={'cell':str});cat=cat[cat.cell==cid].sort_values(['t_days','event_id'])
 t=cat.t_days.to_numpy();m=cat.magnitude.to_numpy();rows=[];models={};warm=None;lastyear=None;startclock=time.time()
 for o in np.arange(day('2010-01-01'),day('2026-01-01')-30+1,30.):
  year=(EPOCH+pd.Timedelta(days=o)).year
  if year!=lastyear:
   model=fit(t,m,day(f'{year-5}-01-01'),day(f'{year}-01-01'),warm);warm=model['z'];par=np.array(model['params']);models[str(year)]=model;lastyear=year
   qs={name:cluster_survival(par,lo,hi) for name,lo,hi in BANDS}
  hi=np.searchsorted(t,o,'left');lo=np.searchsorted(t,o-365,'left');hist=cat.iloc[:hi]
  f=feature_vector(hist,o,cell,par)
  row={'country':co,'cell':cid,'origin':str((EPOCH+pd.Timedelta(days=o)).date()),'origin_day':o,'year':year,'center_lat':cell['center_lat'],'center_lon':cell['center_lon'],**f}
  row.update(dict(zip(['mu','K','alpha','c','omori_p','branching','beta'],par)))
  for name,lower,upper in BANDS:
   row['etas_'+name]=forecast_from_q(par,t[lo:hi],m[lo:hi],o,qs[name]);target=(m>=lower)&(m<upper)
   row['y_'+name]=int(np.any(target&(t>=o)&(t<o+30)));row['count_'+name]=int(np.sum(target&(t>=o)&(t<o+30)))
   long=target&(t<o)&(t>=o-1826.25);rate=(long.sum()+.5)/1827.25
   row['poisson_'+name]=-np.expm1(-30*rate)
   for w in [90,365]:
    n=np.sum(target&(t>=o-w)&(t<o));lam=(n+365*rate)/(w+365);row[f'adaptive{w}_{name}']=-np.expm1(-30*lam)
   # Every preceding forecast is exactly 30 days long and completed by current origin.
   for w in [3,6,12,24]:
    old=rows[-w:]; yy=np.array([r['y_'+name] for r in old]);pp=np.array([r['etas_'+name] for r in old]);err=yy-pp
    row[f'memory{w}_{name}']=float(np.arcsinh(err.sum()/np.sqrt((pp*(1-pp)).sum()+1)))
    row[f'frequency{w}_{name}']=float((yy.sum()+6*row['etas_'+name])/(len(old)+6))
   row['memory_persist_'+name]=float(min(abs(row['memory6_'+name]),abs(row['memory24_'+name]))*np.sign(row['memory6_'+name])) if row['memory6_'+name]*row['memory24_'+name]>0 else 0.
  rows.append(row)
 pd.DataFrame(rows).to_csv(out,index=False);modelout.write_text(json.dumps(models,indent=2));return f'{co} {cid} {len(rows)} origins {time.time()-startclock:.1f}s'

def main():
 cells=json.loads((ROOT/'data/processed/cells.json').read_text())
 with concurrent.futures.ProcessPoolExecutor(8) as pool:
  futures=[pool.submit(run_cell,c) for c in cells]
  for f in concurrent.futures.as_completed(futures):print(f.result(),flush=True)
 d=pd.concat([pd.read_csv(ROOT/f'results/forecast_{c["country"]}_{c["cell"]}.csv',dtype={'cell':str}) for c in cells],ignore_index=True)
 for co in d.country.unique():
  z=d[d.country==co].copy();z.to_csv(ROOT/f'data/processed/{co}_forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 print('DONE',len(d),flush=True)
if __name__=='__main__':main()
