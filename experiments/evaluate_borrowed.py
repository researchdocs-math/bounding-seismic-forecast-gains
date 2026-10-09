"""Locked chronological evaluation, annual updating and synchronized uncertainty."""
from borrowed_memory import *
from scipy.special import expit
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

def job_eval(job):
 co,t=job;d=dataset(co);spec=json.loads((ROOT/'results/frozen_selection.json').read_text());ss={r['family']:r for r in spec if r['country']==co and r['target']==t};assert len(ss)==len(FAMS)
 train=d[d.end<pd.Timestamp('2020-01-01')];test=d[d.date>=pd.Timestamp('2020-01-01')].copy();out=test[['country','cell','origin','year','center_lat','center_lon','y_'+t,'etas_'+t]].copy().rename(columns={'y_'+t:'y','etas_'+t:'raw_etas'});out['target']=t
 for f,r in ss.items():
  model=Learner(r['columns'],r['config'],t).fit(train,train['y_'+t]);out[f]=model.predict(test)
  for yr in sorted(test.year.unique()):
   tr=d[d.end<pd.Timestamp(f'{yr}-01-01')];te=d[d.year==yr];m=Learner(r['columns'],r['config'],t).fit(tr,tr['y_'+t]);out.loc[out.year==yr,'annual_'+f]=m.predict(te)
 for ref in ['control','blend']:
  r=ss['borrowed'];cc=cols(ref,t,d);m=Learner(cc,r['config'],t).fit(train,train['y_'+t]);out['matched_'+ref]=m.predict(test)
  for yr in sorted(test.year.unique()):
   tr=d[d.end<pd.Timestamp(f'{yr}-01-01')];te=d[d.year==yr];m=Learner(cc,r['config'],t).fit(tr,tr['y_'+t]);out.loc[out.year==yr,'annual_matched_'+ref]=m.predict(te)
 # Selection of this reference uses validation only.
 allowed=['etas_cal','blend','original','local','neighbor','control'];chosen=min(allowed,key=lambda f:ss[f]['validation_loss']);out['selected_baseline']=out[chosen];out['annual_selected_baseline']=out['annual_'+chosen];out['baseline_name']=chosen
 out.to_csv(ROOT/f'results/predictions_{co}_{t}.csv',index=False);print('EVALUATED',co,t,'baseline',chosen,flush=True);return co,t

def bootstrap(delta,block,seed=740321):
 z=np.asarray(delta,float);T=len(z);rng=np.random.default_rng(seed);ix=(rng.integers(0,T,(20000,int(np.ceil(T/block)),1))+np.arange(block))%T;ix=ix.reshape(20000,-1)[:,:T];b=z[ix].mean(1);mu=z.mean(0);return mu,np.quantile(b,.025,axis=0),np.quantile(b,.975,axis=0),(1+(np.abs(b-mu)>=np.abs(mu)).sum(0))/20001

def summarize(countries):
 metrics=[];pairs=[];breakdowns=[];reliability=[];years=[];cells=[]
 for co in countries:
  for t in TARGETS:
   d=pd.read_csv(ROOT/f'results/predictions_{co}_{t}.csv');y=d.y.to_numpy();models=['raw_etas']+FAMS+['matched_control','matched_blend','selected_baseline'];models+=['annual_'+f for f in FAMS+['matched_control','matched_blend','selected_baseline']]
   for f in models:
    p=d[f].to_numpy();z=loss(y,p);metrics.append(dict(country=co,target=t,model=f,n=len(y),positive=int(y.sum()),log_loss=z.mean(),brier=np.mean((y-p)**2),mean_p=p.mean(),observed=y.mean(),auc=roc_auc_score(y,p)))
    if f in ['etas_cal','blend','control','borrowed','annual_borrowed','annual_control']:
     bins=pd.qcut(p,8,duplicates='drop');a=pd.DataFrame(dict(p=p,y=y,bin=bins));b=a.groupby('bin',observed=True).agg(n=('y','size'),positive=('y','sum'),predicted=('p','mean'),observed=('y','mean')).reset_index(drop=True)
     for r in b.to_dict('records'):reliability.append(dict(country=co,target=t,model=f,**r))
   for regime in ['','annual_']:
    for f in ['original','local','neighbor','control','borrowed','all']:
     refs=['etas_cal','blend','selected_baseline']+(['control','matched_control','matched_blend'] if f=='borrowed' else [])
     for ref in refs:
      z=loss(y,d[regime+ref])-loss(y,d[regime+f]);zz=pd.DataFrame(dict(origin=d.origin,z=z)).groupby('origin').z.mean().sort_index().to_numpy()[:,None]
      for block in [3,6,12]:
       mu,lo,hi,pv=bootstrap(zz,block);pairs.append(dict(country=co,target=t,regime=regime or 'fixed',model=f,reference=ref,block=block,gain=mu[0],lower=lo[0],upper=hi[0],p=pv[0]))
      if f=='borrowed' and ref in ['blend','control','matched_control','selected_baseline']:
       for yr in sorted(d.year.unique()):
        s=d.year==yr;years.append(dict(country=co,target=t,regime=regime or 'fixed',reference=ref,year=int(yr),gain=z[s].mean(),omit_gain=z[~s].mean(),n=int(s.sum()),positive=int(y[s].sum())))
       for c in sorted(d.cell.unique()):
        s=d.cell==c;cells.append(dict(country=co,target=t,regime=regime or 'fixed',reference=ref,cell=c,gain=z[s].mean(),omit_gain=z[~s].mean(),n=int(s.sum()),positive=int(y[s].sum())))
       for outcome in [0,1]:
        s=y==outcome;breakdowns.append(dict(country=co,target=t,regime=regime or 'fixed',reference=ref,outcome=outcome,n=int(s.sum()),gain=z[s].mean(),contribution=z[s].sum()/len(z)))
 a=pd.DataFrame(pairs)
 for _,ix in a.groupby(['regime','reference','block']).groups.items():a.loc[ix,'p_holm_all']=multipletests(a.loc[ix,'p'],method='holm')[1]
 for regime in ['fixed','annual_']:
  for ref in ['etas_cal','blend','control','matched_control','selected_baseline']:
   for b in [3,6,12]:
    ix=a.index[(a.country=='alaska')&(a.regime==regime)&(a.model=='borrowed')&(a.reference==ref)&(a.block==b)]
    if len(ix):a.loc[ix,'p_holm_alaska3']=multipletests(a.loc[ix,'p'],method='holm')[1]
 for name,rows in [('metrics',metrics),('paired_tests',a),('outcome_contributions',breakdowns),('reliability',reliability),('year_stability',years),('cell_stability',cells)]:pd.DataFrame(rows).to_csv(ROOT/f'results/{name}.csv',index=False)
 print(pd.DataFrame(metrics).query("target=='ge35' and model in ['etas_cal','blend','control','borrowed','annual_control','annual_borrowed','selected_baseline']").to_string(index=False),flush=True)
 print(a[(a.target=='ge35')&(a.model=='borrowed')&(a.block==6)].to_string(index=False),flush=True)

if __name__=='__main__':
 import sys
 countries=sys.argv[1:] or ['italy','greece','alaska'];p=ROOT/'results/frozen_selection.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==p.with_suffix('.sha256').read_text().strip()
 # Freeze executable architecture before opening a new geographic evaluation.
 sources={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'experiments').glob('*.py')};(ROOT/('results/evaluation_lock_'+'_'.join(countries)+'.json')).write_text(json.dumps(dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(),sources=sources,selection_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
 with concurrent.futures.ProcessPoolExecutor(4) as pool:list(pool.map(job_eval,[(co,t) for co in countries for t in TARGETS]))
 summarize(countries)
