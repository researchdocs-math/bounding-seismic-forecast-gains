"""Freeze cross-region feature choice before the Chile audit, then evaluate all fixed families."""
from persistence_models import *
from evaluate_borrowed import bootstrap
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

def run(job):
 co,t=job;d=pdata(co);spec=json.loads((ROOT/'results/persistence_selection.json').read_text());ss={r['family']:r for r in spec if r['country']==co and r['target']==t};tr=d[d.end<pd.Timestamp('2020-01-01')];te=d[d.date>=pd.Timestamp('2020-01-01')];out=te[['country','cell','origin','year','center_lat','center_lon','y_'+t,'etas_'+t]].copy().rename(columns={'y_'+t:'y','etas_'+t:'raw_etas'});out['target']=t
 chosen=json.loads((ROOT/'results/persistence_family_lock.json').read_text())['selected_family']
 for f,r in ss.items():
  m=Learner(r['columns'],r['config'],t).fit(tr,tr['y_'+t]);out[f]=m.predict(te)
  for yr in sorted(te.year.unique()):
   h=d[d.end<pd.Timestamp(f'{yr}-01-01')];z=d[d.year==yr];m=Learner(r['columns'],r['config'],t).fit(h,h['y_'+t]);out.loc[out.year==yr,'annual_'+f]=m.predict(z)
 for f in ['ratebank',chosen]:
  r=ss[f]
  for base in (['blend'] if f=='ratebank' else ['ratebank']):
   cc=pcols(base,t,d);m=Learner(cc,r['config'],t).fit(tr,tr['y_'+t]);out['matched_'+f]=m.predict(te)
   for yr in sorted(te.year.unique()):
    h=d[d.end<pd.Timestamp(f'{yr}-01-01')];z=d[d.year==yr];m=Learner(cc,r['config'],t).fit(h,h['y_'+t]);out.loc[out.year==yr,'annual_matched_'+f]=m.predict(z)
 out['selected']=out[chosen];out['annual_selected']=out['annual_'+chosen];out['selected_name']=chosen
 base=min(['etas_cal','blend','original','ratebank'],key=lambda x:ss[x]['validation_loss']);out['baseline']=out[base];out['annual_baseline']=out['annual_'+base];out['baseline_name']=base
 out.to_csv(ROOT/f'results/persistence_predictions_{co}_{t}.csv',index=False);print('PREDICTED',co,t,chosen,base,flush=True)

def summarize(countries):
 metrics=[];pairs=[];year=[];cell=[];reliability=[];contributions=[]
 chosen=json.loads((ROOT/'results/persistence_family_lock.json').read_text())['selected_family']
 for co in countries:
  for t in TARGETS:
   d=pd.read_csv(ROOT/f'results/persistence_predictions_{co}_{t}.csv');y=d.y.to_numpy()
   for regime in ['','annual_']:
    for f in PFAMS+['selected','baseline','matched_ratebank','matched_'+chosen]:
     p=d[regime+f].to_numpy();metrics.append(dict(country=co,target=t,regime=regime or 'fixed',model=f,n=len(y),positive=int(y.sum()),loss=float(loss(y,p).mean()),brier=float(np.mean((y-p)**2)),predicted=float(p.mean()),observed=float(y.mean()),auc=roc_auc_score(y,p)))
     if f in ['etas_cal','blend','ratebank','selected']:
      bins=pd.qcut(p,8,duplicates='drop');z=pd.DataFrame(dict(p=p,y=y,bin=bins));g=z.groupby('bin',observed=True).agg(n=('y','size'),positive=('y','sum'),predicted=('p','mean'),observed=('y','mean'))
      for r in g.to_dict('records'):reliability.append(dict(country=co,target=t,regime=regime or 'fixed',model=f,**r))
    for f in ['ratebank','duration','spread','marks','persistence','joint','selected']:
     refs=['etas_cal','blend','baseline']
     if f!='ratebank':refs+=['ratebank']
     if f=='ratebank' or f==chosen:refs+=['matched_'+f]
     for ref in refs:
      z=loss(y,d[regime+ref])-loss(y,d[regime+f]);a=pd.DataFrame(dict(origin=d.origin,z=z)).groupby('origin').z.mean().sort_index().to_numpy()[:,None]
      for b in [3,6,12]:
       mu,lo,hi,p=bootstrap(a,b);pairs.append(dict(country=co,target=t,regime=regime or 'fixed',model=f,reference=ref,block=b,gain=mu[0],lower=lo[0],upper=hi[0],p=p[0]))
      if f in ['ratebank',chosen]:
       for yr in sorted(d.year.unique()):
        s=d.year==yr;year.append(dict(country=co,target=t,regime=regime or 'fixed',model=f,reference=ref,year=int(yr),gain=z[s].mean(),omit_gain=z[~s].mean()))
       for cid in sorted(d.cell.unique()):
        s=d.cell==cid;cell.append(dict(country=co,target=t,regime=regime or 'fixed',model=f,reference=ref,cell=cid,gain=z[s].mean(),omit_gain=z[~s].mean()))
       for outcome in [0,1]:
        s=y==outcome;contributions.append(dict(country=co,target=t,regime=regime or 'fixed',model=f,reference=ref,outcome=outcome,n=int(s.sum()),gain=z[s].mean(),contribution=z[s].sum()/len(z)))
 a=pd.DataFrame(pairs)
 for _,ix in a.groupby(['regime','reference','block']).groups.items():a.loc[ix,'p_holm_all']=multipletests(a.loc[ix,'p'],method='holm')[1]
 for _,ix in a[a.country.eq('chile')].groupby(['regime','reference','block','model']).groups.items():a.loc[ix,'p_holm_chile3']=multipletests(a.loc[ix,'p'],method='holm')[1]
 for name,rows in [('metrics',metrics),('paired_tests',a),('year_stability',year),('cell_stability',cell),('reliability',reliability),('outcome_contributions',contributions)]:pd.DataFrame(rows).to_csv(ROOT/f'results/persistence_{name}.csv',index=False)
 print(pd.DataFrame(metrics).query("target=='ge35' and regime=='fixed'").to_string(index=False),flush=True)

if __name__=='__main__':
 import sys
 countries=sys.argv[1:] or ['italy','greece','alaska','chile'];p=ROOT/'results/persistence_selection.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==p.with_suffix('.sha256').read_text().strip()
 lock=ROOT/'results/persistence_family_lock.json'
 if not lock.exists():
  x=pd.DataFrame(json.loads(p.read_text()));v=x[(x.country!='chile')&(x.target=='ge35')].pivot(index='country',columns='family',values='validation_loss');candidates=['duration','spread','marks','persistence','joint'];gain={f:float((v.ratebank-v[f]).mean()) for f in candidates};chosen=max(gain,key=gain.get);lock.write_text(json.dumps(dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),selected_family=chosen,validation_mean_gain=gain,selection_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),sources={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in (ROOT/'experiments').glob('*.py')}),indent=2))
 with concurrent.futures.ProcessPoolExecutor(8) as pool:list(pool.map(run,[(co,t) for co in countries for t in TARGETS]))
 summarize(countries)
