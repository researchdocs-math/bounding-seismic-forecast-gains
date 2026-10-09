"""Locked final audit of online calibration correction."""
from online_calibration import *
from evaluate_borrowed import bootstrap
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

def evaluate(countries):
 spec=json.loads((ROOT/'results/online_selection.json').read_text());metrics=[];paired=[];cal=[];years=[];cellstats=[];contributions=[]
 for co in countries:
  for t in TARGETS:
   d=pd.read_csv(ROOT/f'results/issued_{co}_{t}.csv',dtype={'cell':str});r=next(x for x in spec if x['country']==co and x['target']==t);p,theta=correct(d,r['base'],r['radius'],r['memory'],r['tau']);d['corrected']=p;d['theta']=theta;d['reference']=d[r['base']];d['baseline_name']=r['base'];d=d[pd.to_datetime(d.origin)>=pd.Timestamp('2020-01-01')].copy();d.to_csv(ROOT/f'results/online_predictions_{co}_{t}.csv',index=False);y=d.y.to_numpy()
   for name in ['reference','corrected']+[f+'_'+h for f in BASES for h in ['expanding','fiveyear']]:
    pp=d[name].to_numpy();metrics.append(dict(country=co,target=t,model=name,n=len(y),positive=int(y.sum()),loss=float(loss(y,pp).mean()),brier=float(np.mean((y-pp)**2)),observed=float(y.mean()),predicted=float(pp.mean()),auc=roc_auc_score(y,pp)))
    if name in ['reference','corrected']:
     bb=pd.qcut(pp,8,duplicates='drop');b=pd.DataFrame(dict(p=pp,y=y,bin=bb));a=b.groupby('bin',observed=True).agg(n=('y','size'),positive=('y','sum'),predicted=('p','mean'),observed=('y','mean'))
     for rr in a.to_dict('records'):cal.append(dict(country=co,target=t,model=name,**rr))
   for ref in ['reference','etas_cal_expanding','blend_expanding','ratebank_expanding','etas_cal_fiveyear','blend_fiveyear','ratebank_fiveyear']:
    z=loss(y,d[ref])-loss(y,d.corrected);zz=pd.DataFrame(dict(origin=d.origin,z=z)).groupby('origin').z.mean().sort_index().to_numpy()[:,None]
    for block in [3,6,12]:
     mu,lo,hi,pv=bootstrap(zz,block);paired.append(dict(country=co,target=t,reference=ref,block=block,gain=mu[0],lower=lo[0],upper=hi[0],p=pv[0]))
    if ref=='reference':
     for yr in sorted(d.year.unique()):
      s=d.year==yr;years.append(dict(country=co,target=t,year=int(yr),gain=float(z[s].mean()),omit_gain=float(z[~s].mean()),positive=int(y[s].sum())))
     for cid in sorted(d.cell.unique()):
      s=d.cell==cid;cellstats.append(dict(country=co,target=t,cell=cid,gain=float(z[s].mean()),omit_gain=float(z[~s].mean()),positive=int(y[s].sum())))
     for yy in [0,1]:
      s=y==yy;contributions.append(dict(country=co,target=t,outcome=yy,n=int(s.sum()),gain=z[s].mean(),contribution=z[s].sum()/len(z)))
 a=pd.DataFrame(paired)
 for _,ix in a.groupby(['country','reference','block']).groups.items():a.loc[ix,'p_holm3']=multipletests(a.loc[ix,'p'],method='holm')[1]
 for _,ix in a.groupby(['reference','block']).groups.items():a.loc[ix,'p_holm15']=multipletests(a.loc[ix,'p'],method='holm')[1]
 for name,rows in [('metrics',metrics),('paired_tests',a),('reliability',cal),('year_stability',years),('cell_stability',cellstats),('outcome_contributions',contributions)]:pd.DataFrame(rows).to_csv(ROOT/f'results/online_{name}.csv',index=False)
 print(pd.DataFrame(metrics).query("model in ['reference','corrected']").to_string(index=False),flush=True);print(a[(a.reference=='reference')&(a.block==6)].to_string(index=False),flush=True)
if __name__=='__main__':
 import sys
 countries=sys.argv[1:] or ['italy','greece','alaska','chile','japan'];p=ROOT/'results/online_selection.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==p.with_suffix('.sha256').read_text().strip();lock=ROOT/'results/online_audit_lock.json'
 if not lock.exists():lock.write_text(json.dumps(dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(),selection_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'experiments').glob('*.py')}),indent=2))
 evaluate(countries)
