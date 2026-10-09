"""Causal spatial pooling of completed forecast errors and matched model families."""
from models import *
from scipy.spatial.distance import cdist
import hashlib,json,concurrent.futures
RADII=[50,100,200];WINDOWS=[3,6,12,24]
FAMS=['etas_cal','blend','original','local','neighbor','control','borrowed','all']
CONFIGS=[{'learner':s,'C':c} for s in ['linear','spline'] for c in [.01,.1,1.,10.]]+[{'learner':'boost','depth':h,'trees':n,'l2':10.} for h in [1,2,3] for n in [150,350]]

def add_features(co):
 d=pd.read_csv(ROOT/f'data/processed/{co}_forecasts.csv.gz',dtype={'cell':str}).sort_values(['origin','cell']);dates=sorted(d.origin.unique());cells=sorted(d.cell.unique());index=pd.MultiIndex.from_product([dates,cells],names=['origin','cell']);d=d.set_index(['origin','cell']).reindex(index);assert d.country.notna().all();T,C=len(dates),len(cells)
 centers=d.reset_index().groupby('cell')[['center_lat','center_lon']].first().reindex(cells);lat=np.deg2rad(centers.center_lat.to_numpy());lon=np.deg2rad(centers.center_lon.to_numpy());a=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2;dist=6371*2*np.arcsin(np.sqrt(a.clip(0,1)))
 weights={s:np.exp(-.5*(dist/s)**2) for s in RADII}
 for w in weights.values():np.fill_diagonal(w,0)
 feature={};current=d['n30'].to_numpy().reshape(T,C)
 for s,W in weights.items():
  feature[f'neighbor_small30_{s}']=np.log1p(current@W.T)
 for t in TARGETS:
  y=d['y_'+t].to_numpy().reshape(T,C);p=d['etas_'+t].to_numpy().reshape(T,C);cnt=d['count_'+t].to_numpy().reshape(T,C);err=y-p;var=p*(1-p)
  for n in WINDOWS:
   sy=np.zeros((T,C));sp=np.zeros((T,C));sv=np.zeros((T,C));sn=np.zeros((T,C))
   for i in range(T):
    j=max(0,i-n);sy[i]=y[j:i].sum(0);sp[i]=p[j:i].sum(0);sv[i]=var[j:i].sum(0);sn[i]=cnt[j:i].sum(0)
   feature[f'own_z{n}_{t}']=np.arcsinh((sy-sp)/np.sqrt(1+sv));feature[f'own_error{n}_{t}']=(sy-sp)/(n+6)
   for s,W in weights.items():
    feature[f'peer_z{s}_{n}_{t}']=np.arcsinh((sy-sp)@W.T/np.sqrt(1+sv@(W*W).T))
    feature[f'peer_error{s}_{n}_{t}']=((sy-sp)@W.T)/(6+n*W.sum(1))
    feature[f'neighbor_count{s}_{n}_{t}']=np.log1p(sn@W.T)
    feature[f'neighbor_expected{s}_{n}_{t}']=np.log1p(sp@W.T)
  for s,W in weights.items():feature[f'neighbor_etas{s}_{t}']=np.log1p(p@W.T)
 out=d.reset_index()[['country','cell','origin']].copy()
 for key,value in feature.items():out[key]=value.reshape(-1)
 out.to_csv(ROOT/f'data/processed/{co}_borrowed.csv.gz',index=False,compression={'method':'gzip','mtime':0});print(co,out.shape,flush=True)
 # Independently recompute one middle-history entry from explicit dates as a causality audit.
 checks=[]
 for t in TARGETS:
  for i in [1,min(40,T-1),T-1]:
   n=6;s=100;c=0;hist=d.reset_index();target_date=pd.Timestamp(dates[i]);z=hist[(pd.to_datetime(hist.origin)<target_date)&(pd.to_datetime(hist.origin)>=target_date-pd.Timedelta(days=30*n))];sy=z.pivot(index='origin',columns='cell',values='y_'+t).reindex(columns=cells).to_numpy();sp=z.pivot(index='origin',columns='cell',values='etas_'+t).reindex(columns=cells).to_numpy();W=weights[s][c];v=np.arcsinh(np.sum((sy-sp)*W)/np.sqrt(1+np.sum(sp*(1-sp)*W**2)));actual=out.loc[(out.origin==dates[i])&(out.cell==cells[c]),f'peer_z{s}_{n}_{t}'].iloc[0];assert np.isclose(v,actual,atol=1e-12);checks.append(dict(target=t,origin=dates[i],absolute_error=float(abs(v-actual))))
 (ROOT/f'results/causality_{co}.json').write_text(json.dumps(checks,indent=2))

def dataset(co):
 d=load(co);x=pd.read_csv(ROOT/f'data/processed/{co}_borrowed.csv.gz',dtype={'cell':str});return d.merge(x,on=['country','cell','origin'],validate='one_to_one')

def cols(f,t,d):
 base=columns('etas_cal' if f=='etas_cal' else 'blend',t)
 own=[c for c in d if c.startswith(('own_z','own_error')) and c.endswith('_'+t)]
 neighbor=[c for c in d if c.startswith('neighbor_') and (c.endswith('_'+t) or c.startswith('neighbor_small'))]
 # Cross-target error borrowing captures magnitude-conditioned residual state.
 peer=[c for c in d if c.startswith(('peer_z','peer_error'))]
 if f in ['original','all']:base+=GROUPS['original']
 if f in ['local','control','borrowed','all']:base+=own
 if f in ['neighbor','control','borrowed','all']:base+=neighbor
 if f in ['borrowed','all']:base+=peer
 return list(dict.fromkeys(base))

def selection_job(job):
 co,t,f=job;d=dataset(co);tr=d[d.end<pd.Timestamp('2015-01-01')];va=d[(d.date>=pd.Timestamp('2015-01-01'))&(d.end<pd.Timestamp('2020-01-01'))];c=cols(f,t,d);records=[]
 for cfg in CONFIGS:
  model=Learner(c,cfg,t).fit(tr,tr['y_'+t]);p=model.predict(va);records.append(dict(country=co,target=t,family=f,config=cfg,columns=c,validation_loss=float(loss(va['y_'+t],p).mean())))
 best=min(records,key=lambda r:r['validation_loss']);print('SELECT',co,t,f,best['config'],best['validation_loss'],flush=True);return dict(best=best,search=records)

def select(countries):
 jobs=[(co,t,f) for co in countries for t in TARGETS for f in FAMS]
 with concurrent.futures.ProcessPoolExecutor(8) as pool:ans=list(pool.map(selection_job,jobs))
 path=ROOT/'results/frozen_selection.json';old=json.loads(path.read_text()) if path.exists() else []
 old=[a for a in old if a['country'] not in countries]+[a['best'] for a in ans];path.write_text(json.dumps(old,indent=2));path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
 (ROOT/('results/validation_search_'+'_'.join(countries)+'.json')).write_text(json.dumps([r for a in ans for r in a['search']],indent=2))

if __name__=='__main__':
 import sys
 action=sys.argv[1];countries=sys.argv[2:] or ['italy','greece']
 if action=='features':
  with concurrent.futures.ProcessPoolExecutor(min(3,len(countries))) as pool:list(pool.map(add_features,countries))
 elif action=='select':select(countries)
