"""Causal regularized intercept updates from completed issued forecasts."""
from persistence_models import *
from scipy.special import expit,logit
BASES=['etas_cal','blend','ratebank']
SPATIAL=['own',100,200,'region'];MEMORY=[3,6,12,24];TAU=[1,4,16,64]

def base_job(job):
 co,t=job;d=pdata(co);spec=json.loads((ROOT/'results/persistence_selection.json').read_text());ss={r['family']:r for r in spec if r['country']==co and r['target']==t};te=d[d.date>=pd.Timestamp('2015-01-01')];out=te[['country','cell','origin','year','center_lat','center_lon','y_'+t]].copy().rename(columns={'y_'+t:'y'});out['target']=t
 for f in BASES:
  # Hyperparameters fixed in development; every fitted coefficient uses only completed earlier forecasts.
  r=ss[f]
  for hist in ['expanding','fiveyear']:
   for yr in sorted(te.year.unique()):
    start=pd.Timestamp('2010-01-01') if hist=='expanding' else pd.Timestamp(f'{max(2010,yr-5)}-01-01');h=d[(d.end<pd.Timestamp(f'{yr}-01-01'))&(d.date>=start)];z=d[d.year==yr];m=Learner(r['columns'],r['config'],t).fit(h,h['y_'+t]);out.loc[out.year==yr,f+'_'+hist]=m.predict(z)
 out.to_csv(ROOT/f'results/issued_{co}_{t}.csv',index=False);print('ISSUED',co,t,flush=True)

def correct(d,base,radius,memory,tau):
 dates=sorted(d.origin.unique());cells=sorted(d.cell.unique());x=d.set_index(['origin','cell']).reindex(pd.MultiIndex.from_product([dates,cells],names=['origin','cell']));T,C=len(dates),len(cells);y=x.y.to_numpy().reshape(T,C);p=x[base].to_numpy().reshape(T,C);ell=logit(np.clip(p,1e-6,1-1e-6));c=x.reset_index().groupby('cell')[['center_lat','center_lon']].first().reindex(cells);lat=np.deg2rad(c.center_lat.to_numpy());lon=np.deg2rad(c.center_lon.to_numpy());aa=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2;dist=12742*np.arcsin(np.sqrt(aa.clip(0,1)))
 W=np.eye(C) if radius=='own' else np.ones((C,C)) if radius=='region' else np.exp(-.5*(dist/float(radius))**2)
 out=p.copy();theta=np.zeros_like(p)
 for i in range(1,T):
  j=max(0,i-memory);E=ell[j:i];Y=y[j:i];a=np.zeros(C)
  for it in range(15):
   prob=expit(E[:,:,None]+a[None,None,:]);w=W.T[None,:,:];g=np.sum(w*(Y[:,:,None]-prob),axis=(0,1))-tau*a;h=np.sum(w*prob*(1-prob),axis=(0,1))+tau;step=g/h;a=np.clip(a+step,-4,4)
   if np.max(np.abs(step))<1e-9:break
  theta[i]=a;out[i]=expit(ell[i]+a)
 arr=pd.DataFrame({'p':out.ravel(),'theta':theta.ravel()},index=x.index);z=d[['origin','cell']].merge(arr.reset_index(),on=['origin','cell'],validate='one_to_one');return z.p.to_numpy(),z.theta.to_numpy()

def select_job(job):
 co,t=job;d=pd.read_csv(ROOT/f'results/issued_{co}_{t}.csv',dtype={'cell':str});val=(pd.to_datetime(d.origin)+pd.Timedelta(days=30)<pd.Timestamp('2020-01-01'));y=d.y.to_numpy();basenames=[f+'_'+h for f in BASES for h in ['expanding','fiveyear']];base=min(basenames,key=lambda b:loss(y[val],d.loc[val,b]).mean());records=[]
 dv=d[val].copy()
 for radius in SPATIAL:
  for w in MEMORY:
   for tau in TAU:
    p,a=correct(dv,base,radius,w,tau);records.append(dict(country=co,target=t,base=base,radius=radius,memory=w,tau=tau,validation_loss=float(loss(dv.y,p).mean()),reference_loss=float(loss(dv.y,dv[base]).mean())))
 best=min(records,key=lambda r:r['validation_loss']);print('ONLINE SELECT',best,flush=True);return best,records

def select(countries):
 with concurrent.futures.ProcessPoolExecutor(8) as pool:
  list(pool.map(base_job,[(c,t) for c in countries for t in TARGETS]));results=list(pool.map(select_job,[(c,t) for c in countries for t in TARGETS]))
 p=ROOT/'results/online_selection.json';old=json.loads(p.read_text()) if p.exists() else [];old=[x for x in old if x['country'] not in countries]+[x[0] for x in results];p.write_text(json.dumps(old,indent=2));p.with_suffix('.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n');(ROOT/('results/online_search_'+'_'.join(countries)+'.json')).write_text(json.dumps([r for a in results for r in a[1]],indent=2))
if __name__=='__main__':
 import sys
 select(sys.argv[1:] or ['italy','greece','alaska','chile'])
