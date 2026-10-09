"""Rate, temporal occupancy and spatial spread from past catalog events only."""
from models import *
from features import day
import concurrent.futures,json
WIN=[7,30,90,365]
def country(co):
 cat=pd.read_csv(ROOT/f'data/processed/{co}_catalog.csv.gz',dtype={'cell':str});fore=pd.read_csv(ROOT/f'data/processed/{co}_forecasts.csv.gz',dtype={'cell':str});cells={c['cell']:c for c in json.loads((ROOT/'data/processed/cells.json').read_text()) if c['country']==co};rows=[]
 for cid,g in fore.groupby('cell'):
  c=cells[cid];h=cat[cat.cell==cid].sort_values('t_days');tt=h.t_days.to_numpy();mm=h.magnitude.to_numpy();la=h.latitude.to_numpy();lo=h.longitude.to_numpy();dep=h.depth_km.to_numpy();gs=c.get('grid_degrees',1)
  for row in g.itertuples():
   o=row.origin_day;end=np.searchsorted(tt,o);a={'country':co,'cell':cid,'origin':row.origin}
   for w in [1,3,7,14,30,60,90,180,365,730,1826]:
    j=np.searchsorted(tt,o-w);m=mm[j:end]
    for k,cut in enumerate([3.,3.5,4.5]):a[f'rate_{w}_{k}']=np.log1p(np.sum(m>=cut)*30/w)
   for w in WIN:
    j=np.searchsorted(tt,o-w);t=tt[j:end];m=mm[j:end];n=len(t);age=o-t;bins=np.clip(np.floor(age).astype(int),0,w-1);daily=np.bincount(bins,minlength=w);active=daily>0
    a[f'persist_days_{w}']=np.log1p(active.sum());a[f'persist_weeks_{w}']=np.log1p(len(np.unique(bins//7)));a[f'persist_daily_max_{w}']=daily.max()/max(n,1);a[f'persist_age_mean_{w}']=age.mean()/w if n else 1.;a[f'persist_old_half_{w}']=float(np.sum(age>w/2)/max(n,1));a[f'persist_fano_{w}']=np.log1p(np.var(daily)/(np.mean(daily)+.1));a[f'persist_daily_entropy_{w}']=float(-np.sum((daily[daily>0]/max(n,1))*np.log(daily[daily>0]/max(n,1)))/np.log(w))
    xs=np.clip(np.floor((lo[j:end]-c['lon_min'])/gs*4).astype(int),0,3);ys=np.clip(np.floor((la[j:end]-c['lat_min'])/gs*4).astype(int),0,3);counts=np.bincount(ys*4+xs,minlength=16);p=counts[counts>0]/max(n,1)
    a[f'spread_active_{w}']=np.log1p(len(p));a[f'spread_entropy_{w}']=float(-np.sum(p*np.log(p))/np.log(16));a[f'spread_dominance_{w}']=float(counts.max()/max(n,1));a[f'spread_depth_sd_{w}']=float(np.std(dep[j:end])) if n else 0.
    if n>1:
     x=(lo[j:end]-c['center_lon'])*111*np.cos(np.deg2rad(c['center_lat']));y=(la[j:end]-c['center_lat'])*111;e=np.maximum(np.linalg.eigvalsh(np.cov([x,y])),0);a[f'spread_area_{w}']=np.log1p(np.sqrt(e.prod()));a[f'spread_radius_{w}']=np.log1p(np.sqrt(e.sum()));a[f'spread_anisotropy_{w}']=np.log((e[1]+1)/(e[0]+1))
    else:
     for key in ['area','radius','anisotropy']:a[f'spread_{key}_{w}']=0.
    # Magnitude-tail borrowing uses bounded empirical tail fractions with six-event prior.
    for k,cut in enumerate([3.5,4.5]):a[f'mark_tail_{w}_{k}']=float((np.sum(m>=cut)+6*10**(-(cut-3)))/(n+6))
    a[f'mark_mean_{w}']=float(np.mean(m)-3) if n else .4343;a[f'mark_sd_{w}']=float(np.std(m)) if n else 0.
   rows.append(a)
 out=pd.DataFrame(rows);out.to_csv(ROOT/f'data/processed/{co}_persistence.csv.gz',index=False,compression={'method':'gzip','mtime':0});print(co,'persistence',out.shape,flush=True)
if __name__=='__main__':
 import sys
 with concurrent.futures.ProcessPoolExecutor(4) as pool:list(pool.map(country,sys.argv[1:] or ['italy','greece','alaska']))
