"""Download and preserve annual USGS responses, then apply frozen cohort rules."""
from pathlib import Path
import requests,concurrent.futures,json,hashlib,time
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def fetch(year):
 p=ROOT/f'data/raw/alaska_{year}.csv';params=dict(format='csv',starttime=f'{year}-01-01',endtime=f'{year+1}-01-01',minmagnitude=3,minlatitude=52,maxlatitude=72,minlongitude=-170,maxlongitude=-130,eventtype='earthquake',orderby='time-asc',limit=20000)
 url='https://earthquake.usgs.gov/fdsnws/event/1/query'
 if not p.exists():
  for attempt in range(5):
   try:
    r=requests.get(url,params=params,timeout=150);r.raise_for_status();assert r.status_code==200;p.write_bytes(r.content);break
   except Exception:
    if attempt==4:raise
    time.sleep(3*(attempt+1))
 return dict(year=year,url=requests.Request('GET',url,params=params).prepare().url,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),retrieved_utc=pd.Timestamp.now(tz='UTC').isoformat())
def prepare():
 xs=[]
 for p in sorted((ROOT/'data/raw').glob('alaska_*.csv')):
  x=pd.read_csv(p);assert len(x)<20000;xs.append(x)
 x=pd.concat(xs,ignore_index=True).drop_duplicates('id').rename(columns={'id':'event_id','depth':'depth_km','mag':'magnitude','magType':'magnitude_type'})
 x['time']=pd.to_datetime(x.time,utc=True,format='mixed');x=x[(x.time>=pd.Timestamp('2000-01-01',tz='UTC'))&(x.time<pd.Timestamp('2026-01-01',tz='UTC'))&x.latitude.between(52,72,inclusive='left')&x.longitude.between(-170,-130,inclusive='left')&x.depth_km.between(0,60)&x.magnitude.between(3,8.5,inclusive='left')].copy()
 x['country']='alaska';ii=np.floor((x.latitude-52)/2).astype(int);jj=np.floor((x.longitude+170)/2).astype(int);x['cell']=(ii*100+jj).astype(str)
 x['t_days']=(x.time-pd.Timestamp('1990-01-01',tz='UTC')).dt.total_seconds()/86400
 h=x[x.time<pd.Timestamp('2010-01-01',tz='UTC')];counts=h.groupby('cell').magnitude.agg(n='size',n35=lambda z:(z>=3.5).sum());ids=counts[(counts.n>=30)&(counts.n35>=5)].index
 cells=[]
 for cid in ids:
  la=52+int(cid)//100*2;lo=-170+int(cid)%100*2
  cells.append(dict(country='alaska',cell=cid,lat_min=la,lon_min=lo,center_lat=la+1,center_lon=lo+1,history_count=int(counts.loc[cid,'n']),history_n35=int(counts.loc[cid,'n35']),grid_degrees=2))
 x['eligible']=x.cell.isin(ids);x=x.sort_values(['t_days','event_id']);cols=['country','event_id','time','t_days','cell','latitude','longitude','depth_km','magnitude','magnitude_type','eligible'];x[cols].to_csv(ROOT/'data/processed/alaska_catalog.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 allcells=json.loads((ROOT/'data/processed/cells.json').read_text());allcells=[c for c in allcells if c['country']!='alaska']+cells;(ROOT/'data/processed/cells.json').write_text(json.dumps(allcells,indent=2))
 audit=dict(events=len(x),cells=len(cells),eligible_events=int(x.eligible.sum()),magnitude_types=x.magnitude_type.value_counts().to_dict(),year_counts=x.groupby(x.time.dt.year).size().to_dict());(ROOT/'results/alaska_catalog_audit.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)
if __name__=='__main__':
 import sys
 if '--offline' not in sys.argv:
  with concurrent.futures.ThreadPoolExecutor(3) as pool:
   meta=list(pool.map(fetch,range(2000,2026)))
  (ROOT/'data/alaska_download_manifest.json').write_text(json.dumps(meta,indent=2))
 prepare()
