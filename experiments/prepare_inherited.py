"""Acquire immutable official responses; build separate national catalog cohorts."""
from pathlib import Path
import concurrent.futures,json,hashlib,time,calendar,io
import requests
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/inherited'; RAW.mkdir(parents=True,exist_ok=True)

def fetch(job):
 name,url,params=job; dest=RAW/(name+'.txt')
 if dest.exists():return {'name':name,'url':url,'query':params,'cached':True,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
 for attempt in range(4):
  try:
   r=requests.get(url,params=params,timeout=150);r.raise_for_status()
   if r.status_code==204:raise ValueError('Empty response')
   dest.write_bytes(r.content)
   return {'name':name,'url':r.url,'status':r.status_code,'bytes':len(r.content),'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat(),'sha256':hashlib.sha256(r.content).hexdigest()}
  except Exception:
   if attempt==3:raise
   time.sleep(2+attempt*3)

def download():
 jobs=[('noa_full','https://www.gein.noa.gr/services/full_catalogue.php',{})]
 for y in range(2000,2026):
  jobs.append((f'ingv_{y}','https://webservices.ingv.it/fdsnws/event/1/query',{'starttime':f'{y}-01-01T00:00:00','endtime':f'{y+1}-01-01T00:00:00','minmagnitude':3,'minlatitude':36,'maxlatitude':47,'minlongitude':6,'maxlongitude':19,'format':'text','orderby':'time-asc','limit':20000}))
 meta=[]
 with concurrent.futures.ThreadPoolExecutor(4) as pool:
  for x in pool.map(fetch,jobs):meta.append(x);print(x['name'],x.get('bytes','cached'),flush=True)
 (ROOT/'data/download_manifest.json').write_text(json.dumps(meta,indent=2))

def prepare():
 frames=[]
 for p in sorted(RAW.glob('ingv_*.txt')):
  x=pd.read_csv(p,sep='|');x.columns=[c.lower().lstrip('#') for c in x.columns]
  x=x.rename(columns={'eventid':'event_id','depth/km':'depth_km','magtype':'magnitude_type'})
  x=x[x.eventtype.eq('earthquake')].copy();frames.append(x[['event_id','time','latitude','longitude','depth_km','magnitude','magnitude_type']])
 italy=pd.concat(frames,ignore_index=True).drop_duplicates('event_id');italy['country']='italy'
 rows=[];bad=[]
 months={m.upper():i for i,m in enumerate(calendar.month_abbr) if i}
 for line in (RAW/'noa_full.txt').read_text(errors='replace').splitlines():
  s=line.split()
  if len(s)<10 or not s[0].isdigit():continue
  try:
   y,mo,dd,hh,mi,sec,lat,lon,dep,mag=s[:10]
   if not 2000<=int(y)<=2025:continue
   d=pd.Timestamp(year=int(y),month=months[mo.upper()],day=int(dd),tz='UTC')+pd.Timedelta(hours=int(hh),minutes=int(mi),seconds=float(sec))
   rows.append([hashlib.sha256(line.strip().encode()).hexdigest()[:20],d.isoformat(),float(lat),float(lon),float(dep),float(mag),'ML','greece'])
  except Exception as e:bad.append([line,str(e)])
 if bad:raise ValueError(str(bad[:4]))
 greece=pd.DataFrame(rows,columns=['event_id','time','latitude','longitude','depth_km','magnitude','magnitude_type','country']).drop_duplicates('event_id')
 records=[];allcells=[]
 for co,x in [('italy',italy),('greece',greece)]:
  bb={'italy':(36,47,6,19),'greece':(33,42,19,29)}[co]
  x['time']=pd.to_datetime(x.time,utc=True,format='mixed');x=x[(x.time>=pd.Timestamp('2000-01-01',tz='UTC'))&(x.time<pd.Timestamp('2026-01-01',tz='UTC'))&x.latitude.between(bb[0],bb[1],inclusive='left')&x.longitude.between(bb[2],bb[3],inclusive='left')&x.depth_km.between(0,60)&x.magnitude.between(3,8.5,inclusive='left')].copy()
  x['cell']=(np.floor(x.latitude).astype(int)*100+np.floor(x.longitude).astype(int)).astype(str)
  x['t_days']=(x.time-pd.Timestamp('1990-01-01',tz='UTC')).dt.total_seconds()/86400
  h=x[x.time<pd.Timestamp('2010-01-01',tz='UTC')]
  count=h.groupby('cell').magnitude.agg(n='size',n35=lambda v:(v>=3.5).sum())
  ids=count[(count.n>=30)&(count.n35>=5)].index
  cells=[]
  for cid in ids:
   la=int(cid)//100;lo=int(cid)%100
   cells.append({'country':co,'cell':cid,'lat_min':la,'lon_min':lo,'center_lat':la+.5,'center_lon':lo+.5,'history_count':int(count.loc[cid,'n']),'history_n35':int(count.loc[cid,'n35'])})
  x=x.sort_values(['t_days','event_id']);x['eligible']=x.cell.isin(ids)
  x.to_csv(ROOT/f'data/processed/{co}_catalog.csv.gz',index=False,compression={'method':'gzip','mtime':0})
  allcells.extend(cells)
  records.append({'country':co,'events':len(x),'eligible_events':int(x.eligible.sum()),'eligible_cells':len(cells),'first':str(x.time.min()),'last':str(x.time.max()),'magnitude_types':x.magnitude_type.value_counts().to_dict(),'year_counts':x.groupby(x.time.dt.year).size().to_dict()})
 (ROOT/'data/processed/cells.json').write_text(json.dumps(allcells,indent=2));(ROOT/'results/catalog_audit.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':
 import sys
 if '--offline' not in sys.argv:download()
 prepare()
