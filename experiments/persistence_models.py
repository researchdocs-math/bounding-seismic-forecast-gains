"""Second-stage models: occupancy conditional on a broad multiscale rate reference."""
from borrowed_memory import *
PFAMS=['etas_cal','blend','original','ratebank','duration','spread','marks','persistence','joint']
def pdata(co):
 d=load(co);x=pd.read_csv(ROOT/f'data/processed/{co}_persistence.csv.gz',dtype={'cell':str});return d.merge(x,on=['country','cell','origin'],validate='one_to_one')
def pcols(f,t,d):
 base=columns('etas_cal' if f=='etas_cal' else 'blend',t)
 if f=='original':return list(dict.fromkeys(base+GROUPS['original']))
 if f in ['etas_cal','blend']:return base
 base += [c for c in d if c.startswith('rate_')]
 if f in ['duration','persistence','joint']:base += [c for c in d if c.startswith('persist_')]
 if f in ['spread','persistence','joint']:base += [c for c in d if c.startswith('spread_')]
 if f in ['marks','joint']:base += [c for c in d if c.startswith('mark_')]
 if f=='joint':base+=GROUPS['original']
 return list(dict.fromkeys(base))
def select_job(job):
 co,t,f=job;d=pdata(co);tr=d[d.end<pd.Timestamp('2015-01-01')];va=d[(d.date>=pd.Timestamp('2015-01-01'))&(d.end<pd.Timestamp('2020-01-01'))];cc=pcols(f,t,d);rows=[]
 for cfg in CONFIGS:
  m=Learner(cc,cfg,t).fit(tr,tr['y_'+t]);p=m.predict(va);rows.append(dict(country=co,target=t,family=f,columns=cc,config=cfg,validation_loss=float(loss(va['y_'+t],p).mean())))
 best=min(rows,key=lambda x:x['validation_loss']);print('SELECT',co,t,f,best['config'],best['validation_loss'],flush=True);return best,rows
if __name__=='__main__':
 import sys
 countries=sys.argv[1:] or ['italy','greece','alaska']
 with concurrent.futures.ProcessPoolExecutor(8) as pool:ans=list(pool.map(select_job,[(c,t,f) for c in countries for t in TARGETS for f in PFAMS]))
 p=ROOT/'results/persistence_selection.json';old=json.loads(p.read_text()) if p.exists() else [];old=[x for x in old if x['country'] not in countries]+[x[0] for x in ans];p.write_text(json.dumps(old,indent=2));p.with_suffix('.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n');(ROOT/('results/persistence_search_'+'_'.join(countries)+'.json')).write_text(json.dumps([x for a in ans for x in a[1]],indent=2))
