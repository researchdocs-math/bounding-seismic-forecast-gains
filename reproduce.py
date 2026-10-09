"""Offline reconstruction and numerical comparison for Bounding Seismic Forecast Gains."""
from pathlib import Path
import os,sys,subprocess,argparse,shutil,json
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parent

def run(script,*args):
 log=ROOT/'run_logs'/((Path(script).stem)+'_'+'_'.join(args)+'.log');print('Running',script,*args,flush=True)
 with log.open('w') as f:r=subprocess.run([sys.executable,str(ROOT/'experiments'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{script} failed; see {log}')

def compare():
 import pandas as pd,numpy as np
 rows=[]
 for ref in sorted((ROOT/'expected/results').glob('*.csv')):
  p=ROOT/'results'/ref.name
  if not p.exists():raise AssertionError(f'Missing reproduced table: {ref.name}')
  a=pd.read_csv(ref);b=pd.read_csv(p)
  if set(a.columns)!=set(b.columns) or a.shape!=b.shape:raise AssertionError(f'Shape/column difference: {ref.name}')
  b=b[a.columns];maxerr=0.
  for c in a:
   if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
    av=a[c].to_numpy(float);bv=b[c].to_numpy(float)
    if not np.allclose(av,bv,rtol=1e-7,atol=2e-8,equal_nan=True):raise AssertionError(f'Numeric mismatch: {ref.name}:{c}')
    if np.any(np.isfinite(av)&np.isfinite(bv)):maxerr=max(maxerr,float(np.nanmax(np.abs(av-bv))))
   elif not a[c].fillna('').astype(str).equals(b[c].fillna('').astype(str)):raise AssertionError(f'Category mismatch: {ref.name}:{c}')
  rows.append(dict(file=ref.name,rows=len(a),max_abs_difference=maxerr))
 required=['online_paired_tests.csv','paired_tests.csv','persistence_paired_tests.csv','skill_resolution.csv','numerical_checks.csv','synthetic_summary.csv']
 assert all(x in [r['file'] for r in rows] for x in required)
 for file in ['frozen_selection.json','persistence_selection.json','online_selection.json']:
  a=json.loads((ROOT/'expected/results'/file).read_text());b=json.loads((ROOT/'results'/file).read_text());assert len(a)==len(b)
  def key(x):return (x['country'],x['target'],x.get('family',''))
  a=sorted(a,key=key);b=sorted(b,key=key)
  for x,y in zip(a,b):
   for k in x:
    if isinstance(x[k],float):assert abs(x[k]-y[k])<2e-8,(file,k)
    else:assert x[k]==y[k],(file,k)
 (ROOT/'results/reproduction_verification.json').write_text(json.dumps(dict(verified_tables=rows,selected_specifications_match=True),indent=2))
 print('Verified',len(rows),'numerical tables and all selected specifications.',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--verify-only',action='store_true');args=ap.parse_args();(ROOT/'run_logs').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
 if not args.verify_only:
  if args.full:
   for p in (ROOT/'results').glob('*'):
    if p.is_file():p.unlink()
   run('prepare_inherited.py','--offline')
   for co in ['alaska','chile','japan']:run('download_'+co+'.py','--offline')
   run('build_forecasts.py')
  run('borrowed_memory.py','features','italy','greece','alaska');run('borrowed_memory.py','select','italy','greece','alaska');run('evaluate_borrowed.py','italy','greece','alaska')
  run('persistence_features.py','italy','greece','alaska','chile','japan');run('persistence_models.py','italy','greece','alaska','chile','japan')
  # Preserve the stage-two family chosen before Japan entered the experiment.
  shutil.copy2(ROOT/'protocol/persistence_family_lock.json',ROOT/'results/persistence_family_lock.json')
  run('evaluate_persistence.py','italy','greece','alaska','chile')
  run('online_calibration.py','italy','greece','alaska','chile','japan');shutil.copy2(ROOT/'protocol/online_audit_lock.json',ROOT/'results/online_audit_lock.json');run('evaluate_online.py','italy','greece','alaska','chile','japan')
  run('final_checks.py');run('synthetic_control.py')
 compare()
