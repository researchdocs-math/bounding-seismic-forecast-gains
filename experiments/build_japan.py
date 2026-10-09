"""Build Japan baselines with the same annual likelihood and branching calculation."""
import concurrent.futures,json,pandas as pd
from build_forecasts import ROOT,run_cell
if __name__=='__main__':
 cells=[c for c in json.loads((ROOT/'data/processed/cells.json').read_text()) if c['country']=='japan']
 with concurrent.futures.ProcessPoolExecutor(4) as pool:
  for f in concurrent.futures.as_completed([pool.submit(run_cell,c) for c in cells]):print(f.result(),flush=True)
 pd.concat([pd.read_csv(ROOT/f'results/forecast_japan_{c["cell"]}.csv',dtype={'cell':str}) for c in cells]).to_csv(ROOT/'data/processed/japan_forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
