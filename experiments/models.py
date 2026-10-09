"""Fixed feature families and probability learners for development and held-out tests."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
TARGETS=['ge35','band35_45','ge45']
GROUPS={'activity':['n30','time_since_last','span50','mean_interval50','cv_interval50','magnitude_last','max_magnitude7'],
'magnitude':['b100','db0','db1','db2','db3','db4','mag_sd30','gr_residual','magnitude_excess','log_sqrt_energy30'],
'geometry':['last_latitude','last_longitude','mean_depth30','clustering50'],
'innovation':['innovation30','innovation90','innovation365','residual_persistence','distributed_residual']}
GROUPS['original']=GROUPS['activity']+GROUPS['magnitude']+GROUPS['geometry']
FAMILIES=['activity','magnitude','geometry','original','innovation','memory','union']
GRID=[{'learner':s,'C':c} for s in ['linear','spline'] for c in [.01,.1,1.,10.]]+[{'learner':'boost','depth':h,'trees':n,'l2':10.} for h in [1,2,3] for n in [150,350]]

def load(co):
 d=pd.read_csv(ROOT/f'data/processed/{co}_forecasts.csv.gz',dtype={'cell':str});d['date']=pd.to_datetime(d.origin);d['end']=d.date+pd.Timedelta(days=30)
 for t in TARGETS:
  for b in ['etas','poisson','adaptive90','adaptive365']:d[f'logit_{b}_{t}']=logit(np.clip(d[f'{b}_{t}'],1e-6,1-1e-6))
  d['mix_activity_'+t]=d['rate_acceleration']*d['memory_persist_'+t]
  d['mix_magnitude_'+t]=(d['b100']-1)*d['memory_persist_'+t]
  d['mix_depth_'+t]=d['depth_change']*d['innovation90']
  d['mix_geometry_'+t]=d['spatial_entropy90']*d['memory_persist_'+t]
 for c in ['n30','time_since_last','span50','mean_interval50','cv_interval50','gr_residual','spatial_radius50','mu','K','c']:d[c]=np.log1p(d[c].clip(lower=0))
 return d

def groupcols(name,t):
 if name=='memory':return [f'{m}{w}_{t}' for m in ['memory','frequency'] for w in [3,6,12,24]]+['memory_persist_'+t]
 if name=='union':return GROUPS['original']+GROUPS['innovation']+groupcols('memory',t)+[f'mix_{x}_{t}' for x in ['activity','magnitude','depth','geometry']]
 return GROUPS.get(name,[])

def columns(name,t):
 base=['logit_etas_'+t]
 if name!='etas_cal':base += [f'logit_{b}_{t}' for b in ['poisson','adaptive90','adaptive365']]
 if name=='state':base+=['mu','K','alpha','c','omori_p','branching','beta']
 return base+groupcols(name,t)+['center_lat','center_lon']

class Learner:
 def __init__(self,cols,config,target):self.cols=cols;self.cfg=config;self.target=target
 def fit(self,d,y):
  cfg=self.cfg;trans=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler())
  if cfg['learner']=='spline':trans=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),SplineTransformer(n_knots=3,degree=2,include_bias=False),StandardScaler())
  self.enc=ColumnTransformer([('x',trans,self.cols),('cell',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['cell'])]);x=self.enc.fit_transform(d)
  if cfg['learner']=='boost':
   self.model=XGBClassifier(n_estimators=cfg['trees'],max_depth=cfg['depth'],learning_rate=.03,reg_lambda=cfg['l2'],min_child_weight=5,subsample=1,colsample_bytree=1,tree_method='hist',n_jobs=1,random_state=271828,objective='binary:logistic',eval_metric='logloss')
   self.model.fit(x,y,base_margin=d['logit_etas_'+self.target].to_numpy())
  else:self.model=LogisticRegression(C=cfg['C'],max_iter=3000,solver='lbfgs').fit(x,y)
  return self
 def predict(self,d):
  x=self.enc.transform(d)
  p=self.model.predict_proba(x,base_margin=d['logit_etas_'+self.target].to_numpy())[:,1] if self.cfg['learner']=='boost' else self.model.predict_proba(x)[:,1]
  return np.clip(p,1e-6,1-1e-6)

def loss(y,p):
 p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6);return -(np.asarray(y)*np.log(p)+(1-np.asarray(y))*np.log1p(-p))
