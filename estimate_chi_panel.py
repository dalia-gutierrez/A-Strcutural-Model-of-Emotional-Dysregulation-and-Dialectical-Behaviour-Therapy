"""Comment 2: identify the income penalty chi from the MIDUS 1-2-3 panel.

The original chi came from a single cross-section (MIDUS 1), so the estimate
could reflect reverse causality (low income raising measured neuroticism/
depression) or time-invariant confounders (background, ability, education). The
panel lets us difference those out:

  pooled OLS     replicates the cross-sectional association
  person FE      identifies chi from WITHIN-person changes over waves, removing
                 every time-invariant confounder
  first-diff     same idea across consecutive waves

chi (model units, cbar=1) = -(coefficient of log income on the BPD-features index).
Standard errors are clustered by person (M2ID).
"""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf

D='data_MIDUS/'
WAVES={
 1:(D+'MIDUS1/02760-0001-Data.dta', dict(neuro='A1SNEURO',cons='A1SCONS',dep='A1PDEPDX',
        sex='A1PRSEX',inc='A1SHHTOT',age='A1PAGE_M2')),
 2:(D+'MIDUS2/04652-0001-Data.dta', dict(neuro='B1SNEURO',cons='B1SCONS1',dep='B1PDEPDX',
        sex='B1PRSEX',inc='B1STINC1',age='B1PAGE_M2')),
 3:(D+'MIDUS3/36346-0001-Data.dta', dict(neuro='C1SNEURO',cons='C1SCONS1',dep='C1PDEPDX',
        sex='C1PRSEX',inc='C1STINC',age='C1PRAGE')),
}

def load_wave(wave, path, m):
    cols=['M2ID']+list(m.values())
    df=pd.read_stata(path, columns=cols, convert_categoricals=False)
    out=pd.DataFrame({'M2ID':df['M2ID'].astype('int64'), 'wave':wave})
    for k,v in m.items(): out[k]=pd.to_numeric(df[v], errors='coerce')
    # cleaning per ICPSR codebooks
    out.loc[~out.neuro.between(1,4),'neuro']=np.nan
    out.loc[~out.cons.between(1,4),'cons']=np.nan
    out.loc[~out.dep.isin([0,1]),'dep']=np.nan
    out.loc[~out.sex.isin([1,2]),'sex']=np.nan
    out.loc[~out.age.between(18,100),'age']=np.nan
    out.loc[(out.inc<=0)|(out.inc>=9999990),'inc']=np.nan  # drop DK/refuse sentinels
    return out

panel=pd.concat([load_wave(w,p,m) for w,(p,m) in WAVES.items()], ignore_index=True)
# pooled standardisation of each component, then BPD-features index
for c in ['neuro','cons','dep']:
    panel[c+'_z']=(panel[c]-panel[c].mean())/panel[c].std(ddof=0)
panel['bpd']=panel['neuro_z']-panel['cons_z']+panel['dep_z']
panel['log_inc']=np.log(panel['inc'].clip(lower=1))
panel=panel.dropna(subset=['log_inc','bpd','age','sex']).copy()
print(f"panel: {len(panel)} person-waves, {panel.M2ID.nunique()} persons, "
      f"by wave {panel.wave.value_counts().sort_index().to_dict()}")

def chi_row(name, model, key='bpd'):
    b=model.params[key]; se=model.bse[key]
    print(f"  {name:18s} chi = {-b:6.3f}  (SE {se:.3f}, n={int(model.nobs)})")
    return dict(spec=name, chi=float(-b), se=float(se), n=int(model.nobs))

res=[]
# 1) pooled OLS with wave + sex + age controls, cluster by person
m1=smf.ols('log_inc ~ bpd + age + C(sex) + C(wave)', data=panel).fit(
    cov_type='cluster', cov_kwds={'groups':panel['M2ID']})
res.append(chi_row('pooled OLS', m1))
# 2) person fixed effects (within): demean by person
within=panel.copy()
for c in ['log_inc','bpd','age']:
    within[c]=within[c]-within.groupby('M2ID')[c].transform('mean')
within['w2']=(panel['wave']==2).astype(float); within['w3']=(panel['wave']==3).astype(float)
for c in ['w2','w3']:
    within[c]=within[c]-within.groupby('M2ID')[c].transform('mean')
# keep only persons with >=2 waves (others contribute nothing within)
nw=panel.groupby('M2ID')['wave'].transform('count'); within=within[nw>=2]
m2=smf.ols('log_inc ~ bpd + age + w2 + w3 - 1', data=within).fit(
    cov_type='cluster', cov_kwds={'groups':within['M2ID']})
res.append(chi_row('person FE (within)', m2))
# 3) first differences between consecutive waves
panel_s=panel.sort_values(['M2ID','wave'])
d=panel_s.groupby('M2ID').agg(list)
rows=[]
for pid,r in d.iterrows():
    ws=r['wave']
    for a in range(len(ws)-1):
        if ws[a+1]-ws[a]>=1:
            rows.append(dict(dlog=r['log_inc'][a+1]-r['log_inc'][a],
                             dbpd=r['bpd'][a+1]-r['bpd'][a],
                             dage=r['age'][a+1]-r['age'][a], M2ID=pid))
fd=pd.DataFrame(rows)
m3=smf.ols('dlog ~ dbpd + dage', data=fd).fit(cov_type='cluster', cov_kwds={'groups':fd['M2ID']})
res.append(chi_row('first differences', m3, 'dbpd'))

json.dump({'results':res,'original_cross_section_chi':0.158}, open('chi_panel.json','w'), indent=2)
print("\nsaved chi_panel.json")
