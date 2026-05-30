"""Estimate the regulation-budget and relief parameters from the Fisher2017 EMA.

  B            regulation-effort budget (z+xi<=B), affect-SD units
               = 95th pct of observed ineffective-coping intensity
  relief_lambda relief ceiling: most momentary affect-relief coping can buy
               = within-person affect headroom (typical -> personal-best mood)
  delta_mu     how much coping worsens NEXT-period affect (the "ineffective"
               spiral) = lagged within-person dNA(t+1)/dCope(t), affect-SD/z-unit
"""
import json, numpy as np, pandas as pd
NA=['irritable','restless','worried','guilty','afraid','angry','hopeless','down','tension','threatened']
COPE=['ruminate','avoid_act','reassure','procrast']

def main(path='data_ESM/data_Fisher2017.csv'):
    d=pd.read_csv(path)
    d['na']=d[NA].mean(axis=1); d['cope']=d[COPE].mean(axis=1)
    d=d.dropna(subset=['na','cope','subj_id']).copy()
    sd=d['na'].std()
    d['beep']=d.groupby('subj_id').cumcount(); d['day']=d['beep']//4
    # B: 95th pct of daily coping in affect-SD units
    daily=d.groupby(['subj_id','day'],as_index=False)[['na','cope']].mean()
    B=float((daily['cope']/sd).quantile(0.95))
    # lambda: within-person headroom (mean -> 10th pct NA) in affect-SD units
    g=d.groupby('subj_id')['na']
    lam=float(((g.transform('mean')-g.transform(lambda s:s.quantile(0.10)))/sd).mean())
    # delta: lagged within-person effect of coping on next-period NA
    d['na_next']=d.groupby('subj_id')['na'].shift(-1); dd=d.dropna(subset=['na_next']).copy()
    for c in ['na','cope','na_next']:
        dd[c+'_dm']=dd[c]-dd.groupby('subj_id')[c].transform('mean')
    X=np.column_stack([dd['cope_dm'],dd['na_dm'],np.ones(len(dd))])
    b=np.linalg.lstsq(X,dd['na_next_dm'],rcond=None)[0][0]
    delta=float(b*sd/sd)  # per z-unit, in affect-SD
    out=dict(B=B, relief_lambda=lam, delta_mu=delta, affect_sd=float(sd),
             n_subjects=int(d.subj_id.nunique()))
    json.dump(out, open('budget_params.json','w'), indent=2)
    print(json.dumps(out, indent=2))
    return out

if __name__=='__main__': main()
