"""Parametric bootstrap of the welfare gain, drawing the primitives from their
Table-1 sampling distributions (point estimates +- reported SE), with alpha_mu
anchored to the Azevedo2024 meta-analysis. Companion to bootstrap_meta.py
(which centred chi and B on their raw cross-sectional values 0.158 / 3.92);
this version centres them on the Table-1 panel-FE / 95th-pct point estimates.
Because the gain is ~flat in chi and B, the two give nearly the same interval.
Run:  PYTHONPATH=. python3 bootstrap_table1.py <offset> <N>
"""
import json, sys, numpy as np
from multiprocessing import Pool
from dbt_model.model import Params
from dbt_model.solver_implicit import solve_hjb_implicit
MIDUS=46072.0
ICs=[(-1.5,0.5),(-1.0,0.5),(-0.5,0.5),(-1.5,1.0),(-1.0,1.0),(-0.5,1.0),(-1.0,0.2),(0.0,0.5)]
def val(sol,x0,Z0):
    xg=sol['x_grid'];Zg=sol['Z_grid'];V=sol['V']
    i=int(np.clip(np.searchsorted(xg,x0)-1,0,len(xg)-2));j=int(np.clip(np.searchsorted(Zg,Z0)-1,0,len(Zg)-2))
    fx=(x0-xg[i])/(xg[i+1]-xg[i]);fz=(Z0-Zg[j])/(Zg[j+1]-Zg[j])
    return (1-fx)*(1-fz)*V[i,j]+fx*(1-fz)*V[i+1,j]+(1-fx)*fz*V[i,j+1]+fx*fz*V[i+1,j+1]
def one(seed):
    r=np.random.default_rng(seed)
    p=Params(utility_type='separable', omega=-12.0, phi_0=4.0, sigma_0=0.6, mu_0=-0.002,
        r=0.04, rho=0.20, beta_phi=0.3, beta_mu=0.4, beta_sigma=0.3, Nx=81, NZ=41, Z_max=3.0,
        chi=float(np.clip(r.normal(0.024,0.009),0.0,None)),      # Table 1 panel-FE
        gamma=float(np.clip(r.normal(0.586,0.067),0.05,None)),   # Table 1
        B=float(np.clip(r.normal(3.85,0.19),1.0,None)),          # Table 1 95th pct
        relief_lambda=float(np.clip(r.normal(0.75,0.06),0.2,None)),
        delta_mu=float(np.clip(r.normal(0.08,0.019),0.0,None)),
        kappa=0.05,
        alpha_phi=float(np.clip(r.normal(0.28,0.32),0,None)),
        alpha_mu=float(np.clip(r.normal(0.90,0.25),0,None)),     # meta-anchor
        alpha_sigma=float(np.clip(r.normal(0.51,1.43),0,None)))
    su=solve_hjb_implicit(p,constrained=False,n_xi=21,max_iter=15)
    sc=solve_hjb_implicit(p,constrained=True,max_iter=15)
    return float(np.mean([np.exp(p.r*(val(su,x,z)-val(sc,x,z)))-1 for x,z in ICs]))
if __name__=='__main__':
    off=int(sys.argv[1]); N=int(sys.argv[2])
    f=open('boot_table1.jsonl','a')
    with Pool(2) as pool:
        for g in pool.imap_unordered(one, range(off,off+N)):
            f.write(json.dumps(g)+'\n'); f.flush()
    f.close()
    print(f'appended {N} draws')
