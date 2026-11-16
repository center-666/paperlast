"""Finite unknown-onset scan: exact Gaussian marginals, dependent-event bounds.

NO independent-window approximation, Monte Carlo observations, or online policy.
Gaussian watermark averaging uses the old secret bank and is labeled diagnostic.
"""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
import measurement_model as mm
from batch_power_and_attacks import secrets,P1,dump
import numpy as np
from scipy.stats import chi2,ncx2
ENDS=np.arange(4,32);STARTS=[7,8,9,10,11]
def kernel(m,kind,aux,Ta,Ms):
    N=len(Ms);G=np.zeros((N,28,1 if kind=='B0' else 9,2,2));B=np.zeros((N,32,2,2))
    for n,M in enumerate(Ms):
        prev=np.zeros((2,2))
        for t in range(32):
            inj=np.zeros((2,2))
            if Ta<=t<Ta+10:
                inj=m.H[0]@np.linalg.matrix_power(m.A,t-Ta)
                if M is not None:inj=M[t]@inj
            prev=m.F(1)@prev+m.G[1]@inj;B[n,t]=prev
        for h,end in enumerate(ENDS):
            infos,_=mm.info_for_attack(m,int(end),kind,aux,Ta,M)
            assert len(infos)==G.shape[2]
            for q,(info,df) in enumerate(infos):G[n,h,q]=info
    return G,np.einsum('mtji,mtjk->mtik',B,B),df
def bounds(G,S,df,xis,Ta,details=False):
    lam=np.maximum(0.,np.einsum('ai,mthij,aj->amth',xis,G,xis));threshold=chi2.isf(.05/(28*G.shape[2]),df)
    p=ncx2.sf(threshold,df,lam)
    damage=np.einsum('ai,mtij,aj->amt',xis,S,xis);bad=damage>=.1
    has=bad.any(axis=2);tau=np.where(has,bad.argmax(axis=2),1000)
    allowed=ENDS[None,None,:,None]<=tau[:,:,None,None]
    selected=np.where(allowed,p,0.);lower=np.maximum(0.,1-selected.sum(axis=(2,3)))*has;upper=(1-selected.max(axis=(2,3)))*has
    out=dict(risk_lower=lower.mean(axis=1),risk_upper=upper.mean(axis=1),danger_reachable_fraction=has.mean(axis=1))
    if details:
        # P(first alarm<=end) from the union of all earlier correlated tests.
        Flo=np.maximum.accumulate(p.max(axis=3),axis=2);Fhi=np.minimum(1.,np.cumsum(p.sum(axis=3),axis=2))
        mask=ENDS>=Ta
        out.update(alarm_CDF_lower=Flo.mean(axis=1),alarm_CDF_upper=Fhi.mean(axis=1),
          truncated_delay_lower=(1-Fhi[:,:,mask]).sum(axis=2).mean(axis=1),truncated_delay_upper=(1-Flo[:,:,mask]).sum(axis=2).mean(axis=1))
    return out
def global_noise_map(m,rows,aux):
    # Primitive independent standard Gaussian vector: x0, w0:30, v0/v1/v2,
    # and all possible auxiliary noises. No physical path is drawn.
    n=2;N=2+31*2+32*3*2+32;B=np.zeros((len(rows),N));offset_v=64;offset_aux=256
    for i,(t,j,c) in enumerate(rows):
        h=m.H[j][c] if j<3 else np.eye(2)[c]
        B[i,:2]=h@np.linalg.matrix_power(m.A,t)
        for r in range(t):B[i,2+2*r:4+2*r]=h@np.linalg.matrix_power(m.A,t-1-r)@np.linalg.cholesky(m.Q)
        if j<3:B[i,offset_v+(t*3+j)*2:offset_v+(t*3+j)*2+2]=np.linalg.cholesky(m.R[j])[c]
        else:
            cR=np.linalg.cholesky(m.R[1])[c];B[i,offset_v+(t*3+1)*2:offset_v+(t*3+1)*2+2]=aux['rho']*np.sqrt(aux['variance']/m.R[1][c,c])*cR
            B[i,offset_aux+t]=np.sqrt(aux['variance']*(1-aux['rho']**2))
    return B
def main():
    batch=json.loads((H/'batch_power_results.json').read_text('utf8'));results={};data={};covchecks={}
    for name in ('unstable','stable'):
        m=mm.ref.load_model(name);M=secrets(m);ones=[None]
        configs=[('B0','B0',None,ones),('B1','B1',None,ones),('B2','B1',None,list(M)),
          ('B3_coord0','B1',mm.schedule(coordinate=0),ones),('B3_coord1','B1',mm.schedule(coordinate=1),ones)]
        _,_,W=P1(m)
        angles=np.linspace(0,np.pi,181,endpoint=False);xs=[]
        for frac in (.25,.5,1.):
            for theta in angles:
                u=np.array([np.cos(theta),np.sin(theta)]);xs.append(u*np.sqrt(10*frac/(u@W@u))*(1-1e-12))
        xs=np.array(xs)
        for label,kind,aux,bank in configs:
            scheme=batch['models'][name]['schemes'][label];profiles=np.array([scheme['P1']['xi'],scheme['P3']['xi']]);out={}
            for Ta in STARTS:
                G,S,df=kernel(m,kind,aux,Ta,bank);details=bounds(G,S,df,profiles,Ta,True)
                amax=dict(lower=-1.,upper=-1.);witness={}
                for i in range(0,len(xs),24):
                    block=xs[i:i+24];v=bounds(G,S,df,block,Ta)
                    for key in ('lower','upper'):
                        q=int(np.argmax(v['risk_'+key]));value=float(v['risk_'+key][q])
                        if value>amax[key]:
                            amax[key]=value;witness[key]=dict(xi=block[q].tolist(),energy=float(block[q]@W@block[q]),
                              risk_interval=[float(v['risk_lower'][q]),float(v['risk_upper'][q])],index=i+q)
                out[str(Ta)]=dict(profiles={lab:{k:(v[p].tolist() if np.ndim(v[p]) else float(v[p])) for k,v in details.items()} for p,lab in enumerate(('P1','P3_batch'))},
                  searched_max_bound_proxy=amax,selected_old_family_parameters=witness,not_true_global_risk_optimization=True)
                # Source detector/kernel arrays allow independent recomputation;
                # keep the five true-start kernels, no simulated observations.
                data[f'{name}__{label}__Ta{Ta}__G']=G;data[f'{name}__{label}__Ta{Ta}__S']=S
            # Explicit cross covariance of TWO overlapping windows, not independence.
            maps=[]
            for end in (8,9):
                raw=mm.raw_matrices(m,mm.rows_for_window(end,kind,aux),aux)
                R=raw['white']@global_noise_map(m,raw['rows'],aux);assert np.max(np.abs(R@R.T-np.eye(len(R))))<1e-9
                maps.append(R)
            cross=maps[0]@maps[1].T
            covchecks[name+'/'+label]=dict(cross_block_Frobenius_norm=float(np.linalg.norm(cross)),
              not_independent=bool(np.linalg.norm(cross)>1e-6),unit_marginal_covariance_error=float(max(np.max(np.abs(v@v.T-np.eye(len(v)))) for v in maps)))
            data[name+'__'+label+'__overlap_cov']=np.block([[maps[0]@maps[0].T,cross],[cross.T,maps[1]@maps[1].T]])
            results[name+'/'+label]=dict(starts=out,threshold=float(chi2.isf(.05/(28*(1 if kind=='B0' else 9)),8 if kind=='B0' else 2)),
              finite_event_FA_bound=.05,number_tests=28*(1 if kind=='B0' else 9),conditional_noise_bound_formula_valid=True,
              floating_numeric_bounds_not_interval_certified=True,
              Gaussian_secret_average_error_certified=False if label=='B2' else None,
              exact_joint_first_alarm_probability_computed=False)
            print(json.dumps(dict(completed=name+'/'+label,risk_bound_proxies={t:z['searched_max_bound_proxy'] for t,z in out.items()})),flush=True)
    dump(H/'online_bounds_results.json',dict(date='2026-11-16',results=results,covariance_checks=covchecks,
      window_ends=ENDS.tolist(),true_start_phases=STARTS,energy=10,danger=.1,
      finite_search_angles=181,energy_fractions=[.25,.5,1.],new_physical_noise_samples=0,
      interpretation='CDF and risk intervals use exact conditional Gaussian marginals plus union/Frechet bounds. B2 averages over128 old secrets only; never call these exact unconditional probabilities or global attack optima.',
      B4_scope='optimal predetermined coverage baseline proven for marginal five-step batch objective; online dangerous-before-alarm optimum not asserted'))
    np.savez_compressed(H/'online_kernels.npz',**data)
if __name__=='__main__':main()
