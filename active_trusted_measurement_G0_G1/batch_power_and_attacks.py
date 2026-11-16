"""Deterministic Gaussian power, standard finite P3 characterization, no new sensors in the filter."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;P=H.parent
import measurement_model as mm
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.stats import ncx2,chi2
from scipy.integrate import quad
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8')
def secrets(m):
    old=P/'attack_gain_switch_audit_20261115';seeds=json.loads((old/'sampling_seed_map.json').read_text('utf8'))[m.name]['watermark']
    return np.stack([mm.ref.original_watermarks(m,32,s,.5,.5)[0][:,0] for s in seeds])
def P1(m,Gaussian=False):
    S,W=mm.physical_damage_matrices(m,Gaussian);rows=[]
    for k,V in enumerate(S):
        ev,U=eigh(V,W);u=U[:,-1];xi=u*np.sqrt(10/(u@W@u))*(1-1e-12)
        rows.append(dict(k=k,value=float(10*ev[-1]),xi=xi.tolist()))
    return max(rows,key=lambda z:z['value']),S,W
def power(G,df,xi,alpha=.05):
    ncp=np.maximum(0.,np.einsum('i,mij,j->m',xi,np.asarray(G).reshape(-1,2,2),xi))
    return float(ncx2.sf(chi2.isf(alpha,df),df,ncp).mean())
def p3(G,df,S,W,Gaussian=False):
    """Fixed detector-only batch. No switched-gain radial argument is used."""
    candidates=[]
    if not Gaussian:
        for h,V in enumerate(S):
            if .1>10*eigh(V,W,eigvals_only=True)[-1]+1e-12:continue
            _,U=eigh(np.asarray(G).reshape(-1,2,2)[0],V)
            candidates.extend(U[:,j] for j in range(2))
            B=V-.01*W;roots=np.roots([B[1,1],2*B[0,1],B[0,0]])
            candidates.extend(np.array([1.,float(z.real)]) for z in roots if abs(z.imag)<1e-10)
            if abs(B[1,1])<1e-12:candidates.append(np.array([0.,1.]))
    else:
        angles=np.linspace(0,np.pi,181,endpoint=False);candidates=[np.array([np.cos(t),np.sin(t)]) for t in angles]
    def evaluate(u):
        h=max(u@v@u for v in S)
        if h<=0:return None
        xi=u*np.sqrt(.1/h)*(1+1e-12);energy=xi@W@xi
        if energy>10*(1+1e-13):return None
        return dict(xi=xi.tolist(),energy=float(energy),expected_peak_damage=float(max(xi@v@xi for v in S)),
          Pd=power(G,df,xi),theta=float(np.arctan2(u[1],u[0])%np.pi))
    valid=[z for u in candidates if (z:=evaluate(u)) is not None]
    assert valid,'P3 infeasible; must not report detection100%'
    if Gaussian:
        vals=[evaluate(np.array([np.cos(t),np.sin(t)])) for t in np.linspace(0,np.pi,181,endpoint=False)]
        for i,z in enumerate(vals):
            if z is None:continue
            prev=vals[(i-1)%181];nxt=vals[(i+1)%181]
            if z['Pd']<=min(prev['Pd'] if prev else 2,nxt['Pd'] if nxt else 2):
                t=z['theta'];f=lambda x:(evaluate(np.array([np.cos(x),np.sin(x)])) or {'Pd':2})['Pd']
                ans=minimize_scalar(f,bounds=(t-np.pi/181,t+np.pi/181),method='bounded',options={'xatol':1e-10})
                r=evaluate(np.array([np.cos(ans.x),np.sin(ans.x)]))
                if r:valid.append(r)
    best=min(valid,key=lambda r:r['Pd']);best['candidate_count']=len(valid)
    best['method']='181-angle search plus local refinement, finite old-secret average; not global probability certificate' if Gaussian else 'all generalized eigenstationary directions plus energy-boundary rays for each peak-time quadratic; floating arithmetic, no interval certificate'
    return best
def main():
    source={};matrices={};sensitivity={};cost=.5*(.5**-2+quad(lambda t:np.exp(-t)/(.5+t)**2,0,np.inf)[0])
    for name in ('unstable','stable'):
        m=mm.ref.load_model(name);M=secrets(m);a,S,W=P1(m);ag,Sg,_=P1(m,True);normal=mm.normal_covariances(m)
        G0,df0,_=mm.one_batch_info(m,7,'B0');G1,df1,raw=mm.one_batch_info(m,7,'B1')
        assert np.linalg.norm(G0)<1e-10 and df0==8 and df1==2
        GB2=np.stack([mm.one_batch_info(m,7,'B1',M=v)[0] for v in M])
        schemes={'B0':(G0[None],df0,S,a),'B1':(G1[None],df1,S,a),'B2':(GB2,2,Sg,ag)}
        auxrows={}
        for c in (0,1):
            phases=[]
            for phase in range(5):
                aux=mm.schedule(cycle_phase=phase,coordinate=c)
                G,df,r=mm.one_batch_info(m,7,'B1',aux);assert df==2
                assert np.linalg.eigvalsh(G-G1).min()>-1e-9
                pp=p3(G[None],2,S,W)
                phases.append(dict(phase=phase,aux_times=[t for t,j,z in r['rows'] if j==3],G=G.tolist(),P3=pp,Pd_at_P1=power(G[None],2,np.array(a['xi']))))
                matrices[name+f'__coord{c}__phase{phase}__G']=G
            auxrows[str(c)]=dict(phases=phases,min_phase_P3_power=min(v['P3']['Pd'] for v in phases),
              best_phase_for_aligned_batch=max(phases,key=lambda z:z['P3']['Pd'])['phase'],
              unknown_start_minimax_phase_tie=True,
              statement='two samples per public10-cycle: any nonuniform spacing has a5-step empty window and attains the B1 worst-start bound; uniform spacing is a standard optimal coverage baseline for this declared marginal-power objective, not optimal online danger risk')
            # Phase0 primary fixed periodic scheme; B4 worst-start baseline is same5-phase family.
            G=np.array(phases[0]['G']);schemes['B3_coord'+str(c)]=(G[None],2,S,a)
            for variance in (.25,.5,1.):
              for delay in (0,1):
               for rho in (0.,.5):
                aux=mm.schedule(coordinate=c,variance=variance,delay=delay,rho=rho)
                G,df,r=mm.one_batch_info(m,7,'B1',aux)
                sensitivity[f'{name}/coord{c}/R{variance}/delay{delay}/rho{rho}']=dict(P3=p3(G[None],df,S,W),
                  Pd_at_P1=power(G[None],df,np.array(a['xi'])),available_aux_rows=sum(j==3 for _,j,_ in r['rows']))
        rows={}
        for b,(G,df,Sm,ap) in schemes.items():
            P=p3(G,df,Sm,W,b=='B2');xi=np.array(ap['xi']);rows[b]=dict(df=df,P1=ap,Pd_at_P1=power(G,df,xi),
              P3=P,min_ncp_at_full_energy=float(10*eigh(G.mean(axis=0),W,eigvals_only=True)[0]),
              note='this eigenvalue is minimum expected noncentrality, not minimum real power' if b=='B2' else 'minimum batch noncentrality under FULL E=10')
            matrices[name+'__'+b+'__G']=G
        source[name]=dict(schemes=rows,auxiliary=auxrows,normal_target_MSE_average7to31=float(np.trace(normal[7:],axis1=1,axis2=2).mean()),
          normal_all_baselines_identical=True,P1_original_gain_unchanged=True,
          existing_local_structural_rank=2,normal_encoded_signal_energy_multiplier_Gaussian=cost)
        matrices[name+'__W']=W;matrices[name+'__S_identity']=S;matrices[name+'__S_Gaussian']=Sg;matrices[name+'__normal_covariances']=normal
        matrices[name+'__B1_raw_cov']=raw['cov'];matrices[name+'__B1_parity_cov']=raw['theta']
        print(json.dumps(dict(completed=name,batch_P3={b:round(z['P3']['Pd'],7) for b,z in rows.items()})),flush=True)
    dump(H/'batch_power_results.json',dict(date='2026-11-16',batch_FA=.05,batch_times=[7,8,9,10,11],
      damage_times='attack ten posteriors; auxiliary never enters estimator',models=source,sensitivity=sensitivity,
      Gaussian_secret_nodes=128,secret_average_is_diagnostic_not_certified=True,new_physical_noise_samples=0,
      no_new_adaptive_or_online_sampling_design=True))
    np.savez_compressed(H/'batch_matrices.npz',**matrices)
if __name__=='__main__':main()
