"""Common no-watermark peak-danger set: existing watermark true parity powers.

Read-only reuse of previous independent matrix construction; no old entry-point
is run. This script writes only next to itself. No physical-noise simulation.
Numerical directional searches/QMC are diagnostics, not global certificates.
"""
from pathlib import Path
import importlib.util
import json, hashlib, math, sys
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.stats import ncx2

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'watermark_boundary_final_20261114/power_comparison.py'
MODEL=HERE.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('previous_readonly_matrix_interface',OLD)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
E=10.;D=.1;L=10;HORIZON=15

def decoder_moment(X,scheme):
    if scheme=='whole_sign':return X
    if scheme=='coordinate_sign':return np.diag(np.diag(X))
    if scheme=='Haar_SO2':return np.trace(X)/2*np.eye(2)
    if scheme in ['Gaussian','normalized_Gaussian']:return X+np.trace(X)*np.eye(2)
    raise ValueError(scheme)

def prefix_moments(A,F,G,scheme):
    """E[B_k'B_k] for k=1..15; max of these is max E, not E max."""
    moments=[];maps0=[]
    deterministic=scheme.startswith('public_') or scheme=='identity'
    word=base.deterministic_word(scheme,L) if deterministic else None
    for k in range(1,HORIZON+1):
        Os=[np.linalg.matrix_power(A,t) for t in range(min(k,L))]
        Cs=[np.linalg.matrix_power(F,k-1-t)@G for t in range(min(k,L))]
        B0=sum((C@O for C,O in zip(Cs,Os)),start=np.zeros((2,2)))
        maps0.append(B0)
        if deterministic:
            B=sum((C@word[t]@O for t,(C,O) in enumerate(zip(Cs,Os))),start=np.zeros((2,2)))
            S=B.T@B
        else:
            S=sum((O.T@decoder_moment(C.T@C,scheme)@O for C,O in zip(Cs,Os)),start=np.zeros((2,2)))
            if scheme in ['Gaussian','normalized_Gaussian']:S+=4*B0.T@B0
            if scheme=='normalized_Gaussian':S*=base.NU
        moments.append((S+S.T)/2)
    return np.array(moments),np.array(maps0)

def roots_quadratic_on_circle(M,target=0.):
    center=np.trace(M)/2-target;a=(M[0,0]-M[1,1])/2;b=M[0,1]
    amp=np.hypot(a,b)
    if amp<1e-17:return []
    v=-center/amp
    if abs(v)>1+1e-11:return []
    ph=np.arctan2(b,a);ac=np.arccos(np.clip(v,-1,1))
    return [float(((ph+ac)/2)%np.pi),float(((ph-ac)/2)%np.pi)]

def geometry(A,F,G):
    W,_,_,_=base.analytic_matrices(A,F,G,'identity')
    ev,U=eigh(W);V=U@np.diag(1/np.sqrt(ev))@U.T
    _,Bs=prefix_moments(A,F,G,'identity')
    S0=np.einsum('kji,kjl->kil',Bs,Bs)
    T=np.array([V.T@S@V for S in S0])
    def vals(t):
        u=np.array([np.cos(t),np.sin(t)])
        return np.einsum('i,kij,j->k',u,T,u)
    # All possible envelope crossings, then retain changes of the active max.
    allroots=[0.,np.pi]
    for i in range(HORIZON):
        for j in range(i):allroots.extend(roots_quadratic_on_circle(T[i]-T[j]))
    allroots=sorted(set(round(x,13) for x in allroots))
    active=[]
    for i in range(1,len(allroots)-1):
        t=allroots[i];lo=allroots[i-1];hi=allroots[i+1]
        if int(np.argmax(vals((t+lo)/2)))!=int(np.argmax(vals((t+hi)/2))):active.append(t)
    boundaries=[]
    for M in T:
        for t in roots_quadratic_on_circle(M,D/E):
            if abs(max(vals(t))-D/E)<1e-9:boundaries.append(t)
    breaks=sorted(set([0.,np.pi]+active+boundaries))
    intervals=[]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        mid=(lo+hi)/2;q=vals(mid)
        intervals.append(dict(lo=lo,hi=hi,active_bias_index=int(np.argmax(q))+1,
                              dangerous=bool(max(q)>=D/E)))
    # Peak P1 is max over times of a quadratic maximum; finite max commutes.
    maxima=[]
    for k,S in enumerate(S0):
        lam,U=eigh(S,W);xi=np.sqrt(E)*U[:,-1]
        maxima.append(dict(bias_index=k+1,value=float(E*lam[-1]),initial_state=xi.tolist()))
    peak=max(maxima,key=lambda p:p['value'])
    return W,V,S0,T,dict(bias_indices=list(range(1,16)),threshold_D=D,energy_E=E,
        envelope_switch_angles=active,danger_boundary_angles=sorted(boundaries),
        intervals=intervals,per_time_P1=maxima,peak_P1=peak,
        total_dangerous_angle_measure=float(sum(q['hi']-q['lo'] for q in intervals if q['dangerous'])),
        angular_period=math.pi,angle_coordinates='xi = r W^(-1/2) [cos(theta), sin(theta)]')

def point_metrics(xi,Qs,S0,S):
    ncps=np.maximum(np.einsum('i,nij,j->n',xi,Qs,xi),0)
    zero=np.einsum('i,kij,j->k',xi,S0,xi)
    actual=np.einsum('i,kij,j->k',xi,S,xi)
    peak0=float(max(zero));peak=float(max(actual))
    return dict(initial_state=xi.tolist(),Pd=float(np.mean(ncx2.sf(base.TAU,8,ncps))),
        expected_ncp=float(np.mean(ncps)),no_watermark_potential_peak=peak0,
        no_watermark_potential_peak_bias_index=int(np.argmax(zero))+1,
        scheme_max_expected_bias_squared=peak,
        scheme_max_expected_bias_index=int(np.argmax(actual))+1,
        scheme_expected_terminal_bias_squared=float(actual[9]),
        no_watermark_terminal_bias_squared=float(zero[9]),
        peak_expected_amplification=peak/peak0 if peak0 else None,
        moment_peak_definition='max_k E ||b_k||^2; not E max_k ||b_k||^2',
        per_time_no_watermark_potential=zero.tolist(),per_time_expected_bias_squared=actual.tolist())

def solve_danger(Qs,W,V,S0,T,S,geom,grid=129):
    def point(t,ret=False):
        e=np.array([np.cos(t),np.sin(t)]);u=V@e
        qmax=max(np.einsum('i,kij,j->k',e,T,e))
        if qmax<D/E*(1-1e-10):return (np.inf,None) if ret else np.inf
        xi=np.sqrt(D/qmax)*u
        lam=np.maximum(np.einsum('i,nij,j->n',xi,Qs,xi),0)
        val=float(np.mean(ncx2.sf(base.TAU,8,lam)))
        return (val,xi) if ret else val
    angles=set(np.linspace(0,np.pi,grid))
    angles.update(geom['envelope_switch_angles']);angles.update(geom['danger_boundary_angles'])
    angles=np.array(sorted(angles));values=np.array([point(t) for t in angles])
    candidates=[(float(v),float(t)) for v,t in zip(values,angles) if np.isfinite(v)]
    refinement_count=0
    # Do not optimize across nonsmooth envelope switches or feasibility gaps.
    for interval in geom['intervals']:
        if not interval['dangerous']:continue
        lo=interval['lo'];hi=interval['hi']
        loc=[i for i,t in enumerate(angles) if lo-1e-12<=t<=hi+1e-12 and np.isfinite(values[i])]
        if not loc:
            mid=(lo+hi)/2;candidates.append((point(mid),mid));continue
        best=sorted(loc,key=lambda i:values[i])[:2]
        for i in best:
            a=max(lo,float(angles[max(0,i-1)]));b=min(hi,float(angles[min(len(angles)-1,i+1)]))
            if b-a>1e-12:
                opt=minimize_scalar(point,bounds=(a,b),method='bounded',options={'xatol':1e-10})
                if np.isfinite(opt.fun):candidates.append((float(opt.fun),float(opt.x)))
                refinement_count+=1
    if not candidates:return dict(feasible=False)
    val,t=min(candidates);_,xi=point(t,True)
    ans=point_metrics(xi,Qs,S0,S)
    ans.update(feasible=True,angle=t,energy=float(xi@W@xi),radial_boundary_used=True,
        angular_grid=grid,local_refinements=refinement_count,global_optimum_certified=False)
    return ans

def run():
    out=dict(date='2026-11-14',model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        reused_matrix_source_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),
        estimator_and_fusion_unchanged=True,attack_class_unchanged=True,
        danger_set='xi^T W10 xi<=10 and max_{k=1..15} ||B0,k xi||^2>=.1',
        common_danger_set_for_all_schemes=True,bias_indices=list(range(1,16)),
        first_full_window='t=0..4, score at packet4',statistic_df=8,alpha_test=base.alpha,
        threshold=base.TAU,event15_FA_bound=.05,single_test_is_not_event_detection=True,
        physical_noise_samples=0,new_watermark_search=False,models={})
    for name,md in base.data['models'].items():
        A=np.array(md['A']);nd=md['nodes'][1];F=np.array(nd['posterior_F']);G=np.array(nd['G'])
        W,V,S0,T,geom=geometry(A,F,G)
        row=dict(A=A.tolist(),F=F.tolist(),G=G.tolist(),W=W.tolist(),
            no_watermark_prefix_moment_matrices=S0.tolist(),geometry=geom,schemes={})
        out['models'][name]=row
        for scheme in base.SCHEMES:
            S,_=prefix_moments(A,F,G,scheme)
            levels=[1024,4096,8192] if scheme in ['Gaussian','normalized_Gaussian','Haar_SO2'] else [0]
            probes=[]
            for n in levels:
                Qs,meta=base.path_Qs(A,scheme,n)
                sol=solve_danger(Qs,W,V,S0,T,S,geom)
                weakest=base.solve_angle(Qs,W)
                xweak=np.array(weakest['initial_state'])
                weakmetrics=point_metrics(xweak,Qs,S0,S);weakmetrics.update(angle=weakest['angle'],energy=float(xweak@W@xweak),global_optimum_certified=False)
                xdanger=np.array(geom['peak_P1']['initial_state'])
                dangermetrics=point_metrics(xdanger,Qs,S0,S);dangermetrics.update(energy=float(xdanger@W@xdanger))
                probes.append(dict(integration=meta,danger_set_minimum=sol,
                    full_energy_minimum=weakmetrics,no_watermark_peak_most_dangerous=dangermetrics))
                print(name,scheme,meta['nodes'],'danger Pd',sol.get('Pd'),'energy',sol.get('energy'),'E10weak',weakmetrics['Pd'],flush=True)
            # Curves use the highest declared deterministic integration level.
            angles=np.linspace(0,np.pi,257)
            curves=[]
            for t in angles:
                e=np.array([np.cos(t),np.sin(t)]);u=V@e;qmax=float(max(np.einsum('i,kij,j->k',e,T,e)))
                xiE=np.sqrt(E)*u;dangerous=E*qmax>=D*(1-1e-12)
                curve=dict(angle=float(t),full_energy=point_metrics(xiE,Qs,S0,S),dangerous=bool(dangerous))
                if dangerous:
                    xid=np.sqrt(D/qmax)*u;curve['minimum_danger_energy']=float(D/qmax)
                    curve['minimum_danger']=point_metrics(xid,Qs,S0,S)
                curves.append(curve)
            row['schemes'][scheme]=dict(prefix_expected_moment_matrices=S.tolist(),probes=probes,curves=curves)
            (HERE/'danger_detection_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'danger_detection_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':run()
