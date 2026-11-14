"""Fixed-probe finite first-alarm/risk bounds, no physical-noise simulation.

Parity windows share noise: use union/max bounds conditional on the full
exogenous watermark path. Current spatial tests are conditionally independent:
multiply before integrating over the full watermark path. No new detector.
"""
from pathlib import Path
import importlib.util,sys,itertools,json,hashlib
import numpy as np
from scipy.linalg import eigh
from scipy.stats import chi2,ncx2,qmc,norm
H=Path(__file__).resolve().parent
OLD=H.parent/'watermark_boundary_final_20261114/power_comparison.py'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('first_alarm_matrix_source',OLD)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
DATA=H/'danger_detection_comparison.json'
d=json.loads(DATA.read_text(encoding='utf-8'))
ALPHA=1/300;E=10.;D=.1;L=10;T=15
TAU_PARITY=float(chi2.isf(ALPHA,8));TAU_SPATIAL=float(chi2.isf(ALPHA,2))
SEEDS=dict(coordinate_sign=2026111431,Gaussian=2026111432,Haar_SO2=2026111433)

def fixed_probes(md):
    W=np.array(md['W']);S=np.array(md['no_watermark_prefix_moment_matrices'])
    peak=md['geometry']['peak_P1'];k=peak['bias_index']
    lam,U=eigh(S[k-1],W)
    whole=np.array(md['schemes']['whole_sign']['probes'][-1]['danger_set_minimum']['initial_state'])
    probes={'peak_strong_fullE':np.array(peak['initial_state']),
            'peak_weak_fullE':np.sqrt(E)*U[:,0],
            'whole_danger_weak_x1p000001':whole*1.000001}
    if np.array(md['A'])[0,0]<1:probes['old_rational_first_packet']=np.array([-1.89,.42])
    return probes

def paths(scheme,n):
    if scheme=='whole_sign':
        s=np.array(list(itertools.product([-1.,1.],repeat=L)))
        return s[:,:,None,None]*np.eye(2)[None,None,:,:],dict(method='complete 1024 full-10 whole-sign words',nodes=1024,finite_distribution_exact=True)
    if scheme=='coordinate_sign':
        z=qmc.Sobol(20,scramble=True,seed=SEEDS[scheme]).random_base2(int(np.log2(n)))
        s=np.where(z<.5,-1.,1.).reshape(n,L,2)
        Ms=np.zeros((n,L,2,2));Ms[:,:,0,0]=s[:,:,0];Ms[:,:,1,1]=s[:,:,1]
        return Ms,dict(method='fixed full-20-bit Sobol diagnostic of iid coordinate signs',nodes=n,seed=SEEDS[scheme],finite_distribution_exact=False,integral_error_certified=False)
    if scheme in ['Gaussian','normalized_Gaussian']:
        z=qmc.Sobol(20,scramble=True,seed=SEEDS['Gaussian']).random_base2(int(np.log2(n)))
        b=norm.ppf(np.clip(z,1e-12,1-1e-12)).reshape(n,L,2)
        Ms=np.eye(2)[None,None,:,:]+b[:,:,:,None]*b[:,:,None,:]
        if scheme=='normalized_Gaussian':Ms*=np.sqrt(base.NU)
        return Ms,dict(method='fixed full-20-dimensional Sobol Gaussian integral diagnostic',nodes=n,seed=SEEDS['Gaussian'],endpoint_clip='[1e-12,1-1e-12]',integral_error_certified=False)
    if scheme=='Haar_SO2':
        z=qmc.Sobol(10,scramble=True,seed=SEEDS[scheme]).random_base2(int(np.log2(n)))
        a=2*np.pi*z;Ms=np.empty((n,L,2,2))
        Ms[:,:,0,0]=np.cos(a);Ms[:,:,1,1]=np.cos(a)
        Ms[:,:,0,1]=-np.sin(a);Ms[:,:,1,0]=np.sin(a)
        return Ms,dict(method='fixed full-10-dimensional Sobol iid SO(2) integral diagnostic',nodes=n,seed=SEEDS[scheme],integral_error_certified=False)
    return base.deterministic_word(scheme,L)[None,:,:,:],dict(method='one public deterministic full-10 path',nodes=1,finite_distribution_exact=True)

def evaluate(A,F,G,Vi,Ms,xi):
    N=len(Ms);O=np.array([np.linalg.matrix_power(A,t) for t in range(L)])
    attacks=np.einsum('tij,j->ti',O,xi)
    delta=np.einsum('ntij,tj->nti',Ms,attacks)
    # q_l index l+3, l=-3,...,14; q0 starts at index3.
    q=np.zeros((N,18,2));q[:,3:13]=delta
    q[:,4:14]-=np.einsum('ij,ntj->nti',A,delta)
    assert np.max(abs(q[:,3]-delta[:,0]))<1e-11
    assert np.max(abs(q[:,13]+np.einsum('ij,nj->ni',A,delta[:,9])))<1e-11
    assert np.max(abs(q[:,14:]))==0.
    Ti=np.linalg.inv(base.theta_matrix(A));pk=np.empty((N,T));sp=np.empty((N,T))
    bias=np.zeros((N,2));damage=np.empty((N,T))
    for k in range(T):
        shift=q[:,k:k+4,:].reshape(N,8)
        lam=np.einsum('ni,ij,nj->n',shift,Ti,shift)
        pk[:,k]=ncx2.sf(TAU_PARITY,8,np.maximum(lam,0))
        bias=np.einsum('ij,nj->ni',F,bias)
        if k<L:
            bias+=np.einsum('ij,nj->ni',G,delta[:,k])
            splam=np.einsum('ni,ij,nj->n',delta[:,k],Vi,delta[:,k])
        else:splam=np.zeros(N)
        damage[:,k]=np.einsum('ni,ni->n',bias,bias)
        sp[:,k]=ncx2.sf(TAU_SPATIAL,2,np.maximum(splam,0))
    maxp=np.maximum.accumulate(pk,axis=1);sump=np.minimum(1,np.cumsum(pk,axis=1))
    cdf_lower=np.mean(maxp,axis=0);cdf_upper=np.mean(sump,axis=0)
    # Derive PMF bounds from CDF bounds without introducing independence.
    lowprev=np.r_[0,cdf_lower[:-1]];upprev=np.r_[0,cdf_upper[:-1]]
    pmf_lower=np.maximum(0,cdf_lower-upprev);pmf_upper=np.minimum(1,cdf_upper-lowprev)
    danger=damage>=D;has=danger.any(axis=1);td=np.argmax(danger,axis=1)
    rows=np.arange(N);lower_at=np.maximum(0,1-np.cumsum(pk,axis=1)[rows,td])
    upper_at=1-maxp[rows,td]
    risklower=float(np.mean(np.where(has,lower_at,0.)))
    riskupper=float(np.mean(np.where(has,upper_at,0.)))
    spatial_survival=np.cumprod(1-sp,axis=1)
    spatial_cdf=1-spatial_survival
    spatialrisk=float(np.mean(np.where(has,spatial_survival[rows,td],0.)))
    assert np.max(abs(cdf_lower[0]-cdf_upper[0]))<1e-13
    assert np.min(cdf_upper-cdf_lower)>-1e-12
    assert -1e-12<=risklower<=riskupper+1e-12<=1+1e-12
    if np.all(has&(td==0)):
        assert abs(risklower-(1-pk[:,0].mean()))<1e-12
        assert abs(riskupper-risklower)<1e-12
    firsthist=[float(np.mean(has&(td==k))) for k in range(T)]
    return dict(parity_single_test_marginal_power=np.mean(pk,axis=0).tolist(),
        parity_first_alarm_CDF_lower=cdf_lower.tolist(),parity_first_alarm_CDF_upper=cdf_upper.tolist(),
        parity_first_alarm_PMF_lower=pmf_lower.tolist(),parity_first_alarm_PMF_upper=pmf_upper.tolist(),
        parity_danger_before_alarm_lower=risklower,parity_danger_before_alarm_upper=riskupper,
        spatial_current_single_test_marginal_power=np.mean(sp,axis=0).tolist(),
        spatial_current_first_alarm_CDF=np.mean(spatial_cdf,axis=0).tolist(),
        spatial_current_first_alarm_PMF=np.mean(np.c_[np.zeros(N),spatial_cdf][:,1:]-np.c_[np.zeros(N),spatial_cdf][:,:-1],axis=0).tolist(),
        spatial_current_danger_before_alarm=spatialrisk,
        dangerous_path_probability=float(np.mean(has)),first_danger_packet_probability=firsthist,
        no_danger_probability=float(np.mean(~has)),
        expected_bias_squared_by_state=damage.mean(axis=0).tolist(),
        expected_peak_bias_squared=float(damage.max(axis=1).mean()),
        max_expected_bias_squared=float(damage.mean(axis=0).max()),
        expected_terminal_bias_squared=float(damage[:,9].mean()),
        spatial_probability_integrated_after_conditional_product=True,
        parity_no_independence_assumption=True,
        first_danger_threshold_tolerance=0.,
        internal_invariants='startup/stop shift, vanished post-stop difference, CDF bounds/order, first-packet danger identity checked',
        bias_state_index_equals_packet_index_plus1=True)

def run():
    out=dict(date='2026-11-14',danger_comparison_input_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        model_sha256=d['model_sha256'],parity_df=8,spatial_current_df=2,
        alpha_per_test=ALPHA,tests=T,packets=list(range(T)),bias_states=list(range(1,T+1)),
        total_event_FA_union_upper=.05,spatial_healthy_event_FA_exact=1-(1-ALPHA)**T,
        fixed_candidate_start=True,unlimited_unknown_start_monitoring=False,
        attack_time_range=[0,9],attack_length=L,energy_E=E,danger_D=D,
        probe_selection='three fixed no-watermark/whole-sign probes, plus prior rational As probe; identical across schemes',
        no_new_attack_or_watermark_search=True,physical_noise_samples=0,
        parity_bounds='conditional full-path max/sum single-test probabilities, then average; no product across overlapping windows',
        spatial_formula='current-only independent spatial noise, conditional full-path product, then average',
        numerical_CDF_interval_certification=False,QMC_integral_error_certification=False,
        not_full_attack_family_minimax=True,seeds=SEEDS,models={})
    for name,md in d['models'].items():
        A=np.array(md['A']);F=np.array(md['F']);G=np.array(md['G']);W=np.array(md['W'])
        J=np.linalg.inv(np.diag([1.1,1.2]));V=.5*np.eye(2)+.5*J@J.T;Vi=np.linalg.inv(V)
        probes=fixed_probes(md)
        model=dict(A=A.tolist(),F=F.tolist(),G=G.tolist(),spatial_covariance=V.tolist(),probes={},schemes={})
        out['models'][name]=model
        for label,xi in probes.items():
            energies=float(xi@W@xi);assert energies<=E*(1+1e-10)
            potential=np.einsum('i,kij,j->k',xi,np.array(md['no_watermark_prefix_moment_matrices']),xi)
            model['probes'][label]=dict(initial_state=xi.tolist(),encoded_energy=energies,
                no_watermark_potential_peak=float(potential.max()),no_watermark_peak_bias_index=int(potential.argmax())+1,
                no_watermark_terminal=float(potential[9]))
        for scheme in base.SCHEMES:
            levels=[1024,4096] if scheme in ['coordinate_sign','Gaussian','normalized_Gaussian','Haar_SO2'] else [0]
            samples=[]
            for n in levels:
                Ms,meta=paths(scheme,n);results={}
                for label,xi in probes.items():results[label]=evaluate(A,F,G,Vi,Ms,xi)
                samples.append(dict(integration=meta,probes=results))
                print(name,scheme,meta['nodes'],{p:(round(r['parity_danger_before_alarm_lower'],5),round(r['parity_danger_before_alarm_upper'],5),round(r['spatial_current_danger_before_alarm'],5)) for p,r in results.items()},flush=True)
            model['schemes'][scheme]=samples
            (H/'first_alarm_bounds.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (H/'first_alarm_bounds.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':run()
