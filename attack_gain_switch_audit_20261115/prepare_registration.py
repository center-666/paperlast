"""Register a small, fixed screening protocol before generating any samples."""
from pathlib import Path
import sys,json,hashlib,datetime
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;P=H.parent;ROOT=P.parent.parent
REF=P/'common_g_reference_revision_20261115'
sys.path.insert(0,str(REF))
import common_g_reference as ref
import numpy as np
import sympy as sp
from scipy.linalg import block_diag,solve_discrete_are,eigh
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8')
def compute_modes(m):
    i,j,other=1,0,2
    C=np.vstack([m.H[i],m.H[other]]);R=block_diag(m.R[i],m.R[other])
    Pp=solve_discrete_are(m.A.T,C.T,m.Q,R)
    L=np.linalg.solve(C@Pp@C.T+R,C@Pp).T;K0,G0=L[:,:2],L[:,2:]
    out={}
    for b in ('B0','B1','B2','B3'):
        modes=[]
        for phi in (0,1):
            effective=1 if b=='B0' else phi
            K,G=(K0,G0) if b=='B2' and phi==0 else (m.K[i],m.G[i])
            J=np.eye(2)-K@m.H[i]-G@(m.H[other]+effective*m.H[j]);F=J@m.A
            rat=sp.Matrix([[sp.Rational(str(v)) for v in row] for row in F])
            Q=sp.eye(2)/4-rat.T*rat
            assert Q[0,0]>0 and Q.det()>0
            modes.append({'phi':phi,'effective_phi':effective,'K':K.tolist(),'G':G.tolist(),'J':J.tolist(),'F':F.tolist(),
                'spectral_radius_float':float(max(abs(np.linalg.eigvals(F)))),'norm2_float':float(np.linalg.norm(F,2)),
                'common_I_bound_r':.5,'rational_minors_for_quarter_I_minus_FtF':[str(Q[0,0]),str(Q.det())]})
        out[b]=modes
    return out,{'C_reject':C.tolist(),'R_reject':R.tolist(),'prior_reject':Pp.tolist(),'nominal_held_mode_MMSE_only':True,
        'DARE_residual_max':float(np.max(np.abs(Pp-(m.A@(Pp-L@C@Pp)@m.A.T+m.Q))))}
def attacks(m,encoding,L=10,E=10.):
    O=[m.H[0]@np.linalg.matrix_power(m.A,t) for t in range(L)]
    W=sum(v.T@v for v in O);F=m.F(1);G=m.G[1]
    u=np.array([-1.89,.42]);ordinary=u*np.sqrt(E/(u@W@u))
    rows=[]
    mu,q=(1.,0.) if encoding=='identity' else (1.,.5)
    for h in range(L):
        Cs=[np.linalg.matrix_power(F,h-t)@G for t in range(h+1)]
        B=sum(C@O[t] for t,C in enumerate(Cs));S=mu**2*B.T@B
        for t,C in enumerate(Cs):S+=q*q*O[t].T@(C.T@C+np.trace(C.T@C)*np.eye(2))@O[t]
        ev,V=eigh(S,W);v=V[:,-1]
        if v[0]<0:v=-v
        xi=v*np.sqrt(E/(v@W@v))
        rows.append({'relative_post_index':h,'max_expected_bias_squared':float(E*ev[-1]),'xi':xi.tolist(),'S':S.tolist()})
    best=max(rows,key=lambda z:z['max_expected_bias_squared'])
    return {'ordinary':{'xi':ordinary.tolist(),'actual_energy':float(ordinary@W@ordinary),'source_direction':'existing first-packet witness (-1.89,.42), normalized to the SAME E=10'},
      'P1':{'xi':best['xi'],'actual_energy':float(np.array(best['xi'])@W@np.array(best['xi'])),'worst_prefix_post_index':best['relative_post_index'],
            'max_expected_bias_squared_all_accept':best['max_expected_bias_squared'],'not_a_switched_closed_loop_worst_attack':True},
      'W':W.tolist(),'all_prefix_P1_checks':rows,'P3_status':'existing definition retained; no new closed-loop P3 optimizer or minimax claim'}
def main():
    assert not (H/'registration.json').exists(),'registration is immutable; do not retune after viewing samples'
    accepted=json.loads((P/'minimum_contribution_gate_20261115/freeze_manifest.json').read_text('utf8'))
    for n in ('common_g_reference.py','inputs/five_node_aggregate_model.json'):
        assert sha(REF/n)==accepted['local_reference_artifact_hashes'][n],n
    frozen={}
    for p in P.rglob('*'):
        if p.is_file() and H not in p.parents and not any(t in {'tmp','__pycache__','literature_work','_build','publication'} for t in p.relative_to(P).parts):frozen[str(p)]=sha(p)
    for p in (ROOT/'AGENTS.md',ROOT/'参考文献/文献库.json',ROOT/'研究规划/第一篇小论文/secure distributed safety.tex'):frozen[str(p)]=sha(p)
    dump(H/'protected_inputs.json',{'date':'2026-11-15','sha256':frozen})
    config={'baseline':'0ea9c7db47f3e1642bbe5cba9ec6f914e4c6981a','T':32,'attack_start':7,'attack_length':10,
      'monitor_start':7,'monitor_end_inclusive':31,'monitor_scope':'one predeclared communication link 0->1; public commissioning time7; not unknown-onset/network-wide validation',
      'energy_budget_coding_domain':10.,'danger_threshold_posterior':.1,'parity_order':4,'event_FA_budget_per_detector':.05,
      'detectors':['single-source temporal parity','raw spatial y0_decoded-H0 inv(H1) y1'],
      'alarm_used_for_gating':'raw spatial only; no OR combination','per_detector_tests':25,'threshold_rule':'Bonferroni alpha/25, nominal chi-square',
      'rejection_hold_updates_including_alarm':5,'new_alarm_restarts_hold':True,'raw_packets_still_monitored_while_rejected':True,
      'samples_per_model':128,'unique_physical_paths_total':256,'physical_seed_base':2026111500,'watermark_seed_base':2026112500,
      'models':['unstable','stable'],'encodings':['identity','original_Gaussian'],'lambda':.5,'q':.5,
      'attack_profiles':['ordinary','P1'],'ordinary_direction':[-1.89,.42],
      'P1_objective':'max over attack prefixes of unconditional expected posterior bias squared in all-accept original filter, under the SAME true E=10',
      'B0':'frozen full common-G, all accept; alarm diagnostic recorded but does not gate',
      'B1':'spatial gate, frozen original K/G, hold5 then readmit',
      'B2':'same spatial gate; frozen original full-mode gain; rejection-mode nominal steady aggregate MMSE gain from remaining local+neighbor2; not conditional switching MMSE',
      'B3':'standard neighbor prediction replacement with original K/G; prove identical to B1',
      'primary_risk':'P(tau_D<tau_A): posterior difference vs own healthy CLOSED-LOOP shadow; tau_A first spatial alarm in monitored event; same-packet detection occurs before fusion',
      'secondary_risk':'same event, difference vs unprotected all-accept healthy nominal; useful first-alarm equality diagnostic, not same as primary after false alarms',
      'normal_metrics':['target mean total state MSE monitored7..31','network mean total state MSE','event FA','rejected fraction'],
      'attack_metrics':['event Pd for each detector','first alarm delay incl onset windows','first-packet danger','primary and secondary danger-before-alarm','cumulative paired bias squared','stop residual','readmission transient'],
      'paired_gain_comparison':'B2 on saved B1 alarm/gate history first; independently compute complete B2 history next and test equality because raw detector is estimator-independent',
      'statistics':'finite 128-sample diagnostics; Wilson95 for proportions, paired sample SE for loss differences; no all-attack worst-risk certification',
      'stop_conditions':['B3 algebraically equals B1: no new mechanism claim','modewise gain recomputation already standard: performance gain alone not originality','no detector/hold/gain/seed retuning after screen'],
      'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sampling_has_started':False}
    dump(H/'registration.json',config)
    modelrows={}
    for name in config['models']:
        m=ref.load_model(name);modes,info=compute_modes(m)
        modelrows[name]={'mode_gains':modes,'rejection_design':info,'attacks':{e:attacks(m,e) for e in config['encodings']}}
    dump(H/'analytic_preflight.json',{'registration_sha256':sha(H/'registration.json'),'models':modelrows,
        'reference_hashes':{n:sha(REF/n) for n in ('common_g_reference.py','inputs/five_node_aggregate_model.json')},
        'new_physical_samples':0,'gain_bound_certificate_scope':'computed F decimals treated as exact rational inputs, not exact DARE/engineering uncertainty certificate'})
    print(json.dumps({'registered':True,'frozen_old_inputs':len(frozen),'screen_samples_per_model':128,
      'norms':{name:{b:[round(v['norm2_float'],6) for v in z['mode_gains'][b]] for b in z['mode_gains']} for name,z in modelrows.items()},
      'P1':{name:{e:z['attacks'][e]['P1'] for e in config['encodings']} for name,z in modelrows.items()}},ensure_ascii=False))
if __name__=='__main__':main()
