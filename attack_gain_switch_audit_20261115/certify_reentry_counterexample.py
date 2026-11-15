"""Certify one negative witness extracted from the ALREADY registered screen.

No new random path, attack direction, threshold, gain, or hold parameter.
Rationalize one saved measurement history and prove all decisions strictly.
"""
from pathlib import Path
import json,hashlib
import sympy as s
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
def sm(x):return s.Matrix([[s.Rational(str(v)) for v in row] for row in x])
def log_bounds(z,N):
    lo=2*sum(z**(2*k+1)/s.Rational(2*k+1) for k in range(N))
    rem=2*z**(2*N+1)/((2*N+1)*(1-z*z))
    return lo,lo+rem
def main():
    pre=json.loads((H/'analytic_preflight.json').read_text('utf8'));cfg=json.loads((H/'registration.json').read_text('utf8'))
    data=np.load(H/'screen_arrays.npz');name='unstable';tag=name+'__identity__ordinary'
    phi=data[tag+'__B1__phi'];re=np.zeros_like(phi,bool);re[:,1:]=(phi[:,1:]==1)&(phi[:,:-1]==0);re[:,:7]=False;re[:,17:]=False
    d1=np.sum((data[tag+'__B1__post']-data[name+'__normal__B1__post'])**2,axis=2)
    d2=np.sum((data[tag+'__B2__post']-data[name+'__normal__B2__post'])**2,axis=2)
    found=np.argwhere(re&(d2>=.1)&(d1<.1));assert len(found)>0
    path,k0=map(int,found[0]);assert (path,k0)==(33,15)
    source=json.loads((P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json').read_text('utf8'))
    A=sm(source['models'][name]['A']);HH=[sm(x) for x in source['H']];D=HH[0]*HH[1].inv()
    V=sm(source['R'][0])+D*sm(source['R'][1])*D.T
    xi=s.Matrix([s.Rational(str(v)) for v in pre['models'][name]['attacks']['identity']['ordinary']['xi']])
    W=sum(((HH[0]*A**t).T*(HH[0]*A**t) for t in range(10)),s.zeros(2));energy=(xi.T*W*xi)[0];assert energy<=10
    # tau=chi2_2.isf(.05/25)=2 log500; log500=8log2+3log(5/4).
    l2,u2=log_bounds(s.Rational(1,3),15);lq,uq=log_bounds(s.Rational(1,9),9)
    low,high=2*(8*l2+3*lq),2*(8*u2+3*uq)
    assert high-low<s.Rational(1,10**12)
    Y=data[name+'__physical__y'][path]
    y={j:[s.Matrix([s.Rational(f'{float(x):.8f}') for x in Y[k,j]]) for k in range(32)] for j in (0,1,2)}
    attack=[HH[0]*A**(k-7)*xi if 7<=k<17 else s.zeros(2,1) for k in range(32)]
    rec={}
    for isattack in (False,True):
        alarms=[];gates=[];count=0;score_slack=[]
        for k in range(32):
            r=y[0][k]+(attack[k] if isattack else s.zeros(2,1))-D*y[1][k];score=(r.T*V.inv()*r)[0]
            if k<7:alarm=False
            else:
                assert score<low or score>high
                alarm=bool(score>high);score_slack.append(low-score if not alarm else score-high)
            if alarm:count=5
            gates.append(int(count==0));alarms.append(alarm);count=max(count-1,0)
        runs={}
        for b in ('B1','B2'):
            post=[];previous=s.zeros(2,1)
            for k in range(32):
                mode=pre['models'][name]['mode_gains'][b][gates[k]];K,G=sm(mode['K']),sm(mode['G']);p=A*previous
                innovations=y[2][k]-HH[2]*p+gates[k]*(y[0][k]+(attack[k] if isattack else s.zeros(2,1))-HH[0]*p)
                value=p+K*(y[1][k]-HH[1]*p)+G*innovations;post.append(value);previous=value
            runs[b]=post
        rec[isattack]={'runs':runs,'gates':gates,'alarms':alarms,'min_score_slack':min(score_slack)}
    assert rec[True]['gates'][k0-1]==0 and rec[True]['gates'][k0]==1
    harms={}
    for b in ('B1','B2'):
        bias=rec[True]['runs'][b][k0]-rec[False]['runs'][b][k0];harms[b]=(bias.T*bias)[0]
    assert harms['B1']<s.Rational(1,10)<harms['B2']
    assert float(harms['B1'])<.089 and float(harms['B2'])>.127
    out={'date':'2026-11-15','source':'first existing registered screen witness in declared case/path/time order; no new sampling',
      'case':tag,'sample_index':path,'physical_seed':cfg['physical_seed_base']+path,'readmission_time':k0,
      'attack_support':[7,16],'energy_exact':str(energy),'energy_le_10_exact':True,'xi':[str(x) for x in xi],
      'threshold_exact':'2 log500','threshold_rational_bounds':[str(low),str(high)],
      'clean_alarms':[i for i,v in enumerate(rec[False]['alarms']) if v],
      'attack_alarms':[i for i,v in enumerate(rec[True]['alarms']) if v],
      'all_gate_decisions_strict_exact':True,
      'clean_min_score_slack':str(rec[False]['min_score_slack']),'attack_min_score_slack':str(rec[True]['min_score_slack']),
      'bias_squared_at_readmission_exact':{b:str(v) for b,v in harms.items()},
      'bias_squared_at_readmission_display':{b:float(v) for b,v in harms.items()},
      'measurement_history_rational':{str(j):[[str(v) for v in row] for row in vals] for j,vals in y.items()},
      'positive_probability_argument':'all finite decisions and two danger comparisons have strict margins; by continuity a nonempty open measurement neighborhood shares them; R_i positive definite gives full support of the finite Gaussian raw-measurement history',
      'conclusion':'B2 is not a pointwise dominance or readmission-safety guarantee over B1; this is AFTER an alarm, not a counterexample to pre-first-alarm equality',
      'new_random_samples':0,'new_attack_or_defense_parameters':0,'candidate_innovation_claim':False,
      'gain_scope':'frozen numerical gains treated as exact rational input, not an interval certificate for the true DARE solution'}
    (H/'reentry_counterexample_exact.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({k:out[k] for k in ('sample_index','readmission_time','clean_alarms','attack_alarms','bias_squared_at_readmission_display','all_gate_decisions_strict_exact')},ensure_ascii=False))
if __name__=='__main__':main()
