"""Population bounds for all OL/H/C; no independent-window approximation."""
from pathlib import Path
import sys,json,hashlib,math
sys.dont_write_bytecode=True
import sympy as s
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
try:from flint import arb,ctx
except ImportError:
    sys.path.insert(0,str(H/'tmp/validated_math'));from flint import arb,ctx
ctx.prec=110
from gain_models import ref
CFG=json.loads((H/'registration.json').read_text('utf8'))
def ba(x):
    q=s.Rational(x);return arb(int(q.p))/arb(int(q.q))
def sm(x):return s.Matrix([[s.Rational(str(v)) for v in row] for row in x])
def enclosure(x):
    den=10**12;lo=s.Rational(math.floor(float(x.lower())*den),den);hi=s.Rational(math.ceil(float(x.upper())*den),den)
    while not x>ba(lo):lo-=s.Rational(1,den)
    while not ba(hi)>x:hi+=s.Rational(1,den)
    return dict(lower=str(lo),upper=str(hi),lower_display=float(lo),upper_display=float(hi),verified_outward=True)
def central_tail(df,tau):
    x=ba(tau)/2;term=arb(1);v=term
    for j in range(1,df//2):term*=x/j;v+=term
    return (-x).exp()*v
def nc_tail_upper(df,tau,lam,N=1024):
    x=ba(tau)/2;a=ba(lam)/2
    term=arb(1);S=term
    for j in range(1,df//2):term*=x/j;S+=term
    ex=(-x).exp();S*=ex;term*=ex
    p=(-a).exp();mass=p;total=p*S
    for j in range(N):
        term*=x/(df//2+j);S+=term
        p*=a/(j+1);mass+=p;total+=p*S
    upper=total+(1-mass)
    return min(arb(1),upper.upper())
def residual_info(m,kind):
    A=sm(m.A);Qs=sm(m.Q);Hs=[sm(m.H[j]) for j in (1,0,2)]
    ids=[1] if kind=='temporal' else [0,1,2]
    rows=[(t,j,c) for t in range(5) for j in ids for c in range(2)]
    O=s.Matrix.vstack(*[Hs[j]*A**t for t in range(5) for j in ids])
    V=s.zeros(len(rows))
    for i,(t,j,c) in enumerate(rows):
        for l,(u,k,d) in enumerate(rows):
            val=0
            for r in range(min(t,u)):
                val+=(Hs[j]*A**(t-1-r)*Qs*(A**(u-1-r)).T*Hs[k].T)[c,d]
            if t==u and j==k:val+=s.Rational(1,2) if c==d else 0
            V[i,l]=val
    if kind=='temporal':
        Z=s.zeros(8,10)
        for t in range(4):Z[2*t:2*t+2,2*t:2*t+2]=-A;Z[2*t:2*t+2,2*t+2:2*t+4]=s.eye(2)
        assert Z*O==s.zeros(8,2)
        P0=Z.T*(Z*V*Z.T).inv()*Z;D=P0[-2:,-2:]
    else:
        Vi=V.inv();P0=Vi-Vi*O*(O.T*Vi*O).inv()*O.T*Vi
        select=[t*6+2+c for t in range(5) for c in (0,1)]
        J=P0.extract(select,select);D=J[-2:,-2:]
    bound=s.Rational(math.ceil(np.linalg.eigvalsh(np.array(D,float))[-1]*10**9),10**9)
    M=bound*s.eye(2)-D
    assert M[0,0]>0 and M.det()>0
    return D,bound
def pulse_lower(g,e,df,tau,rho):
    m=s.Rational(1,10)/(s.Rational(e)*g*g)-s.Rational(1,4)
    if m<=0:start=s.Rational(0);angular=s.Rational(1)
    else:
        start=s.Rational(math.ceil((np.sqrt(1+8*float(m))-1)/2*1000),1000)
        assert (start+start*start)/2>=m
        angular=s.Rational(1,2)
    pfa=central_tail(df,tau);total=arb(0);bins=64
    for i in range(bins):
        lo=start+s.Rational(4*i,bins);hi=start+s.Rational(4*(i+1),bins)
        lam=s.Rational(e)*rho*(s.Rational(1,2)+hi)**2
        pd=nc_tail_upper(df,tau,lam)
        miss=1-pd-3*pfa
        if miss>0:total+=ba(angular)*((-ba(lo)).exp()-(-ba(hi)).exp())*miss
    return total,dict(radial_start_exact=str(start),radial_stop_exact=str(start+4),angular_fraction=str(angular),bins=bins)
def main():
    cou=json.loads((H/'coupling_certificates.json').read_text('utf8'))['models']
    out={};detcheck={}
    for lab,df in [('temporal',8),('joint_GLRT',10)]:
        tau=CFG['conservative_thresholds'][lab];fa=central_tail(df,tau)
        assert 28*fa<ba(s.Rational(1,20))
        detcheck[lab]=dict(df=df,threshold=tau,single_FA=enclosure(fa),total_FA_upper=enclosure(28*fa),
          unavoidable_pre_alarm_risk_upper=enclosure(1-fa))
    for name,rows in cou.items():
        m=ref.load_model(name);infos={lab:residual_info(m,lab) for lab in ('temporal','joint_GLRT')};r={}
        for gain,z in rows.items():
            rr={}
            for lab,df in [('temporal',8),('joint_GLRT',10)]:
                D,rho=infos[lab];tau=CFG['conservative_thresholds'][lab];g=s.Rational(z['G7_sigma_min_lower_exact'])
                full,meta=pulse_lower(g,10,df,tau,rho)
                small,smallmeta=pulse_lower(g,s.Rational(1,25),df,tau,rho)
                rr[lab]=dict(full_budget_OL_witness_lower=enclosure(full),full_witness=meta,
                  beta25_feasible_OL_witness_lower=enclosure(small),low_energy_witness=smallmeta,
                  every_OL_H_C_class_upper=detcheck[lab]['unavoidable_pre_alarm_risk_upper'],
                  last_packet_noncentral_matrix=[[str(v) for v in row] for row in D.tolist()],
                  last_packet_spectral_upper_exact=str(rho),
                  classes_share_witness=True,no_strict_information_risk_order_proved=True)
            r[gain]=rr
        out[name]=r
        print(json.dumps(dict(model=name,population_gain_bounds=len(r))),flush=True)
    beta=ba(s.Rational(1,20))+(ba(s.Rational(7,8))*ba(s.Rational(1,25))).sqrt()
    assert beta<ba(s.Rational(1,4))
    result=dict(date='2026-11-17',models=out,detectors=detcheck,
      attacked_alarm_bound_for_OL_and_strict_H_energy_1_over_25=enclosure(beta),
      C_general_energy_does_not_certify_stealth=True,full_class_suprema_not_computed=True,
      overlaps_not_assumed_independent=True,Monte_Carlo=False)
    (H/'population_risk_bounds.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
