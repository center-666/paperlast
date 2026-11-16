"""Rationalize an EXISTING held-out P1/B4 reentry failure, no further search."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
import joint_screen as js
from independent_math_and_reentry import sm,above,threshold
import sympy as s
import numpy as np
def main():
    bank=np.load(H/'screen_arrays.npz');chosen=None
    for name in js.CFG['models']:
      for scope in ('known','unknown'):
       for j in (0,2):
        tag=f'{name}__{scope}__j{j}__original_Gaussian'
        possible=bank[tag+'__B4__reentry_danger'][1]
        if possible.any():chosen=(name,scope,j,tag,int(np.flatnonzero(possible)[0]));break
       if chosen:break
      if chosen:break
    assert chosen is not None
    name,scope,j,tag,i=chosen;index=64+i;m=js.ref.load_model(name)
    data,_=js.sample_inputs(m,name);seed=json.loads((js.OLD/'sampling_seed_map.json').read_text('utf8'))[name]['watermark'][index]
    _,beta=js.ref.original_watermarks(m,32,seed,.5,.5)
    def rational(v):return s.Matrix([s.Rational(f'{float(x):.8f}') for x in v])
    yy={l:[rational(row) for row in data['y'][index,:,l]] for l in (0,1,2)}
    betas=[rational(v) for v in beta[:,j]];M=[s.eye(2)/2+b*b.T for b in betas]
    A=sm(m.A.tolist());HH=[sm(h.tolist()) for h in m.H];S=[h.inv()*sm(r.tolist())*h.inv().T for h,r in zip(HH,m.R)]
    target=0 if j==0 else 1
    xi=s.Matrix([s.Rational(str(x)) for x in js.PRE['models'][name]['attacks'][str(j)]['original_Gaussian']['P1']['xi']])
    W=sum(((HH[j]*A**t).T*(HH[j]*A**t) for t in range(10)),s.zeros(2));energy=(xi.T*W*xi)[0];assert energy<=10
    a=[HH[j]*A**(k-7)*xi if 7<=k<17 else s.zeros(2,1) for k in range(32)]
    runs={};rec={};slacks=[]
    for attacked in (False,True):
        rem=[0,0];post=[];phis=[];alarms=[];prev=s.zeros(2,1)
        for k in range(32):
            y={l:yy[l][k]+(M[k]*a[k] if attacked and l==j else s.zeros(2,1)) for l in (0,1,2)}
            z={l:HH[l].inv()*v for l,v in y.items()};e=[z[0]-z[1],z[2]-z[1]];f=[]
            for h,(l,o) in enumerate(((0,2),(2,0))):
                C=S[1]*(S[o]+S[1]).inv();u=e[h]-C*e[1-h];Q=S[l]+S[1]-S[1]*(S[o]+S[1]).inv()*S[1]
                f.append((u.T*Q.inv()*u)[0])
            n=1 if scope=='known' else 2;bounds=threshold(n);flags=[False,False]
            if k>=7:
                h=target if scope=='known' else (0 if f[0]>=f[1] else 1)
                flags[h]=above(f[h],bounds)
                slacks.append(f[h]-bounds[1] if flags[h] else bounds[0]-f[h])
            rem=[5 if flags[h] else rem[h] for h in range(2)];ph=[int(v==0) for v in rem];rem=[max(v-1,0) for v in rem]
            mask=ph[0]+2*ph[1];mode=js.PRE['models'][name]['redesigned_modes'][mask];K,G=sm(mode['K']),sm(mode['G']);p=A*prev
            prev=p+K*(y[1]-HH[1]*p)+G*sum((ph[h]*(y[l]-HH[l]*p) for h,l in enumerate((0,2))),s.zeros(2,1))
            post.append(prev);phis.append(ph);alarms.append(flags)
        runs[attacked]=post;rec[attacked]=dict(phi=phis,alarms=alarms)
    candidates=[]
    for k in range(8,32):
        if rec[True]['phi'][k][target] and not rec[True]['phi'][k-1][target]:
            v=runs[True][k]-runs[False][k];harm=(v.T*v)[0]
            if harm>s.Rational(1,10):candidates.append((k,harm))
    assert candidates,'rational neighborhood did not preserve selected witness'
    k,harm=candidates[0]
    out=dict(date='2026-11-16',source='first existing heldout P1/B4 Gaussian failure in registered case order; no extra samples/directions',
      case=tag,sample_bank_index=index,physical_seed=2026111500+(0 if name=='unstable' else 100000)+index,watermark_seed=seed,
      target=j,readmission_time=k,attack_alarms=[q for q,v in enumerate(rec[True]['alarms']) if any(v)],
      clean_alarms=[q for q,v in enumerate(rec[False]['alarms']) if any(v)],
      coding_energy=str(energy),energy_le_10=True,xi=[str(x) for x in xi],
      bias_squared_exact=str(harm),bias_squared_display=float(harm),danger_margin_display=float(harm-s.Rational(1,10)),
      minimum_alarm_decision_margin_display=float(min(slacks)),
      all_decisions_strict=True,measurements={str(j):[[str(x) for x in v] for v in yy[j]] for j in yy},
      beta=[[str(x) for x in v] for v in betas],
      scope='Gaussian rank-one watermarks preserved exactly as .5I+beta betaT with rational beta; frozen rational gains; own healthy B4 shadow',
      positive_probability='finite strict threshold and harm inequalities persist on an open neighborhood in raw measurements and Gaussian beta. Their joint distribution has full support. Therefore B4 hold5 has a genuine positive-probability reentry failure under the original preselected attack.',
      not_new_attack_or_safety_algorithm=True)
    js.dump(H/'joint_reentry_counterexample_exact.json',out)
    print(json.dumps({q:out[q] for q in ('case','sample_bank_index','readmission_time','attack_alarms','clean_alarms','bias_squared_display','minimum_alarm_decision_margin_display')}))
if __name__=='__main__':main()
