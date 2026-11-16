"""Independent rational Gaussian identities, mode bounds, and old witness replay.

Does not call joint_screen scores, decisions, update, or old certificate scripts.
Finite decimals of frozen gains are rational inputs, not interval DARE solutions.
"""
from pathlib import Path
import sys,json,hashlib,itertools
sys.dont_write_bytecode=True
import sympy as s
H=Path(__file__).resolve().parent;P=H.parent;OLD=P/'attack_gain_switch_audit_20261115'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf8')
def sm(x):return s.Matrix([[s.Rational(str(v)) for v in row] for row in x])
def logs(z,N=35):
    lo=2*sum(z**(2*k+1)/s.Rational(2*k+1) for k in range(N))
    return lo,lo+2*z**(2*N+1)/((2*N+1)*(1-z*z))
def threshold(n):
    # chi2_2 upper tail .05/(25*n): 2log(500*n).
    l2,u2=logs(s.Rational(1,3));lq,uq=logs(s.Rational(1,9));ln,un=logs(s.Rational(n-1,n+1))
    return 2*(8*l2+3*lq+ln),2*(8*u2+3*uq+un)
def above(v,bounds):
    lo,hi=bounds
    assert v<lo or v>hi,'undecided rational threshold'
    return bool(v>hi)
def exact_geometry(source):
    HH=[sm(v) for v in source['H']];S=[h.inv()*sm(r)*h.inv().T for h,r in zip(HH,source['R'])]
    V=(S[0]+S[1]).row_join(S[1]).col_join(S[1].row_join(S[2]+S[1]));Vi=V.inv();rows={}
    for idx,(j,o) in enumerate(((0,2),(2,0))):
        C=S[1]*(S[o]+S[1]).inv();Gamma=S[j]+S[1]-S[1]*(S[o]+S[1]).inv()*S[1]
        L=s.eye(2).row_join(-C) if idx==0 else (-C).row_join(s.eye(2))
        E=s.eye(2).col_join(s.zeros(2)) if idx==0 else s.zeros(2).col_join(s.eye(2))
        assert L*V*L.T==Gamma
        assert L*V*(s.zeros(2).row_join(s.eye(2)) if idx==0 else s.eye(2).row_join(s.zeros(2))).T==s.zeros(2)
        assert E.T*Vi*E==Gamma.inv()
        assert Vi*E*(E.T*Vi*E).inv()*E.T*Vi==L.T*Gamma.inv()*L
        healthycov=(S[1].inv()+S[o].inv()).inv()
        assert Gamma==S[j]+healthycov
        gap=S[j]+S[1]-Gamma;assert gap[0,0]>0 and gap.det()>0
        rows[str(j)]=dict(conditional_covariance=str(Gamma),single_covariance=str(S[j]+S[1]),
          positive_information_gain=True,GLRT_equals_leave_one=True,noise_variance_ratio=[float(Gamma[k,k]/(S[j]+S[1])[k,k]) for k in range(2)])
    # Profiled common-state Gaussian likelihood equals the four-dimensional residual.
    C=s.eye(2).col_join(s.eye(2)).col_join(s.eye(2));R=s.diag(S[0],S[1],S[2])
    L=s.eye(2).row_join(-s.eye(2)).row_join(s.zeros(2)).col_join(s.zeros(2).row_join(-s.eye(2)).row_join(s.eye(2)))
    profile=R.inv()-R.inv()*C*(C.T*R.inv()*C).inv()*C.T*R.inv()
    assert profile==L.T*Vi*L
    return HH,S,V,rows
def main():
    pre=json.loads((H/'analytic_preflight.json').read_text('utf8'));source=json.loads((P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json').read_text('utf8'))
    HH,S,V,geo=exact_geometry(source)
    cert=[];ranks={}
    for name in ('unstable','stable'):
        A=sm(source['models'][name]['A'])
        for typ in ('fixed_modes','redesigned_modes'):
            for mode in pre['models'][name][typ]:
                K,G=sm(mode['K']),sm(mode['G']);F=(s.eye(2)-K*HH[1]-G*sum((HH[j] for j in mode['accepted']),s.zeros(2)))*A
                Q=s.Rational(81,100)*s.eye(2)-F.T*F;assert Q[0,0]>0 and Q.det()>0
                cert.append(dict(model=name,type=typ,mask=mode['mask'],F_exact=str(F),minors=[str(Q[0,0]),str(Q.det())]))
        O0=HH[0].col_join(HH[0]*A);O2=HH[2].col_join(HH[2]*A)
        ranks[name]=dict(single_time_aggregate_ambiguity_rank=HH[0].row_join(-HH[2]).rank(),
          two_time_autonomous_aggregate_rank=O0.row_join(-O2).rank(),
          commutator_det=str((A*HH[2]-HH[2]*A).det()))
        assert ranks[name]['two_time_autonomous_aggregate_rank']==4
    old=json.loads((OLD/'reentry_counterexample_exact.json').read_text('utf8'))
    A=sm(source['models']['unstable']['A']);xi=s.Matrix([s.Rational(v) for v in old['xi']])
    y={int(j):[s.Matrix([s.Rational(v) for v in row]) for row in rows] for j,rows in old['measurement_history_rational'].items()}
    a=[HH[0]*A**(k-7)*xi if 7<=k<17 else s.zeros(2,1) for k in range(32)]
    W=sum(((HH[0]*A**t).T*(HH[0]*A**t) for t in range(10)),s.zeros(2));assert (xi.T*W*xi)[0]<=10
    l2,u2=logs(s.Rational(1,3));lq,uq=logs(s.Rational(1,9));health=(2*(4*l2+lq),2*(4*u2+uq))
    modes=pre['models']['unstable'];runs={};records={}
    for scope in ('known','unknown'):
      for b in ('B0','B1','B2','B4','B5','B4H'):
        for attacked in (False,True):
            posts=[];phis=[];alarmhistory=[];remaining=[0,0];active=[False,False];passes=[0,0];prev=s.zeros(2,1)
            scores_saved=[]
            for k in range(32):
                z={j:HH[j].inv()*(y[j][k]+(a[k] if attacked and j==0 else s.zeros(2,1))) for j in (0,1,2)}
                e=[z[0]-z[1],z[2]-z[1]]
                p=[(v.T*c.inv()*v)[0] for v,c in zip((e[0],e[1],z[0]-z[2]),(S[0]+S[1],S[2]+S[1],S[0]+S[2]))]
                f=[]
                for idx,(j,o) in enumerate(((0,2),(2,0))):
                    C=S[1]*(S[o]+S[1]).inv();u=e[idx]-C*e[1-idx];Q=S[j]+S[1]-S[1]*(S[o]+S[1]).inv()*S[1]
                    f.append((u.T*Q.inv()*u)[0])
                flags=[False,False];ok=[True,True]
                n=1 if scope=='known' else 2;bounds=threshold(3 if b=='B5' else n)
                if b in ('B0','B1','B2'):
                    flags=[above(p[i],bounds) for i in range(2)]
                    if scope=='known':flags[1]=False
                elif b in ('B4','B4H'):
                    if scope=='known':flags[0]=above(f[0],bounds)
                    else:
                        i=0 if f[0]>=f[1] else 1;flags[i]=above(f[i],bounds)
                    ok=[not above(v,health) for v in f]
                else:
                    good=[not above(v,bounds) for v in p]
                    if all(good):chosen=3
                    elif scope=='known':chosen=2
                    elif good[0] and (not good[1] or p[0]<=p[1]):chosen=1
                    elif good[1]:chosen=2
                    else:chosen=0
                    flags=[not bool(chosen>>h&1) for h in range(2)]
                if k<7:flags=[False,False]
                for h in range(2):
                    if b=='B4H':
                        passes[h]=passes[h]+1 if ok[h] else 0
                        if flags[h]:active[h]=True;passes[h]=0
                        if passes[h]>=5:active[h]=False
                    else:
                        if flags[h]:remaining[h]=5
                ph=[int(not active[h]) if b=='B4H' else int(remaining[h]==0) for h in range(2)]
                remaining=[max(0,v-1) for v in remaining]
                if b=='B0':ph=[1,1]
                mask=ph[0]+2*ph[1];mode=modes['fixed_modes' if b in ('B0','B1') else 'redesigned_modes'][mask]
                K,G=sm(mode['K']),sm(mode['G']);prior=A*prev
                nu=sum((ph[h]*(y[j][k]+(a[k] if attacked and j==0 else s.zeros(2,1))-HH[j]*prior) for h,j in enumerate((0,2))),s.zeros(2,1))
                prev=prior+K*(y[1][k]-HH[1]*prior)+G*nu
                posts.append(prev);phis.append(ph);alarmhistory.append(flags);scores_saved.append([str(v) for v in f])
            runs[scope,b,attacked]=posts
            records[scope,b,attacked]=dict(phi=phis,alarms=alarmhistory)
    out={}
    for scope in ('known','unknown'):
      for b in ('B0','B1','B2','B4','B5','B4H'):
        ra=records[scope,b,True];rc=records[scope,b,False];d=[]
        for k in range(32):
            v=runs[scope,b,True][k]-runs[scope,b,False][k];d.append((v.T*v)[0])
        read=[k for k in range(8,32) if ra['phi'][k][0] and not ra['phi'][k-1][0]]
        out[scope+'/'+b]=dict(attack_alarms=[k for k,v in enumerate(ra['alarms']) if any(v)],
          clean_alarms=[k for k,v in enumerate(rc['alarms']) if any(v)],readmissions=read,
          danger_at_old_time15_exact=bool(d[15]>=s.Rational(1,10)),bias_squared_at15=float(d[15]),
          readmission_values=[dict(k=k,bias_squared=float(d[k]),strictly_dangerous=bool(d[k]>s.Rational(1,10))) for k in read],
          rational_readmission_harms={str(k):str(d[k]) for k in read})
    assert out['known/B1']['bias_squared_at15']<.089 and out['known/B2']['bias_squared_at15']>.127
    assert out['known/B1']['attack_alarms']==[7,10]
    dump(H/'independent_math_certificate.json',dict(date='2026-11-16',geometry=geo,joint_V=str(V),
      aggregate_attribution=ranks,mode_certificates=cert,common_norm_bound=.9,
      scope='exact rational geometry and gains as finite decimals; not exact original-author 50-node/DARE uncertainty proof'))
    dump(H/'old_counterexample_new_policies.json',dict(source_sha256=hashlib.sha256((OLD/'reentry_counterexample_exact.json').read_bytes()).hexdigest(),
      old_energy=str((xi.T*W*xi)[0]),all_decisions_exact_rational_vs_log_bounds=True,results=out,
      unchanged_old_witness=True,no_new_sample_or_attack=True))
    print(json.dumps({k:{f:v[f] for f in ('attack_alarms','readmissions','bias_squared_at15')} for k,v in out.items()}))
if __name__=='__main__':main()
