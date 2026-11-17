"""Within the SAME frozen model: tiny Gamma does not remove current Q-information."""
from pathlib import Path
import json,sys,math
sys.dont_write_bytecode=True
import sympy as s
import numpy as np
from risk_bounds import arb,ctx,ba,enclosure,sm
from gain_models import ref
H=Path(__file__).resolve().parent
ctx.prec=100
def mat(x):return [[ba(v) for v in row] for row in x.tolist()]
def mm(A,B):return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def box(mid,rad):return mid+arb(0,rad)
def integrate(T,Py,nr,na,R=12):
    Tb=mat(T);Pb=mat(Py);detP=ba(Py.det()).sqrt();total=arb(0)
    for ir in range(nr):
        lo=arb(R)*ir/nr;hi=arb(R)*(ir+1)/nr;r=box((lo+hi)/2,(hi-lo)/2)
        mass=(-lo).exp()-(-hi).exp();row=arb(0)
        for ia in range(na):
            theta=box(arb.pi()*(2*ia+1)/(2*na),arb.pi()/(2*na))
            u=[theta.cos(),theta.sin()]
            U=[[u[i]*u[j] for j in range(2)] for i in range(2)]
            M=[[(arb(1)/2 if i==j else arb(0))+r*U[i][j] for j in range(2)] for i in range(2)]
            S=[[(arb(2) if i==j else arb(0))-2*r/(arb(1)/2+r)*U[i][j] for j in range(2)] for i in range(2)]
            V=mm(mm(S,Pb),S);ds=detP/(arb(1)/2*(arb(1)/2+r))
            den=V[0][0]+V[1][1]+2*ds
            if not den>0:
                floor=4*detP/(arb(1)/2*(arb(1)/2+hi))
                upper=den.upper()
                den=box((floor.lower()+upper)/2,(upper-floor.lower())/2)
            D=[[(V[i][j]+(ds if i==j else 0))/den for j in range(2)] for i in range(2)]
            Q=mm(mm(M,Tb),M)
            value=Q[0][0]*D[1][1]+Q[1][1]*D[0][0]-2*Q[0][1]*D[0][1]
            row+=value
        total+=mass*row/na
    # Nonnegative omitted integrand; trace(T) bounds its largest eigenvalue.
    tail=ba(s.trace(T))*(-arb(R)).exp()*((arb(R)+arb(1)/2)**2+2*(arb(R)+arb(1)/2)+2)
    return total,tail
def main():
    gs=json.loads((H/'gain_designs.json').read_text('utf8'))['models'];out={}
    for name,ds in gs.items():
        m=ref.load_model(name);g=ds['raw_orthogonal'];A=sm(m.A)
        Ks=list(map(sm,g['K']));Gs=list(map(sm,g['G']))
        F=[(s.eye(2)-K*sm(m.H[1])-G*sm(m.H[0]+m.H[2]))*A for K,G in zip(Ks,Gs)]
        W=[None]*32;W[-1]=s.eye(2)
        for t in range(30,-1,-1):W[t]=s.eye(2)+F[t+1].T*W[t+1]*F[t+1]
        def Phi(u,t):
            p=s.eye(2)
            for k in range(t+1,u+1):p=F[k]*p
            return p
        R=s.zeros(20)
        for i,t in enumerate(range(7,17)):
            for j,u in enumerate(range(7,17)):
                C=Gs[t].T*Phi(u,t).T*W[u]*Gs[u] if t<=u else (Gs[u].T*Phi(t,u).T*W[t]*Gs[t]).T
                if i==j:C=s.Rational(5,4)*C+s.trace(C)*s.eye(2)/4
                R[2*i:2*i+2,2*j:2*j+2]=C
        upper=10*max(R[i,i]+sum(abs(R[i,j]) for j in range(20) if j!=i) for i in range(20))
        C=None;Px=s.eye(2)
        for k in range(8):
            if k:Px=A*Px*A.T+sm(m.Q)
            D=-s.eye(2) if k==0 else A*C*A.T-sm(m.Q)
            C=(s.eye(2)-Ks[k]*sm(m.H[1])-Gs[k]*sm(m.H[0]+m.H[2]))*D
        Ey=C+Gs[7]*sm(m.R[0]);Py=Px+sm(m.R[0])
        T=Gs[7].T*W[7]*Gs[7]
        cross_matrix=Gs[7].T*W[7]*Ey
        cross_gamma=s.Rational(7,4)*s.trace(cross_matrix*Py.inv()*cross_matrix.T)
        cross_bound=2*(10*ba(cross_gamma)).sqrt()
        history=[]
        for nr,na in [(64,32),(128,64),(256,128),(512,256)]:
            core,tail=integrate(T,Py,nr,na)
            # Full expected bias >= truncated integral; total cross may have
            # either sign, but the Gaussian-regression upper bound is tiny.
            safe_energy=s.Rational(10)*(1-s.Rational(1,10**12))**2
            lower_actual=ba(safe_energy)*core-cross_bound
            history.append(dict(radial_bins=nr,angular_bins=na,
              actual_increment_lower_enclosure=enclosure(lower_actual),
              omitted_positive_tail_upper=enclosure(10*tail),
              strictly_above_all_OL=bool(lower_actual>ba(upper))))
            print(json.dumps(dict(model=name,nr=nr,na=na,lower=str(lower_actual),OL_upper=float(upper),
              strict=bool(lower_actual>ba(upper)))),flush=True)
            if lower_actual>ba(upper):break
        assert lower_actual>ba(upper)
        out[name]=dict(OL_all_waveform_Gershgorin_upper_exact=str(upper),OL_upper=float(upper),
          current_perp_pulse_lower=history[-1]['actual_increment_lower_enclosure'],
          refinements=history,cross_gamma_upper_exact=str(cross_gamma),coded_energy_exact=str(safe_energy),
          cross_abs_upper=enclosure(cross_bound),same_energy10=True,no_realized_M_or_error_input=True,
          information_channel='conditional quadratic Q rather than healthy-error cross term')
    (H/'quadratic_information_counterexample.json').write_text(json.dumps(dict(date='2026-11-17',models=out,
      same_frozen_five_node_model=True,parameter_free_stress_test=True,new_watermark=False),ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
