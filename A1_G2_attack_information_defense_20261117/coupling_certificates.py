"""Frozen-gain rational contraction and first-packet information bounds."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import sympy as s
import numpy as np
from gain_models import ref
H=Path(__file__).resolve().parent
def sm(a):return s.Matrix([[s.Rational(str(v)) for v in row] for row in a])
def main():
    source=json.loads((H/'gain_designs.json').read_text('utf8'))['models'];out={}
    for name,ds in source.items():
        m=ref.load_model(name);A=sm(m.A);Q=sm(m.Q);Hi=sm(m.H[1]);Hs=sm(m.H[0]+m.H[2]);R0=sm(m.R[0]);R1=sm(m.R[1]);R2=sm(m.R[2])
        px=[s.eye(2)]
        for k in range(7):px.append(A*px[-1]*A.T+Q)
        Y={}
        for n in (7,8):
            M=s.zeros(2*n)
            for i in range(n):
                for j in range(n):
                    C=A**(i-j)*px[j] if i>=j else px[i]*(A**(j-i)).T
                    if i==j:C+=R0
                    M[2*i:2*i+2,2*j:2*j+2]=C
            Y[n]=M.inv()
        rr={}
        for label,g in ds.items():
            Ks=list(map(sm,g['K']));Gs=list(map(sm,g['G']));Fs=[]
            for K,G in zip(Ks,Gs):
                F=(s.eye(2)-K*Hi-G*Hs)*A
                assert s.trace(F.T*F)<s.Rational(1,4)
                Fs.append(F)
            C=[];Pe=[];eye=[]
            for k in range(32):
                K,G=Ks[k],Gs[k];J=s.eye(2)-K*Hi-G*Hs
                D=-s.eye(2) if k==0 else A*C[-1]*A.T-Q
                V=s.eye(2) if k==0 else A*Pe[-1]*A.T+Q
                C.append(J*D);Pe.append(J*V*J.T+K*R1*K.T+G*(R0+R2)*G.T)
                eye.append(C[-1]+G*R0)
            cross=[]
            for t in range(8):
                Phi=s.eye(2)
                for k in range(t+1,8):Phi=Fs[k]*Phi
                cross.append(Phi*eye[t])
            G=Gs[7];CH=s.Matrix.hstack(*cross[:7]);CC=s.Matrix.hstack(*cross)
            hu=s.trace(G.T*CH*Y[7]*CH.T*G)
            cu=s.Rational(7,4)*s.trace(G.T*CC*Y[8]*CC.T*G)
            hl=s.Rational(4,9)*sum(v*v for v in G.T*cross[6])/s.trace(px[6]+R0)
            assert hu>=hl>=0 and cu>=hu
            small=s.Rational(1,10**25)
            if label=='raw_orthogonal':assert cu<small
            normal=sum(s.trace(p) for p in Pe)/32
            delta2=max(sum(v*v for v in Y0) for Y0 in eye)
            maxG2=max(sum(v*v for v in G0) for G0 in Gs)
            assert maxG2<1
            uniform=s.Rational(14,3)*maxG2*delta2
            if label=='raw_orthogonal':assert uniform<s.Rational(1,10**27)
            # Exact lower singular-value bound for population risk witnesses.
            lo=s.Rational(int(np.floor(np.linalg.svd(np.array(g['G'][7]))[1][-1]*10**6)),10**6)
            test=G.T*G-lo*lo*s.eye(2)
            assert test[0,0]>0 and test.det()>0
            rr[label]=dict(Gamma_H_lower_exact=str(hl),Gamma_H_upper_exact=str(hu),Gamma_C_upper_exact=str(cu),
              Gamma_H_lower=float(hl),Gamma_H_upper=float(hu),Gamma_C_upper=float(cu),
              candidate_Gamma_C_less_than_1e_minus_25=bool(cu<small),first_target_cross_covariance=[[str(v) for v in row] for row in eye[7].tolist()],
              exact_healthy_MSE0_31=str(normal),healthy_MSE_display=float(normal),
              max_raw_target_correlation_squared_exact=str(delta2),
              uniform_Gamma_C_upper_0_31_exact=str(uniform),
              G7_sigma_min_lower_exact=str(lo),all_32_modes_Frobenius_squared_less_than_one_quarter=True,
              full_pulse_first_bias_lower_exact=str(s.Rational(10)*(1-s.Rational(1,10**12))**2*lo*lo/4),
              full_pulse_danger_for_all_secrets=bool(s.Rational(10)*(1-s.Rational(1,10**12))**2*lo*lo/4>s.Rational(1,10)),
              interpretation='Gamma for B=G7*M7 and healthy posterior error at7; raw history genie used only for bounds')
        base=s.Rational(rr['Niu']['exact_healthy_MSE0_31'])
        for label,z in rr.items():
            cost=s.Rational(z['exact_healthy_MSE0_31'])
            assert cost<=s.Rational(6,5)*base
            z['healthy_cap_20_certified']=True
            if label in ('robust_matched','shrink_matched'):
                target=s.Rational(rr['raw_orthogonal']['exact_healthy_MSE0_31'])
                assert abs(cost-target)<s.Rational(1,10**12)
                z['normal_matching_absolute_error_exact']=str(abs(cost-target))
        assert s.Rational(rr['raw_orthogonal']['Gamma_C_upper_exact'])<s.Rational(rr['Niu']['Gamma_H_lower_exact'])
        rr['raw_orthogonal']['strict_Gamma_reduction_vs_Niu_certified']=True
        out[name]=rr
        print(json.dumps({name:{k:dict(Gamma_C_upper=r['Gamma_C_upper'],sigmaG=r['G7_sigma_min_lower_exact']) for k,r in rr.items()}}),flush=True)
    (H/'coupling_certificates.json').write_text(json.dumps(dict(date='2026-11-17',models=out,
      rationalized_gain_inputs=True,ideal_constraint_zero_is_separate_from_roundoff=True,
      model_or_DARE_uncertainty_certified=False),ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
