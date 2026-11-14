"""Rational DARE enclosure for the declared two-dimensional aggregate instance.

High precision only proposes P0; rational sub/supersolution inequalities prove
the enclosure. This is not a claim about the original missing 50-node config.
"""
from pathlib import Path
from decimal import Decimal
import json,hashlib
import mpmath as mp
import sympy as s
H=Path(__file__).resolve().parent;R=s.Rational
mp.mp.dps=90
C=s.Matrix([[R(11,10),0],[0,R(6,5)],[R(11,5),0],[0,R(12,5)]])
V=s.diag(R(1,2),R(1,2),1,1);J=C.T*V.inv()*C
I=s.eye(2);eps=R(1,10**25)
raw=json.loads((H/'five_node_aggregate_model.json').read_text(),parse_float=Decimal)
witness=json.loads((H/'first_packet_limit.json').read_text(),parse_float=Decimal)
def mm(A):return mp.matrix([[mp.mpf(str(v.p))/mp.mpf(str(v.q)) for v in row] for row in A.tolist()])
def pd(M):
    assert M==M.T
    minors=[M[0,0],M.det()]
    assert all(v>0 for v in minors)
    return [str(v) for v in minors]
rows=[]
for name,d in raw['models'].items():
    A=s.Matrix([[R(str(v))for v in row]for row in d['A']]);Am=mm(A);Jm=mm(J)
    Pm=mp.eye(2)
    for it in range(300):
        N=Am*(Pm**-1+Jm)**-1*Am.T+mp.eye(2)
        diff=max(abs(v)for v in N-Pm);Pm=N
        if diff<mp.mpf('1e-75'):break
    else:raise AssertionError('candidate generation failed')
    P0=s.Matrix(2,2,lambda i,j:R(mp.nstr(Pm[i,j],60)))
    # Preserve exact symmetry after decimal conversion.
    P0=(P0+P0.T)/2
    f=lambda P:A*(P.inv()+J).inv()*A.T+I
    lo=P0-eps*I;hi=P0+eps*I
    proofs={'positive_lower':pd(lo),'subsolution':pd(f(lo)-lo),'supersolution':pd(hi-f(hi))}
    L0=P0*C.T*(C*P0*C.T+V).inv();F0=(I-L0*C)*A;G0=L0[:,2:4]
    assert sum(v*v for v in C)<25 and sum(v*v for v in A)<4
    assert sum(v*v for v in L0)<4
    # DeltaL=(I-L0 C) DeltaP C^T (C P* C^T+V)^-1.
    dl=110*eps;df=10*dl
    node=d['nodes'][1]
    Fs=s.Matrix([[R(str(v))for v in row]for row in node['posterior_F']])
    Gs=s.Matrix([[R(str(v))for v in row]for row in node['G']])
    assert sum(v*v for v in F0-Fs)<R(1,10**24)
    assert sum(v*v for v in G0-Gs)<R(1,10**24)
    total=R(1,10**12)+df
    assert total<R(1,10**10)
    center_bound=R(119,1000) if name=='unstable' else R(99,1000)
    proofs['F_center_norm']=pd(center_bound**2*I-F0.T*F0)
    full_bound=R(12,100) if name=='unstable' else R(1,10)
    assert center_bound+df<full_bound
    row=next(v for v in witness['rows']if v['model']==name)
    a=s.Matrix([R(str(v))for v in row['first_packet_attack']])
    e=(a.T*a)[0];d0=(a.T*G0.T*G0*a)[0]
    assert e<R(3,2) and d0>R(102,1000)
    # ||G0 a||<=||L0|| ||a||<4, ||a||<2; squared-norm error<16 dl+4dl^2.
    assert d0-(16*dl+4*dl**2)>R(101,1000)>R(1,10)
    rows.append(dict(model=name,P0=[[str(v)for v in row]for row in P0.tolist()],
        epsilon=str(eps),rational_leading_minors=proofs,
        gain_error_upper=str(dl),F_error_upper=str(df),
        saved_F_G_operator_error_upper=str(total),
        exact_F_norm_upper=str(full_bound),
        witness_total_energy=str(e),exact_DARE_witness_damage_lower='101/1000',
        original_missing_50_node_configuration_certified=False))
out=dict(date='2026-11-14',method='rational Riccati sub/supersolutions, monotone map and unique stabilizing DARE solution',
    assumptions='declared aggregate MMSE C/R, positive Q, observable local H; zero entering displacement, no gating',
    no_physical_samples=True,rows=rows)
(H/'aggregate_gap_certificate.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{'model':r['model'],'exact_F_norm_upper':r['exact_F_norm_upper'],
                  'exact_witness_damage_lower':r['exact_DARE_witness_damage_lower']}for r in rows],indent=2))
