"""Validated rectangular enclosures of the current-only sign policy expectation.

r=||beta||^2~Exp(1), theta uniform[0,pi). Physical Gaussian noise
and the secret independent future watermarks have been integrated analytically.
"""
from pathlib import Path
import json,sys,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parent.parent.parent
try:
    from flint import arb,ctx
except ImportError:
    # Optional task-local dependency; alternatively install requirements.txt.
    sys.path.insert(0,str(H/'tmp/validated_math'))
    from flint import arb,ctx
ctx.prec=100
import sympy as s
def ba(x):
    z=s.Rational(x);return arb(int(z.p))/arb(int(z.q))
def mat(rows):return [[ba(v) for v in row] for row in rows]
def dot(a,b):return sum((x*y for x,y in zip(a,b)),arb(0))
def rectangle(mid,rad):return mid+arb(0,rad)
def enclosure(x):
    import math
    den=10**12
    lo=s.Rational(math.floor(float(x.lower())*den),den)
    hi=s.Rational(math.ceil(float(x.upper())*den),den)
    while not x>ba(lo):lo-=s.Rational(1,den)
    while not ba(hi)>x:hi+=s.Rational(1,den)
    return dict(lower_exact=str(lo),upper_exact=str(hi),
                lower_display=float(lo),upper_display=float(hi),
                decimal_endpoints_verified_outward=True)
def integrate(inp,nr,na,R=24):
    v=[x[0] for x in mat(inp['first_waveform'])];C=mat(inp['C0'])
    d=mat(inp['future_mean_row'])[0];ell=[x[0] for x in mat(inp['ell'])];Py=mat(inp['Py'])
    half=arb(1)/2;tr=(C[0][0]+C[1][1])/2
    D=[[C[i][j]-(tr if i==j else 0) for j in range(2)] for i in range(2)]
    base=dot(v,[dot(row,ell) for row in C]);factor=(2/arb.pi()).sqrt()
    total=arb(0)
    for ir in range(nr):
        lo=arb(R)*ir/nr;hi=arb(R)*(ir+1)/nr
        r=rectangle((lo+hi)/2,(hi-lo)/2)
        mass=(-lo).exp()-(-hi).exp()
        row=arb(0)
        for ia in range(na):
            theta=rectangle(arb.pi()*(2*ia+1)/(2*na),arb.pi()/(2*na))
            u=[theta.cos(),theta.sin()]
            z=[2*ell[i]-2*r/(half+r)*u[i]*dot(u,ell) for i in range(2)]
            num=base+r*(dot(v,u)*dot(u,[dot(rowD,z) for rowD in D])-
                        dot(v,[dot(rowD,u) for rowD in D])*dot(u,z))+dot(d,z)
            var=dot(z,[dot(rowP,z) for rowP in Py])
            if not var>0:
                floor=dot(ell,ell)/(2*(half+hi)**2)
                # Intersect the elementary spectral lower bound with the
                # interval upper bound; both are valid for this rectangle.
                upper=var.upper()
                assert upper>floor
                var=rectangle((floor.lower()+upper)/2,(upper-floor.lower())/2)
            row+=factor*num/var.sqrt()
        total+=mass*row/na
    # |conditional covariance|/sd <= ||(.5+r)C0+d||/sqrt(lambda_min Py).
    # Py>=R0=.5I, ||v0||<=1; use explicit rational upper norms.
    Cf=s.Matrix([[s.Rational(x) for x in r] for r in inp['C0']])
    df=s.Matrix([[s.Rational(x) for x in r] for r in inp['future_mean_row']])
    import math
    ac=s.Rational(math.ceil(float(sum(x*x for x in Cf))**.5*10**8),10**8)
    ad=s.Rational(math.ceil(float(sum(x*x for x in df))**.5*10**8),10**8)
    assert ac*ac>=sum(x*x for x in Cf) and ad*ad>=sum(x*x for x in df)
    tail=arb(3)/2*(-arb(R)).exp()*((arb(R)+arb(3)/2)*ba(ac)+ba(ad))
    result=total+arb(0,tail.upper())
    return result,dict(radial_cutoff=R,rectangles=nr*na,
      absolute_tail_bound=str(tail),C0_norm_upper=str(ac),future_row_norm_upper=str(ad))
def main():
    cert=json.loads((H/'exact_information_certificates.json').read_text('utf8'));out={}
    for name in ('unstable','stable'):
        rows={}
        for label in ('single','cumulative'):
            z=cert['models'][name][label];history=[]
            OL=ba(z['OL_upper_increment']);Q=ba(z['witness_quadratic_increment'])
            Hu=OL+2*(10*ba(z['history_genie_hcov_trace_exact'])).sqrt() if label=='single' else None
            for nr,na in [(128,64),(256,128),(512,256),(1024,512)]:
                value,meta=integrate(z['current_integral_inputs'],nr,na)
                increment=Q+2*arb(10).sqrt()*value
                passed=bool(increment>Hu) if Hu is not None else bool(increment>OL)
                history.append(dict(radial_bins=nr,angular_bins=na,cross_interval=str(value),
                  cross_enclosure=enclosure(value),increment_interval=str(increment),
                  increment_enclosure=enclosure(increment),strict_target_passed=passed,**meta))
                print(json.dumps(dict(model=name,objective=label,nr=nr,na=na,increment=str(increment),passed=passed)),flush=True)
                if passed:break
            rows[label]=dict(refinements=history,strict_C_policy_above_OL=bool(increment>OL),
              current_inputs_sha256=hashlib.sha256(json.dumps(z['current_integral_inputs'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
              H_genie_upper_interval=str(Hu) if Hu is not None else None,
              H_genie_upper_enclosure=enclosure(Hu) if Hu is not None else None,
              strict_C_policy_above_all_H_single=bool(increment>Hu) if Hu is not None else None,
              rigorous_interval=True,not_causal_C_global_optimum=True)
        out[name]=rows
    (H/'validated_current_integrals.json').write_text(json.dumps(dict(date='2026-11-17',
       models=out,Arb_bits=100,new_physical_samples=0),ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
