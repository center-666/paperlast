"""Independent exact-moment checks for the exogenous modal-outage example."""
from pathlib import Path
import json
import numpy as np


def recursive_moments(a,Q,R,pi,p11,T):
    p01=pi*(1-p11)/(1-pi)
    if not 0<=p01<=1:raise ValueError('infeasible stationary Markov parameters')
    s=0.;t=0.;trajectory=[]
    for _ in range(T):
        s=a*a*(p11*s+p01*t)+Q*pi;t=R*(1-pi)
        trajectory.append(s+t)
    return np.asarray(trajectory)


def stationary_formula(a,Q,R,pi,p11):
    return R*(1-pi)+pi*(Q+a*a*(1-p11)*R)/(1-a*a*p11)


def main():
    a=1.2;Q=1.;R=.1;pi=.05;rows=[]
    for p11 in (.05,.6,.8):
        exact=recursive_moments(a,Q,R,pi,p11,500)
        finite=a*a*p11<1
        row={'a':a,'Q':Q,'R':R,'stationary_missing_probability':pi,
             'p11':p11,'p01':pi*(1-p11)/(1-pi),'a_squared_times_p11':a*a*p11,
             'mean_square_stable':finite,'exact_risk_at_T150':float(exact[149])}
        if finite:
            reference=stationary_formula(a,Q,R,pi,p11)
            assert abs(exact[-1]-reference)<1e-12
            row['stationary_risk']=reference
        else:assert exact[-1]>exact[149]>exact[49]
        rows.append(row)
    # A single deterministic scheduled example independently computes exact
    # Gaussian-noise variance by a convolution matrix, not the Markov formula.
    for q in [np.array([1,1,0,1,0,1,1,1,0],bool),np.zeros(9,bool),np.ones(9,bool)]:
        coefficients=np.zeros((9,18));variance=0.
        for k in range(9):
            if k and q[k]:coefficients[k]=a*coefficients[k-1]
            coefficients[k,2*k]=np.sqrt(Q) if q[k] else 0.
            coefficients[k,2*k+1]=-np.sqrt(R) if not q[k] else 0.
            variance=a*a*variance+Q if q[k] else R
            assert abs(np.sum(coefficients[k]**2)-variance)<1e-11
    result={'status':'PASS','scope':'Exact theoretical illustration with exogenous Markov gates; not the endogenous detector model',
            'rows':rows,'independent_deterministic_gaussian_convolution_check':True}
    path=Path(__file__).resolve().parent/'modal_risk_validation.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
