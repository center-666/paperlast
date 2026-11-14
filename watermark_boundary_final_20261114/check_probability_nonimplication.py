"""Exact counterexample to inferring true power from a mean noncentrality.

This is a logical counterexample on scalar Lambda distributions, not a newly
designed watermark, nor a fair-resource comparison of allowed schemes.
"""
from pathlib import Path
from fractions import Fraction as F
from math import factorial
import json
from scipy.stats import chi2,ncx2
H=Path(__file__).resolve().parent
alpha=F(1,300)
# e>8/3 implies sf(chi2_8,24)=373 e^-12 < alpha, hence tau_alpha<24.
assert F(373)*F(3,8)**12<alpha
# e<11/4. One Poisson-mixture term j=5 at noncentrality10, threshold24:
lo=F(4,11)**17*F(5**5,factorial(5))*sum((F(12**k,factorial(k))for k in range(9)),F(0))
# Lambda_B=0 with .99 probability and2000 with .01. Its power<=.99alpha+.01.
up=F(99,100)*alpha+F(1,100)
assert lo>up
tau=float(chi2.isf(float(alpha),8))
out=dict(date='2026-11-14',degrees_of_freedom=8,alpha=str(alpha),
    law_A=dict(Lambda=10,mean=10,power_rigorous_lower=str(lo),power_display=float(ncx2.sf(tau,8,10))),
    law_B=dict(Lambda_support=[0,2000],probabilities=['99/100','1/100'],mean=20,
               power_rigorous_upper=str(up),power_display=float(.99*ncx2.sf(tau,8,0)+.01*ncx2.sf(tau,8,2000))),
    strict_reversal_certified=True,
    proof='chi2 Poisson mixture; retain j5 only; 8/3<e<11/4; exact rational comparison',
    allowed_watermark_comparison=False,fair_resource_claim=False,new_algorithm=False)
(H/'probability_nonimplication.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
