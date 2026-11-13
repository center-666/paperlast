"""Fresh exact certificate from F/g decimal inputs, not old classification.

Python integer arithmetic classifies W/100-B'B at all 13 times. A second
floating implementation computes informative spectral margins only.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal
from itertools import product
from math import lcm
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent
INPUT=P/'checks/水印收束_理论与映射核验_20261105.json'
raw=INPUT.read_bytes()
d=json.loads(raw,parse_float=Decimal)
F=[[Fraction(x) for x in row] for row in d['F']]
g=[Fraction(x) for x in d['g']]
den=lcm(*[v.denominator for v in sum(F,[])+g])
Fi=[[int(x*den) for x in row] for row in F]; gi=[int(x*den) for x in g]
W=np.array([[10,4.5],[4.5,2.85]])
Fn=np.array(F,float); gn=np.array(g,float)
words=list(product([-1,1],repeat=10))
records=[]; proof_hash=hashlib.sha256()
bytime=np.zeros((13,3),dtype=int)
# A conservative common positive Loewner margin, checked with integers.
margin=Fraction(1,4000)
for wi,s in enumerate(words):
    I=[[0,0] for _ in range(4)]; B=np.zeros((4,2)); dp=1
    types=[]; mineig=float('inf'); min_margin_det=0
    for k in range(13):
        I=[[sum(Fi[i][j]*I[j][a] for j in range(4))+
            (s[k]*gi[i]*[10,k][a]*dp if k<10 else 0)
            for a in range(2)] for i in range(4)]
        dp*=den; bden=10*dp; qden=2000*bden*bden
        H00=sum(r[0]**2 for r in I); H01=sum(r[0]*r[1] for r in I); H11=sum(r[1]**2 for r in I)
        a=200*bden*bden-2000*H00; b=90*bden*bden-2000*H01; c=57*bden*bden-2000*H11
        det=a*c-b*b
        if a>0 and det>0: typ=0
        elif a>=0 and c>=0 and det>=0: typ=1
        else: typ=2
        types.append(typ); bytime[k,typ]+=1
        proof_hash.update(f'{wi},{k},{a},{b},{c}\n'.encode())
        if typ==0:
            # Check Q >= margin I strictly, with a common integer scaling.
            aa=a*margin.denominator-qden*margin.numerator
            cc=c*margin.denominator-qden*margin.numerator
            bb=b*margin.denominator
            if aa<=0 or aa*cc-bb*bb<=0: min_margin_det+=1
        B=Fn@B+(np.outer(gn,[1,k/10])*s[k] if k<10 else 0)
        mineig=min(mineig,float(np.linalg.eigvalsh(W/100-B.T@B)[0]))
    typ=max(types)
    records.append(dict(index=wi,bits=''.join('1' if x==1 else '0' for x in s),signs=list(s),
                        classification=['strict_safe','boundary_only','danger_reachable'][typ],
                        per_time_classification=types,minimum_eigen_margin_float=mineig,
                        common_margin_certificate_passed=(min_margin_det==0 and typ==0)))
safe=[r['index'] for r in records if r['classification']=='strict_safe']
boundary=[r['index'] for r in records if r['classification']=='boundary_only']
danger=[r['index'] for r in records if r['classification']=='danger_reachable']
assert all(records[1023-i]['classification']==records[i]['classification'] for i in range(1024))
assert all(records[i]['common_margin_certificate_passed'] for i in safe)
# Compare only after the independent calculation is complete.
old=json.loads((P/'glrt_screening_20261112/exact_sign_danger_cap.json').read_text(encoding='utf-8'))
assert danger==old['danger_capable_word_union_indices']
assert len(safe)==960 and not boundary and len(danger)==64
out=dict(date='2026-11-13',input_sha256=hashlib.sha256(raw).hexdigest(),
         exact_input='JSON decimals parsed as Decimal then Fraction; frozen model, not DARE interval',
         common_denominator=str(den),safe_indices=safe,boundary_indices=boundary,danger_indices=danger,
         safe_count=len(safe),boundary_count=len(boundary),danger_count=len(danger),
         per_time_counts_safe_boundary_danger=bytime.tolist(),
         overall_sign_reversal_symmetry=True,common_exact_margin=str(margin),
         exact_margin_is_on_k_0_to_12=True,
         minimum_safe_eigen_margin_float=min(records[i]['minimum_eigen_margin_float'] for i in safe),
         all_matrix_integer_stream_sha256=proof_hash.hexdigest(),records=records,
         integer_matrix='Q=[[a,b],[b,c]]/(2000*(10*d^(k+1))^2); recurrence and entries in source',
         times=list(range(13)),physical_trajectories=0)
(HERE/'safe_words.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['records','safe_indices','danger_indices']},ensure_ascii=False,indent=2))
