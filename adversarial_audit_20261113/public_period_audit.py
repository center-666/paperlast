"""Finite exhaustive public-period baseline, with independent floating maps.

This script is an audit comparator, not a new watermark or a global optimum
over every deterministic sequence. Periods 1,...,10 are enumerated completely.
No safe_words classifications are used as algorithm input.
"""
from pathlib import Path
from itertools import product
import hashlib,json
import numpy as np
import sympy as sp

HERE=Path(__file__).resolve().parent
P=HERE.parent
source=P/'checks/水印收束_理论与映射核验_20261105.json'
d=json.loads(source.read_text(encoding='utf-8'))
F,g,W=np.array(d['F']),np.array(d['g']),np.array(d['W10'])
v,U=np.linalg.eigh(W)
Winvhalf=(U/np.sqrt(v))@U.T
words=np.array(list(product([-1.,1.],repeat=10)))
maps=np.zeros((1024,4,2));peak=np.zeros(1024);peak_k=np.zeros(1024,dtype=int)
cap13=np.zeros(1024)
for k in range(13):
    maps=np.einsum('ij,wja->wia',F,maps)
    if k<10:
        maps+=g[None,:,None]*words[:,k,None,None]*np.array([1.,k/10])[None,None,:]
    V=maps@Winvhalf
    vals=10*np.linalg.eigvalsh(np.einsum('wki,wkj->wij',V,V))[:,-1]
    take=vals>peak;peak_k[take]=k;peak=np.maximum(peak,vals)
    if k==12:cap13=vals
# Previously independently proved rational ||F^r||^2<1.03,
# ||F^10||<.998 imply a per-word conservative infinite-tail bound.
tail_cap=1.03*cap13
upper=np.maximum(peak,tail_cap)
case_rows=[]
alllegal=[]
for period in range(1,11):
    legal=[]
    for bits in product([0,1],repeat=period):
        ids=[]
        for phase in range(period):
            window=[bits[(phase+j)%period] for j in range(10)]
            ids.append(sum(x<<(9-j) for j,x in enumerate(window)))
        # Check every unknown-onset phase; classify from fresh maps only.
        if all(upper[i]<.1 for i in ids):
            row=dict(period=period,bits=list(bits),signs=[2*x-1 for x in bits],
                     rotation_word_indices=ids,lower=float(max(peak[ids])),
                     all_time_upper_using_per_word_tail=float(max(upper[ids])),
                     exact_peak_if_lower_equals_upper=bool(max(peak[ids])==max(upper[ids])))
            legal.append(row);alllegal.append(row)
    best=min(legal,key=lambda r:r['all_time_upper_using_per_word_tail']) if legal else None
    case_rows.append(dict(period=period,total_period_words=2**period,legal_words=len(legal),best=best))
best=min(alllegal,key=lambda r:r['all_time_upper_using_per_word_tail'])
alternating=next(r for r in alllegal if r['period']==2 and r['bits']==[0,1])
# An exact, separate two-word result is stronger than the finite period scan:
# the unavoidable first-update peak is attained by public alternation at every
# later time as well. Powers and a closed convolution are used, not old integer
# recurrence/certificate code. Tail uses F-power inequalities audited separately.
Fr=sp.Matrix([[sp.Rational(str(x)) for x in row] for row in d['F']])
gr=sp.Matrix([sp.Rational(str(x)) for x in d['g']])
Wr=sp.Matrix([[10,sp.Rational(9,2)],[sp.Rational(9,2),sp.Rational(57,20)]])
first=sp.simplify(10*(gr.T*gr)[0]*Wr.inv()[0,0])
powers=[sp.eye(4)]
for _ in range(13):powers.append(powers[-1]*Fr)
exact_rows=[]
for t in range(1,14):
    B=sp.zeros(4,2)
    for j in range(min(t,10)):
        B+=powers[t-1-j]*gr*sp.Matrix([[1,sp.Rational(j,10)]])*(-1)**j
    M=first*Wr/10-B.T*B
    minors=[M[0,0],M[1,1],M.det()]
    assert all(x>=0 for x in minors)
    exact_rows.append(dict(offset_state=t,PSD_exact=True,zero_determinant=bool(M.det()==0)))
    if t==13:
        Mt=first*Wr/10-sp.Rational(103,100)*B.T*B
        assert Mt[0,0]>0 and Mt.det()>0
        exact_tail=True
result=dict(date='2026-11-13',purpose='adversarial finite public deterministic baseline',
  original_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
  methodology='fresh floating B recursion, every phase, periods1..10 exhaustive; old classifications not input',
  finite_period_class_complete=True,all_deterministic_sequences_global_optimum=False,
  numerical_interval_certificate=False,infinite_tail_uses_separately_certified_F_power_bounds=True,
  ratio=.01,E=10,D=.1,period_summaries=case_rows,best_period=best,alternating=alternating,
  total_enumerated_period_words=sum(2**i for i in range(1,11)),
  total_legal_period_words=len(alllegal),all_legal_period_words=alllegal,
  exact_public_alternation=dict(first_update_unavoidable_peak_exact=str(first),
      first_update_unavoidable_peak_float=float(first),
      closed_form_convolution_checks=exact_rows,
      tail_comparison_using_F_power_103_over100=exact_tail,
      universal_lower_bound_for_every_sign_sequence=True,
      global_minimax_pathwise_damage_attained_conditional_on_tail_power_certificate=True,
      meaning='all deterministic or random sign supports have first-update worst damage >=C; alternation reaches exactly C'),
  prefix_peak_damage_by_10_word=peak.tolist(),prefix_peak_time_by_10_word=peak_k.tolist(),
  per_word_infinite_tail_upper=tail_cap.tolist(),no_noise_or_physical_samples=True)
(HERE/'public_period_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['total_enumerated_period_words','total_legal_period_words','best_period','alternating','period_summaries']},ensure_ascii=False,indent=2))
