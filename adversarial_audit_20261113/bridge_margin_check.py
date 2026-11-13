"""Independent conservative perturbation propagation for the DARE bridge.

No old robustness script is called. Rational inequalities use the freshly
rebuilt norm/margin certificate and independently enclosed DARE errors.
"""
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal
import json
HERE=Path(__file__).resolve().parent
cert=json.loads((HERE/'independent_cert.json').read_text())
physical=json.loads((HERE/'physical_origin_audit.json').read_text())
for key in ['exact_F_operator_total_error_upper','exact_g_Euclidean_total_error_upper']:
    assert Q(physical['reconstruction'][key])<Q(1,10**12)
eps=Q(1,10**12);M=Q(203,200) #1.015 >sqrt1.03, hence ||F||<=M
Mp=Q(127,125) #1.016
assert Q(103,100)<M*M and M+eps<Mp
gg=Q(56,1000); hh=Q(27,20)
raw=json.loads((HERE.parent/'checks/水印收束_理论与映射核验_20261105.json').read_text(encoding='utf-8-sig'),parse_float=Decimal)
assert sum(Q(v)**2 for v in raw['g'])<gg*gg
assert hh*hh>Q(181,100)
assert 10*gg*hh*M**12<1 # ||B_t||<=1 for t<=13
u=Q(0)
for k in range(13):u=Mp*u+eps*(1+hh)
assert u<Q(1,10**9)
assert 2*u+u*u<Q(1,10**8)
remaining=Q(1,4000)-(2*u+u*u)
assert remaining>Q(1,5000)
assert Q(499,500)+10*Mp**9*eps<Q(999,1000)
assert Q(203,200)+9*Mp**8*eps<Mp
assert Q(29665,100000)**2>Q(88,1000)
assert Q(3873,1000)**2>15 # ||xi||<=sqrt15 since W>=2/3I
assert Q(29665,100000)+u*Q(3873,1000)<Q(297,1000)
tail=(Mp*Q(297,1000))**2
assert tail<Q(1,10)
out=dict(date='2026-11-13',input_matrix_error_upper='1e-12',
    fresh_closed_convolution_margin=cert['common_margin_strict'],
    prefix_B_error_upper=str(u),remaining_Q_margin_lower=str(remaining),
    perturbed_F10_upper='.999',perturbed_remainder_Fpower_upper='1.016',
    all_tail_damage_upper=str(tail),all_tail_damage_upper_float=float(tail),
    certified_for_exact_declared_stacked_KF=True,engineering_parameters_certified=False)
(HERE/'bridge_margin_check.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
