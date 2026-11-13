"""Exact permission-boundary witness; frozen decimals are rational inputs.

This is a stronger-attacker test, not a replacement for the autonomous attack.
No original file is modified and no physical trajectory/noise is sampled.
"""
from fractions import Fraction as Q
from pathlib import Path
from hashlib import sha256
import json

HERE=Path(__file__).resolve().parent
P=HERE.parent
source=P/'checks/水印收束_理论与映射核验_20261105.json'
raw=source.read_bytes()
d=json.loads(raw.decode('utf-8-sig'))
F=[[Q(str(x)) for x in row] for row in d['F']]
g=[Q(str(x)) for x in d['g']]
W=[[Q(str(x)) for x in row] for row in d['W10']]
xi=[Q(1,2),Q(1)]
energy=sum(xi[i]*W[i][j]*xi[j] for i in range(2) for j in range(2))
assert energy==Q(197,20) and energy<10
b=[Q(0)]*4
for t in range(10):
    amp=xi[0]+Q(t,10)*xi[1]
    b=[sum(F[i][j]*b[j] for j in range(4))+g[i]*amp for i in range(4)]
damage=sum(x*x for x in b)
assert damage>Q(1,10)
# Encoding/decoding are s_t in {-1,+1}; a_t=s_t H_t xi is decoded as H_t xi.
# Each sample energy is unchanged, so the same energy is valid for every word.
out=dict(date='2026-11-13',input_sha256=sha256(raw).hexdigest(),
         exact_input_model='frozen F/g decimals interpreted as rational',
         xi=[str(x) for x in xi],coded_energy=str(energy),
         terminal_squared_bias_float=float(damage),
         dangerous_exact_comparison=True,
         terminal_squared_bias_exact=str(damage),
         comparison_to_threshold_exact=str(damage-Q(1,10)),
         encoded_attack='a_t=s_t*(p+t*v/10), t=0,...,9',
         decoded_attack='s_t*a_t=p+t*v/10',
         knowledge='current sign known before each injection; stronger A3',
         autonomous_coded_attack_family_preserved=False,
         full_start_stop_stealth_claim=False,
         physical_samples=0)
(HERE/'permission_counterexamples.json').write_text(
    json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in
                 ['terminal_squared_bias_exact','comparison_to_threshold_exact']},ensure_ascii=False,indent=2))
