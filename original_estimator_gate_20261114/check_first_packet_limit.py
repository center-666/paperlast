"""First-packet dangerous witness and oracle Neyman-Pearson limit.

Uses ONLY this round's explicitly declared original-architecture instance.
No Monte Carlo, schedule optimization, new detector, or old F/g numbers.
The theorem is analytic; DARE/SVD/normal CDF numbers are diagnostics.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import norm
H=Path(__file__).resolve().parent
src=H/'five_node_aggregate_model.json'
d=json.loads(src.read_text(encoding='utf-8'))
E,D,alpha,factor=10.,.1,.05,1.01
rows=[]
for name,row in d['models'].items():
    G=np.array(row['nodes'][1]['G'])
    U,s,Vh=np.linalg.svd(G)
    a=factor*np.sqrt(D)/s[0]*Vh[0]
    damage=float(np.linalg.norm(G@a)**2)
    energy=float(a@a)
    assert energy<E and damage>D
    nu=float(np.linalg.norm(a)/np.sqrt(.5))
    upper=float(norm.cdf(norm.ppf(alpha)+nu))
    assert 0<upper<1
    rows.append(dict(model=name,first_packet_attack=a.tolist(),
        autonomous_initial_state=(np.linalg.solve(np.array(d['H'][0]),a)).tolist(),
        actual_total_energy=energy,first_posterior_bias= (G@a).tolist(),
        squared_bias=damage,oracle_mean_shift_SNR=nu,
        oracle_max_detection_power_at_alpha=upper,
        dangerous_before_alarm_probability_lower=1-upper,
        attained_actual_detector_risk=False))
out=dict(date='2026-11-14',E=E,D=D,alpha_event=alpha,
    instance_sha256=hashlib.sha256(src.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode('utf-8')).hexdigest(),
    input_hash_format='UTF-8/LF normalized model text',
    allowed_class='single autonomous segment L=1 with total coded injection energy <=E',
    schemes='no watermark or scalar block +/-1 encoding, before any new integrity resource',
    timing='receiver alarms before including packet; on no alarm uses original full G',
    bias_reference='same-noise attack-free fixed-gain ungated shadow estimator; not independently alarm-gated healthy closed loop',
    oracle='receiver additionally knows physical x up to current time; attacked sensor current v~N(0,.5I) remains unknown',
    test='any randomized decision measurable from original receiver transcript, ideal secret and this oracle, H0 event probability <=.05',
    bound='power <= Phi(Phi^-1(alpha)+||a||/sqrt(.5)); all earlier alarm events may be included',
    limits=['not a computed exact worst event risk','not applicable unchanged to Gaussian matrix watermark',
            'not a bound for packet verification or a changed fusion rule','no global clean copy of attacked current sensor packet supplied',
            'floating displayed constants; analytic NP statement does not require DARE interval certificate'],
    rows=rows)
(H/'first_packet_limit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
