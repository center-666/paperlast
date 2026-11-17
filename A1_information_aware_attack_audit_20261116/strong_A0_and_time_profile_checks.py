"""Conservative baseline supplement, not feedback-policy retuning.

1. Correct A0 P3 feasibility with the analytic population damage moments.
2. Test whether peak-bias gains are already explained by a public first-packet
   concentration, which is a deterministic member of the general A1-H class.
"""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import numpy as np
from scipy.linalg import eigh
import attack_models as am
from screen_attack_classes import evaluate,detectors
H=Path(__file__).resolve().parent;P=H.parent
def main():
    out={}
    for name in ('unstable','stable'):
        m=am.ref.load_model(name)
        mats=np.load(P/'active_trusted_measurement_G0_G1/batch_matrices.npz')
        S=mats[name+'__S_Gaussian'];W=mats[name+'__W'];candidates=[]
        for j,t in enumerate(np.linspace(0,2*np.pi,12,endpoint=False)):
            u=np.array([np.cos(t),np.sin(t)]);r=np.sqrt(.1/max(u@v@u for v in S))*(1+1e-12)
            xi=r*u
            if xi@W@xi>10*(1-2e-12):continue
            candidates.append(dict(id='A0_population_P3/ray'+str(j),family='A0',xi=xi.tolist(),
              theta=0.,kind='projected_feedback',population_peak_damage=float(max(xi@v@xi for v in S)),
              encoded_energy=float(xi@W@xi)))
        tr=am.old_sample_bank(m,range(16));dt=detectors(m,tr)
        rows={p['id']:dict(policy=p,metrics=evaluate(m,tr,p,dt)) for p in candidates}
        chosen={}
        for d in ('temporal','joint_GLRT'):
            p=min(rows.values(),key=lambda r:(r['metrics']['detectors'][d]['event_alarm_rate'],-r['metrics']['peak_mean_bias']))
            chosen[d]=p['policy']['id']
        b=am.old_sample_bank(m,range(16,64));db=detectors(m,b)
        validation={pid:dict(policy=rows[pid]['policy'],metrics=evaluate(m,b,rows[pid]['policy'],db)) for pid in set(chosen.values())}
        # Maximum EXPECTED first-packet damage, no observed-packet input.
        T=m.G[1].T@m.G[1];V=T+.25*(T+np.trace(T)*np.eye(2))
        ev,U=eigh(V);packet=np.sqrt(10)*(1-1e-12)*U[:,-1]
        pub=dict(id='public_first_packet',family='A1-H',xi=[0.,0.],theta=0.,
            kind='public_first_packet',packet=packet.tolist())
        old=json.loads((P/'active_trusted_measurement_G0_G1/batch_power_results.json').read_text('utf8'))['models'][name]['schemes']['B2']['P1']
        analytic=dict(id='A0_analytic_P1',family='A0',xi=old['xi'],theta=0.,kind='projected_feedback')
        out[name]=dict(analytic_A0_P1=dict(population_expected_peak=old['value'],policy=analytic,
              validation=evaluate(m,b,analytic,db)),
          population_P3_candidates=rows,population_P3_selected_using_training=chosen,
          population_P3_validation=validation,public_first_packet=dict(policy=pub,
            theoretical_expected_peak=float(10*ev[-1]),validation=evaluate(m,b,pub,db)),
          no_feedback_policy_reselected=True)
    result=dict(date='2026-11-17',scope='auditor baseline supplement after initial screen; same fixed12 directions and OLD sample splits',
      reason='training sample damage must not replace population A0 P3 feasibility; and temporal concentration must not be credited automatically to information',
      models=out,new_physical_paths=0,not_global_P3_probability_optimality=True)
    (H/'strong_baseline_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps({n:{'P3':{d:out[n]['population_P3_validation'][pid]['metrics']['detectors'][d] for d,pid in z['population_P3_selected_using_training'].items()},
         'public_first_packet':z['public_first_packet']} for n,z in out.items()},ensure_ascii=False))
if __name__=='__main__':main()
