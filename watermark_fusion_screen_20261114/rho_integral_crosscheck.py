"""Reproduce selected-point integral checks and the single safety-derived cap.

No direction/budget reselection. --mode split is the most resolved published
diagnostic; these are NOT interval-certified integrals or a global optimizer.
"""
import argparse,json
import numpy as np
from rho_first_packet import H,integrate_risk,model_params,first_packet_metric
from rho_numeric import data,stationary,rho_moments
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['split','adaptive','high_order','safe','necessary'],default='split')
args=parser.parse_args()
if args.mode=='necessary':
    out={}
    for name in data['models']:
        p=model_params(name);v=stationary(p,*rho_moments(p['G'],.1,128,128))[0]
        out[name]=dict(necessary_cap_for_zero_first_packet_risk=.1,normal_trace_at_boundary=v,
            relative_loss=v/p['reference_trace']-1,
            note='necessary first-packet cap, not sufficient full-event cap; numerical normal-cost diagnostic')
    dest='rho_necessary_cap_point.json'
elif args.mode=='safe':
    out={}
    for name in data['models']:
        p=model_params(name);ref=p['reference_trace'];gamma=.09
        a=stationary(p,*rho_moments(p['G'],gamma,128,128))[0]
        r=gamma/np.linalg.norm(p['G'],2);b=stationary(p,r,r*r)[0]
        out[name]=dict(gamma_derived=gamma,candidate_trace=a,candidate_relative_loss=a/ref-1,
            sign_fixed_rho=r,sign_fixed_trace=b,sign_fixed_relative_loss=b/ref-1,reference_trace=ref,
            note='single theoretical sufficient-cap point, outside frozen normal budgets')
    dest='rho_safe_derived_point.json'
else:
    d=json.loads((H/'rho_first_packet_screen.json').read_text());out=[]
    for name,m in d['models'].items():
        p=model_params(name);C=first_packet_metric(p['A'])
        for row in m['rows']:
            for key,kind,param in [('Gaussian_fixed_rho','fixed',row['fixed_rho']),('Gaussian_aware','aware',row['gamma'])]:
                r=row[key]
                if args.mode=='split':v=integrate_risk(p['G'],C,r['direction_angle'],r['energy'],kind,param,192,80,True,True)
                elif args.mode=='adaptive':v=integrate_risk(p['G'],C,r['direction_angle'],r['energy'],kind,param,128,80,True)
                else:v=integrate_risk(p['G'],C,r['direction_angle'],r['energy'],kind,param,256,128)
                vk={'split':'risk_split_angle_adaptive_radial','adaptive':'risk_adaptive_radial','high_order':'risk_256'}[args.mode]
                dk={'split':'difference_from_uniform128','adaptive':'difference_from_fixed_rule_128','high_order':'difference_from_128'}[args.mode]
                out.append(dict(model=name,eta=row['eta'],scheme=key,**{vk:v,dk:v-r['risk']}))
                print(name,row['eta'],key,v,flush=True)
    dest={'split':'rho_first_packet_split_angle.json','adaptive':'rho_first_packet_adaptive_check.json','high_order':'rho_first_packet_high_order.json'}[args.mode]
(H/dest).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print('Wrote',dest)
